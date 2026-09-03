#!/usr/bin/env python3
"""
Stardew Valley Auto Fishing Bot - YOLO 版

检测:
  - UI 模板: 未锁定时全屏定位钓鱼 UI；连续几次稳定命中后锁定，
    锁定后只截取 UI 框区域给 YOLO，并按固定间隔复检 UI 是否还在
  - YOLO: 0=bar, 1=fish（与 yolo_dataset/data.yaml 一致）

控制:
  - 默认用 rl_fishing 训练的 RL 策略（观测为归一化位置/速度/进度/上一动作）
  - 无 RL 模型时回退到“速度预测 + 刹车时机 + PID 微调”

性能:
  - 自动 CUDA 推理（可用时 FP16），支持同目录 best.onnx
  - 暂停时不截图、不推理，几乎不占 CPU
  - Overlay 限频重绘

热键: F8 开始/暂停, F9 退出
依赖:
  pip install ultralytics opencv-python numpy mss pynput pywin32

权重优先级:
  1) yolo_runs/stardew_fish/weights/best.pt (或同目录 best.onnx)
  2) weights/best.pt
  3) yolov8n.pt (仅演示，请先训练)
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mss
from pynput.keyboard import Key, Controller, Listener

try:
    import win32api
    import win32con
    import win32gui
except ImportError:
    print("请安装 pywin32: pip install pywin32")
    sys.exit(1)

try:
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
except ImportError:
    print("请安装 ultralytics: pip install ultralytics")
    sys.exit(1)

USE_QUANTIZE_ARG = "quantize" in DEFAULT_CFG_DICT  # ultralytics>=8.4 用 quantize=16 表示 FP16

# ====================== 配置 ======================
# 启动默认控制模式（放在最上面，方便直接改）:
#   "simple" = 旧版上下判断（最稳的基线）
#   "pid"    = 速度预测 + PID
#   "rl"     = rl_fishing 训练的策略
# 命令行 --mode 或启动时菜单可以覆盖这里；运行中 F7 仍可切换。
CONTROL_MODE = "simple"

HOLD_KEY = "c"
SIMPLE_BAND = 12          # SIMPLE 迟滞带 px：按住中鱼低于中心 12px 才松，
                         # 松开中鱼高于中心 12px 才按，避免疯狂抖键
FPS_TARGET = 40           # 运行帧率上限
PAUSE_POLL_S = 0.05       # 暂停时轮询间隔（不截图，CPU 占用几乎为 0）

CONF = 0.25               # YOLO 置信度
IMGSZ = 640               # YOLO 输入尺寸

ROI_PAD = 60              # UI 框外扩，避免鱼/条贴边被裁
UI_DETECT_STABLE = 3      # 未锁定时连续几次检测到 UI 才锁定
UI_RECHECK_EVERY = 20     # 锁定后每隔多少帧复检一次 UI 是否还在
UI_MAX_MISS = 3           # 连续几次复检失败就解锁，重新全屏定位

FISH_MEMORY_FRAMES = 4    # 鱼短暂漏检时沿用上一帧位置，避免误松键
BAR_MEMORY_FRAMES = 4     # bar 短暂漏检时的容错
MAX_FISH_STEP_PX = 160.0  # 鱼相邻帧最大位移，超过视为误检(跳变)忽略
MAX_BAR_STEP_PX = 220.0   # bar 相邻帧最大位移，超过视为误检(跳变)忽略

# ====================== 控制参数（参考 stardew-valley-fishing-assistant） ======================
# 坐标约定: y 向下为正；e = 鱼 - bar 中心，鱼在上方时 e<0。
# 控制律: u = kp*e + kd*(v_fish - v_bar) + kff*v_fish
#         误差死区 -> 施密特迟滞 -> 最短切换，量化成 按住/松开
KP = 0.04                # 位置增益（参考项目 PD 参数）
KD = 0.18                # 阻尼（误差变化率）
KFF = 0.08               # 鱼速前馈
DEADBAND_PX = 2.0        # 误差死区
HYST_U = 0.08            # 控制量迟滞
MIN_SWITCH_S = 0.001     # 最短切换间隔（参考默认几乎为 0，靠迟滞防抖）
SMOOTH_ALPHA = 0.50      # 位置低通
LARGE_ERR_PX = 55.0      # 大误差阈值
LARGE_ERR_BOOST = 1.5    # 大误差时控制量放大
VEL_WINDOW_S = 0.15      # bar 速度估计窗口：太大会让 PID 慢半拍
FISH_WINDOW_S = 0.15     # 鱼速度估计窗口
VEL_ALPHA = 0.80         # 速度低通滤波系数（越大越跟手、噪声越大）
MAX_SPEED = 1600.0       # 速度估计上限，防止单帧抖动带偏
MIN_HOLD_S = 0.05        # 每次按键最短按住时间（帧间不抖键）
MIN_RELEASE_S = 0.02     # 每次松开后的最短冷却

# ====================== RL 集成 ======================
# 用 rl_fishing 里训练好的策略做按键决策（取代 PID）。
# 观测与训练环境一致：[bar_y, bar_vel, fish_y, fish_vel, progress, prev_action]，
# 位置按 UI 轨道高度归一化，速度按“虚拟池塘 568px、60Hz”换算。
USE_RL = True
RL_PROGRESS = 0.30        # 暂未读游戏进度条，先用训练初始值附近估计
RL_POND = 568.0           # 与 rl_fishing/env.py 的 POND 保持一致
RL_VEL_SCALE = 6.0        # 与 rl_fishing/env.py 的 VEL_SCALE 保持一致
RL_TICK_HZ = 60.0
RL_VEL_FACTOR = 0.35      # 真实游戏 bar 比训练环境快，先缩小速度避免观测饱和(0.35 可调)

START_HOTKEY = Key.f8
STOP_HOTKEY = Key.f9
TOGGLE_HOTKEY = Key.f7    # 运行时切换 RL / PID

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"

# ====================== 全局 ======================
keyboard = Controller()
running = False
exit_flag = False
hwnd_overlay = None
screen_w = screen_h = 0

model = None
device = "cpu"
is_torch_model = True
engine_name = "CPU"
tpl_ui = None
has_rl = False            # 是否成功加载 RL 模型
ctrl_mode = 2             # F7 循环: 0=RL 1=PID 2=SIMPLE；默认 SIMPLE 便于回退


def on_press(key):
    global running, exit_flag, ctrl_mode
    try:
        if key == START_HOTKEY:
            running = not running
            print(f"[{'RUNNING' if running else 'PAUSED'}]")
        elif key == STOP_HOTKEY:
            exit_flag = True
            print("退出中...")
            return False
        elif key == TOGGLE_HOTKEY:
            for _ in range(3):          # 循环切换并跳过不可用的 RL
                ctrl_mode = (ctrl_mode + 1) % 3
                if ctrl_mode == 0 and not has_rl:
                    continue
                break
            names = {0: "RL", 1: "PID", 2: "SIMPLE"}
            print(f"控制模式: {names[ctrl_mode]}")
    except Exception:
        pass


def ctrl_mode_name():
    names = {0: "RL", 1: "PID", 2: "SIMPLE"}
    return names.get(ctrl_mode, "?")


def find_weights():
    for folder in (
        SCRIPT_DIR / "yolo_runs" / "stardew_fish" / "weights",
        SCRIPT_DIR / "weights",
        SCRIPT_DIR,
    ):
        pt = folder / "best.pt"
        if pt.exists():
            return pt
        onnx = folder / "best.onnx"
        if onnx.exists():
            return onnx
    return None


def load_models():
    global model, device, is_torch_model, engine_name, tpl_ui
    w = find_weights()
    if w is None:
        w = SCRIPT_DIR / "yolov8n.pt"
        print("未找到训练权重，使用 yolov8n.pt 演示（请先训练）")
    else:
        print("加载权重:", w)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(w))
    is_torch_model = w.suffix.lower() == ".pt"

    if is_torch_model:
        model.to(device)
        model.fuse()
        if device != "cpu":
            engine_name = "CUDA"
            print(f"推理设备: GPU {torch.cuda.get_device_name(0)} (FP16)")
        else:
            print("推理设备: CPU（未检测到 CUDA，帧率会较低）")
    else:
        engine_name = "ONNX"
        print(f"推理设备: ONNX ({'CUDA' if device != 'cpu' else 'CPU'})")

    ui_full = TEMPLATE_DIR / "ui_full.png"
    if ui_full.exists():
        tpl_ui = cv2.imread(str(ui_full), cv2.IMREAD_GRAYSCALE)
        print("UI模板:", ui_full)
    else:
        print("未找到 UI 模板，将退化为全屏 YOLO")


# ====================== 截屏 ======================
def grab_screen(sct, rect=None):
    """截取全屏或指定区域(屏幕坐标)。返回 (BGR frame, 左上角屏幕坐标)。"""
    mon = sct.monitors[1]
    if rect is not None:
        x, y, w, h = rect
        left = max(mon["left"], int(x))
        top = max(mon["top"], int(y))
        right = min(mon["left"] + mon["width"], int(x + w))
        bottom = min(mon["top"] + mon["height"], int(y + h))
        if right - left < 80 or bottom - top < 80:
            reg = mon
        else:
            reg = {"left": left, "top": top, "width": right - left, "height": bottom - top}
    else:
        reg = mon

    shot = sct.grab(reg)
    frame = cv2.cvtColor(np.asarray(shot), cv2.COLOR_BGRA2BGR)
    return frame, reg["left"], reg["top"]


# ====================== UI 模板定位 ======================
def locate_ui(frame):
    """在 frame(局部坐标) 内定位 UI 模板，返回 (x, y, w, h) 或 None。
    大图(全屏)先缩小再匹配以提速；小图(ROI)直接匹配。
    """
    if tpl_ui is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, tw = tpl_ui.shape[:2]
    h, w = gray.shape[:2]

    fx = 1.0
    if w > 1000:  # 全屏搜索：先缩到 1000 宽以内（粗定位，锁定后 ROI 复检会精确修正）
        fx = 1000.0 / w
        gray = cv2.resize(gray, None, fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
        h, w = gray.shape[:2]

    best = None
    for s in (0.6, 0.8, 1.0, 1.25, 1.5, 1.8):
        nw = max(20, int(tw * s * fx))
        nh = max(20, int(th * s * fx))
        if nw >= w or nh >= h:
            continue
        tpl = cv2.resize(tpl_ui, (nw, nh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)
        if best is None or max_v > best[0]:
            best = (max_v, max_l[0] / fx, max_l[1] / fx, tw * s, th * s)

    if best is None or best[0] < 0.45:
        return None
    _, x, y, rw, rh = best
    return (int(x), int(y), int(rw), int(rh))


# ====================== Overlay ======================
def create_overlay():
    global hwnd_overlay, screen_w, screen_h
    screen_w = win32api.GetSystemMetrics(0)
    screen_h = win32api.GetSystemMetrics(1)
    wc = win32gui.WNDCLASS()
    wc.hInstance = win32api.GetModuleHandle(None)
    wc.lpszClassName = "StardewFishOverlay"
    wc.lpfnWndProc = win32gui.DefWindowProc
    wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
    try:
        win32gui.RegisterClass(wc)
    except win32gui.error:
        pass
    ex = (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT |
          win32con.WS_EX_TOPMOST | win32con.WS_EX_TOOLWINDOW)
    hwnd_overlay = win32gui.CreateWindowEx(
        ex, "StardewFishOverlay", "Stardew YOLO",
        win32con.WS_POPUP, 0, 0, screen_w, screen_h,
        None, None, wc.hInstance, None
    )
    win32gui.SetLayeredWindowAttributes(
        hwnd_overlay, win32api.RGB(0, 0, 0), 255, win32con.LWA_COLORKEY
    )
    win32gui.ShowWindow(hwnd_overlay, win32con.SW_SHOW)
    print(f"Overlay {screen_w}x{screen_h}")


def update_overlay(info):
    if not hwnd_overlay:
        return
    hdc = win32gui.GetDC(hwnd_overlay)
    try:
        br = win32gui.CreateSolidBrush(win32api.RGB(0, 0, 0))
        win32gui.FillRect(hdc, (0, 0, screen_w, screen_h), br)
        win32gui.DeleteObject(br)

        def rect(x, y, w, h, color, t=2):
            pen = win32gui.CreatePen(win32con.PS_SOLID, t, color)
            op = win32gui.SelectObject(hdc, pen)
            ob = win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
            win32gui.Rectangle(hdc, int(x), int(y), int(x + w), int(y + h))
            win32gui.SelectObject(hdc, op)
            win32gui.SelectObject(hdc, ob)
            win32gui.DeleteObject(pen)

        def circle(cx, cy, r, color, t=2):
            pen = win32gui.CreatePen(win32con.PS_SOLID, t, color)
            op = win32gui.SelectObject(hdc, pen)
            ob = win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))
            win32gui.Ellipse(hdc, int(cx - r), int(cy - r), int(cx + r), int(cy + r))
            win32gui.SelectObject(hdc, op)
            win32gui.SelectObject(hdc, ob)
            win32gui.DeleteObject(pen)

        def text(x, y, s, color):
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                win32gui.SetTextColor(hdc, win32api.RGB(0, 0, 0))
                win32gui.ExtTextOut(hdc, int(x + dx), int(y + dy), 0, None, s)
            win32gui.SetTextColor(hdc, color)
            win32gui.ExtTextOut(hdc, int(x), int(y), 0, None, s)

        st = info.get("status", "PAUSED")
        col = win32api.RGB(0, 255, 0) if st == "RUNNING" else win32api.RGB(255, 80, 80)
        fps = info.get("fps", 0)
        tag = f"  {fps:.0f} FPS" if fps > 0 else ""
        mode = info.get("mode", "?")
        text(16, 16,
             f"[F8] {st}  [F7] {mode}  [F9] Quit  {engine_name}{tag}", col)

        if info.get("dbg"):
            text(16, 40, info["dbg"], win32api.RGB(255, 255, 255))

        if info.get("ui"):
            x, y, w, h = info["ui"]
            rect(x, y, w, h, win32api.RGB(0, 255, 255), 2)

        if info.get("bar_box"):
            x1, y1, x2, y2 = info["bar_box"]
            rect(x1, y1, x2 - x1, y2 - y1, win32api.RGB(0, 255, 0), 2)
            text(x2 + 4, y1, "BAR", win32api.RGB(0, 255, 0))

        if info.get("fish_pt"):
            fx, fy = info["fish_pt"]
            circle(fx, fy, 8, win32api.RGB(80, 160, 255), 3)
            text(fx + 10, fy - 8, "FISH", win32api.RGB(80, 160, 255))
    finally:
        win32gui.ReleaseDC(hwnd_overlay, hdc)


def destroy_overlay():
    global hwnd_overlay
    if hwnd_overlay:
        win32gui.DestroyWindow(hwnd_overlay)
        hwnd_overlay = None


# ====================== YOLO 检测 ======================
def yolo_detect(frame, imgsz):
    """对整张 frame 跑 YOLO，返回 (fish_pt, bar_box)，坐标为局部图像坐标。"""
    if frame is None or frame.size == 0:
        return None, None

    kwargs = dict(
        verbose=False,
        conf=CONF,
        imgsz=imgsz,
        device=device,
        max_det=4,
    )
    if is_torch_model and device != "cpu":
        if USE_QUANTIZE_ARG:
            kwargs["quantize"] = 16   # FP16
        else:
            kwargs["half"] = True

    results = model.predict(frame, **kwargs)
    if not results:
        return None, None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    best_fish = None
    best_bar = None
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
        if cls == 1:            # fish
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if best_fish is None or conf > best_fish[2]:
                best_fish = (cx, cy, float(conf))
        elif cls == 0:          # bar
            if best_bar is None or conf > best_bar[4]:
                best_bar = (x1, y1, x2, y2, float(conf))

    fish_pt = (best_fish[0], best_fish[1]) if best_fish else None
    bar_box = best_bar[:4] if best_bar else None
    return fish_pt, bar_box


# ====================== PID 控制器 ======================
def _linfit_slope(points):
    """对 (t, y) 点列做速度拟合，返回斜率(px/s)。
    只有 2 个点时直接用两点差分（最新一帧，滞后最小）。"""
    n = len(points)
    if n < 2:
        return None
    if n == 2:
        dt = points[1][0] - points[0][0]
        if dt <= 1e-4:
            return None
        return (points[1][1] - points[0][1]) / dt
    sx = sy = sxx = sxy = 0.0
    t0 = points[0][0]
    for t, y in points:
        x = t - t0
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
    den = n * sxx - sx * sx
    if den <= 1e-9:
        return None
    return (n * sxy - sx * sy) / den


class FishPID:
    """参考 stardew-valley-fishing-assistant 的 PD + 迟滞控制器。

    - 位置先低通（SMOOTH_ALPHA），速度由最新帧差分估计
    - u = kp*e + kd*(v_fish - v_bar) + kff*v_fish
    - e<0(鱼在上) → u<0 → 按住；经过死区/迟滞/最短切换防抖
    """

    def __init__(self):
        self.bar_hist = deque(maxlen=20)
        self.fish_hist = deque(maxlen=14)
        self.last_t = None
        self.v_bar = 0.0      # 估计 bar 速度，向下为正 px/s
        self.v_fish = 0.0     # 估计鱼速度
        self.last_action = None   # True=按住 / False=松开
        self.err = 0.0
        self.target_v = 0.0
        self.fish_filt = None
        self.bar_filt = None
        self.last_switch_t = 0.0

    def reset(self):
        self.bar_hist.clear()
        self.fish_hist.clear()
        self.last_t = None
        self.v_bar = 0.0
        self.v_fish = 0.0
        self.last_action = None
        self.fish_filt = None
        self.bar_filt = None
        self.last_switch_t = 0.0

    @staticmethod
    def _push(hist, max_age, t, y):
        hist.append((t, y))
        while len(hist) > 1 and t - hist[0][0] > max_age:
            hist.popleft()

    @staticmethod
    def _smooth(old, raw):
        v = old + VEL_ALPHA * (raw - old)
        return max(-MAX_SPEED, min(MAX_SPEED, v))

    def observe(self, t, bar_cy, fish_y, bar_seen, fish_seen):
        """只更新位置/速度估计（RL 模式也用它构造观测）。返回本帧 dt。"""
        if self.last_t is None or t - self.last_t > 0.5:
            self.reset()
            self.last_t = t
        dt = t - self.last_t
        self.last_t = t
        dt = max(1e-4, min(dt, 0.2))

        if bar_seen:
            self._push(self.bar_hist, VEL_WINDOW_S, t, bar_cy)
        if fish_seen:
            self._push(self.fish_hist, FISH_WINDOW_S, t, fish_y)

        raw = _linfit_slope(self.bar_hist)
        if raw is not None:
            self.v_bar = self._smooth(self.v_bar, raw)
        raw = _linfit_slope(self.fish_hist)
        if raw is not None:
            self.v_fish = self._smooth(self.v_fish, raw)
        return dt

    def control(self, t, bar_cy, fish_y, bar_seen, fish_seen,
                bar_h=0.0, top=0.0, bottom=0.0):
        """每帧调用一次。返回 True=按住 / False=松开。"""
        self.observe(t, bar_cy, fish_y, bar_seen, fish_seen)

        if not bar_seen or not fish_seen:   # 与参考一致：丢一帧就松开
            self.last_action = False
            return False

        # 位置低通（参考 smooth_alpha=0.5）
        if self.fish_filt is None:
            self.fish_filt = float(fish_y)
        else:
            self.fish_filt = (SMOOTH_ALPHA * fish_y +
                              (1 - SMOOTH_ALPHA) * self.fish_filt)
        if self.bar_filt is None:
            self.bar_filt = float(bar_cy)
        else:
            self.bar_filt = (SMOOTH_ALPHA * bar_cy +
                             (1 - SMOOTH_ALPHA) * self.bar_filt)

        e = self.fish_filt - self.bar_filt
        e_p = e if abs(e) >= DEADBAND_PX else 0.0
        u = (KP * e_p + KD * (self.v_fish - self.v_bar) +
             KFF * self.v_fish)
        if LARGE_ERR_PX > 0 and abs(e) > LARGE_ERR_PX:
            u *= LARGE_ERR_BOOST
        self.err = e
        self.target_v = u

        prev = self.last_action if self.last_action is not None else False
        if u < -HYST_U:
            want = True
        elif u > HYST_U:
            want = False
        else:
            want = prev
        # 最短切换间隔
        if want != prev:
            if t - self.last_switch_t < MIN_SWITCH_S:
                want = prev
            else:
                self.last_switch_t = t
        self.last_action = want
        return want


# ====================== RL 策略（替代 PID 决策） ======================
class RLPolicy:
    """加载 rl_fishing/models 下的 PPO 策略。"""

    def __init__(self, path):
        from stable_baselines3 import PPO

        self.model = PPO.load(str(path), device="cpu")

    def decide(self, obs):
        act = self.model.predict(obs.reshape(1, -1), deterministic=True)[0][0]
        return bool(int(act))   # True=按住


def load_rl_policy():
    global has_rl
    if not USE_RL:
        print("USE_RL=False，控制模式: PID")
        has_rl = False
        return None
    model_dir = SCRIPT_DIR / "rl_fishing" / "models"
    best = model_dir / "best_model.zip"
    path = best if best.exists() else model_dir / "final_model.zip"
    if not path.exists():
        print("未找到 RL 模型，控制模式: PID")
        has_rl = False
        return None
    try:
        rl = RLPolicy(path)
        print(f"控制模式: RL（{path}）")
        has_rl = True
        return rl
    except Exception as exc:
        print(f"RL 模型加载失败，控制模式: PID（{exc}）")
        has_rl = False
        return None


def rl_make_obs(fish_y, bar_cy, ui_rect, pid, prev_action):
    """构造与训练环境一致的 6 维观测（真实像素 -> 虚拟池塘归一化）。"""
    x, y, w, h = ui_rect
    th = max(1.0, float(h))
    pos_f = float(np.clip((fish_y - y) / th, 0.0, 1.0))
    pos_b = float(np.clip((bar_cy - y) / th, 0.0, 1.0))
    # 真实速度 px/s -> 虚拟速度(px/tick)，再按 VEL_SCALE 归一化；
    # RL_VEL_FACTOR 用于补偿真实游戏与训练环境的绝对速度差异
    k = RL_POND / (th * RL_TICK_HZ * RL_VEL_SCALE) * RL_VEL_FACTOR
    vel_f = float(np.clip(pid.v_fish * k, -1.0, 1.0))
    vel_b = float(np.clip(pid.v_bar * k, -1.0, 1.0))
    return np.array(
        [pos_b, vel_b, pos_f, vel_f, RL_PROGRESS, float(prev_action)],
        dtype=np.float32,
    )


# ====================== 主循环 ======================
def main():
    global running, exit_flag, ctrl_mode

    ap = argparse.ArgumentParser(description="星露谷自动钓鱼 - YOLO 版")
    ap.add_argument(
        "--mode",
        choices=("rl", "pid", "simple"),
        default=CONTROL_MODE,
        help="启动时的控制模式；运行中仍可用 F7 切换",
    )
    args = ap.parse_args()

    print("=" * 52)
    print("  星露谷自动钓鱼 - YOLO 版")
    print("  F8 开始/暂停 | F7 切换 RL/PID/SIMPLE | F9 退出")
    print("=" * 52)

    load_models()
    print("预热模型（首次推理会初始化 CUDA/ONNX，稍等）...")
    yolo_detect(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), IMGSZ)
    create_overlay()
    listener = Listener(on_press=on_press)
    listener.start()

    is_holding = False
    hold_since = None       # 当前这轮按住开始时刻
    release_since = 0.0     # 上次松开时刻
    was_running = False
    pid = FishPID()
    rl = load_rl_policy()
    # 按启动参数预先选择模式（RL 不可用时自动回退 PID）
    mode_map = {"rl": 0, "pid": 1, "simple": 2}
    ctrl_mode = mode_map[args.mode]
    if ctrl_mode == 0 and not has_rl:
        print("RL 模型不可用，回退到 PID")
        ctrl_mode = 1
    mode_names = {0: "RL", 1: "PID", 2: "SIMPLE"}
    print(f"启动控制模式: {mode_names[ctrl_mode]}（F7 可切换）")
    dbg_log = []
    dbg_log_path = SCRIPT_DIR / "debug_log.csv"

    def flush_dbg_log():
        if not dbg_log:
            return
        first = not dbg_log_path.exists()
        with open(dbg_log_path, "a", encoding="utf-8", newline="") as f:
            if first:
                f.write(
                    "t,algo,fish_y,bar_y,vf_px_s,vb_px_s,pos_f,pos_b,"
                    "vel_f,vel_b,action,holding,ui_h,fish_seen,bar_seen\n"
                )
            for row in dbg_log:
                f.write(",".join(row) + "\n")
        dbg_log.clear()

    ui_rect = None       # 屏幕坐标 (x, y, w, h)；None = 未锁定，需全屏找 UI
    ui_hits = 0          # 未锁定阶段的连续命中次数
    ui_miss = 0          # 锁定阶段复检失败次数
    frame_idx = 0

    last_fish = None     # 屏幕坐标 (x, y)，供漏检时沿用
    last_bar = None      # 屏幕坐标 (x1, y1, x2, y2)
    fish_mem = 0
    bar_mem = 0

    last_paint = 0.0
    fps_frames = 0
    stats_t = time.perf_counter()

    try:
        with mss.MSS() as sct:
            while not exit_flag:
                # ---------- 暂停：不截图、不推理 ----------
                if not running:
                    if was_running:
                        pid.reset()          # 回来时速度历史已失效
                        was_running = False
                    if is_holding:
                        keyboard.release(HOLD_KEY)
                        is_holding = False
                        hold_since = None
                        release_since = time.perf_counter()
                    if time.perf_counter() - last_paint >= 0.2:
                        update_overlay({"status": "PAUSED",
                                        "mode": ctrl_mode_name()})
                        last_paint = time.perf_counter()
                        win32gui.PumpWaitingMessages()
                    time.sleep(PAUSE_POLL_S)
                    continue

                t0 = time.perf_counter()
                was_running = True
                frame_idx += 1

                # ---------- 截图范围 ----------
                if ui_rect is None:
                    # 未锁定：全屏截图，同时找 UI
                    frame, ox, oy = grab_screen(sct, None)
                    do_ui_check = True
                else:
                    # 已锁定：只截 UI 框(外扩)内的画面
                    x, y, w, h = ui_rect
                    frame, ox, oy = grab_screen(
                        sct, (x - ROI_PAD, y - ROI_PAD, w + ROI_PAD * 2, h + ROI_PAD * 2)
                    )
                    # 定期复检 UI 是否还在
                    do_ui_check = (frame_idx % UI_RECHECK_EVERY == 0)

                # ---------- UI 模板：定位/锁定/复检 ----------
                if tpl_ui is not None and do_ui_check:
                    hit = locate_ui(frame)
                    if hit is not None:
                        hx, hy, hw, hh = hit
                        hit_screen = (ox + hx, oy + hy, hw, hh)
                        if ui_rect is None:
                            ui_hits += 1
                            if ui_hits >= UI_DETECT_STABLE:
                                ui_rect = hit_screen
                                ui_hits = 0
                                print("UI 已锁定，只截取 UI 区域")
                        else:
                            ui_rect = hit_screen   # 位置有漂移时顺带修正
                            ui_miss = 0
                    else:
                        if ui_rect is None:
                            ui_hits = 0
                        else:
                            ui_miss += 1
                            if ui_miss >= UI_MAX_MISS:
                                ui_rect = None
                                ui_miss = 0
                                print("UI 丢失，重新全屏定位")

                # ---------- YOLO：锁定后 frame 已是 UI 区域 ----------
                fish_pt, bar_box = yolo_detect(frame, IMGSZ)

                # ---------- 检测稳定性：短时记忆 + 跳变过滤 ----------
                fish_seen_now = False
                if fish_pt is not None:
                    cand_fish = (ox + fish_pt[0], oy + fish_pt[1])
                    ok = (last_fish is None or
                          abs(cand_fish[1] - last_fish[1]) <= MAX_FISH_STEP_PX)
                    if ok:
                        last_fish = cand_fish
                        fish_seen_now = True
                        fish_mem = 0
                if not fish_seen_now:
                    fish_mem += 1
                    if fish_mem > FISH_MEMORY_FRAMES:
                        last_fish = None

                bar_seen_now = False
                if bar_box is not None:
                    bx1, by1, bx2, by2 = bar_box
                    cand_bar = (ox + bx1, oy + by1, ox + bx2, oy + by2)
                    old_cy = ((last_bar[1] + last_bar[3]) / 2
                              if last_bar is not None else None)
                    new_cy = (cand_bar[1] + cand_bar[3]) / 2
                    ok = (old_cy is None or abs(new_cy - old_cy) <= MAX_BAR_STEP_PX)
                    if ok:
                        last_bar = cand_bar
                        bar_seen_now = True
                        bar_mem = 0
                if not bar_seen_now:
                    bar_mem += 1
                    if bar_mem > BAR_MEMORY_FRAMES:
                        last_bar = None

                # ---------- 按键控制（RL 或 PID） ----------
                ctrl_t = time.perf_counter()
                ctrl_tag = "PID"
                if last_fish is not None and last_bar is not None:
                    bar_cy = (last_bar[1] + last_bar[3]) / 2
                    if ctrl_mode == 2:
                        ctrl_tag = "SIMPLE"
                        bar_ctr = (last_bar[1] + last_bar[3]) / 2
                        if is_holding:
                            # 按住中：鱼落到中心下方超过带宽才松开
                            act = last_fish[1] > bar_ctr + SIMPLE_BAND
                        else:
                            # 松开中：鱼升到中心上方超过带宽才按住
                            act = last_fish[1] < bar_ctr - SIMPLE_BAND
                    elif rl is not None and ui_rect is not None and ctrl_mode == 0:
                        ctrl_tag = "RL"
                        pid.observe(
                            ctrl_t,
                            bar_cy,
                            last_fish[1],
                            bar_seen=bar_seen_now,
                            fish_seen=fish_seen_now,
                        )
                        obs = rl_make_obs(
                            last_fish[1], bar_cy, ui_rect, pid, int(is_holding)
                        )
                        act = rl.decide(obs)
                        pid.err = last_fish[1] - bar_cy     # 仅用于显示
                        pid.target_v = 0.0
                    else:
                        ctrl_tag = "PID"
                        bar_h = max(10.0, last_bar[3] - last_bar[1])
                        if ui_rect is not None:
                            top, bottom = ui_rect[1], ui_rect[1] + ui_rect[3]
                        else:
                            top, bottom = 0.0, float(screen_h)
                        act = pid.control(
                            ctrl_t,
                            bar_cy,
                            last_fish[1],
                            bar_seen=bar_seen_now,
                            fish_seen=fish_seen_now,
                            bar_h=bar_h,
                            top=top,
                            bottom=bottom,
                        )
                    # 通用调试日志（任意模式都会记录）
                    if ui_rect is not None:
                        th = max(1.0, float(ui_rect[3]))
                        k = (RL_POND / (th * RL_TICK_HZ * RL_VEL_SCALE)
                             * RL_VEL_FACTOR)
                        pos_f = np.clip((last_fish[1] - ui_rect[1]) / th, 0, 1)
                        pos_b = np.clip((bar_cy - ui_rect[1]) / th, 0, 1)
                        vel_f = np.clip(pid.v_fish * k, -1, 1)
                        vel_b = np.clip(pid.v_bar * k, -1, 1)
                    else:
                        pos_f = pos_b = vel_f = vel_b = 0.0
                    dbg_log.append(
                        [
                            f"{ctrl_t:.3f}",
                            ctrl_tag,
                            f"{last_fish[1]:.1f}",
                            f"{bar_cy:.1f}",
                            f"{pid.v_fish:.1f}",
                            f"{pid.v_bar:.1f}",
                            f"{float(pos_f):.3f}",
                            f"{float(pos_b):.3f}",
                            f"{float(vel_f):.3f}",
                            f"{float(vel_b):.3f}",
                            "1" if act else "0",
                            "1" if is_holding else "0",
                            f"{ui_rect[3]:.0f}" if ui_rect else "0",
                            "1" if fish_seen_now else "0",
                            "1" if bar_seen_now else "0",
                        ]
                    )
                    if len(dbg_log) >= 500:
                        flush_dbg_log()
                    if act is True:
                        if (not is_holding and
                                ctrl_t - release_since >= MIN_RELEASE_S):
                            keyboard.press(HOLD_KEY)
                            is_holding = True
                            hold_since = ctrl_t
                            release_since = None
                    elif act is False:
                        if (is_holding and hold_since is not None and
                                ctrl_t - hold_since >= MIN_HOLD_S):
                            keyboard.release(HOLD_KEY)
                            is_holding = False
                            hold_since = None
                            release_since = ctrl_t
                    # act is None: 保持当前按键状态，减少抖动
                elif is_holding:
                    keyboard.release(HOLD_KEY)
                    is_holding = False
                    hold_since = None
                    release_since = ctrl_t

                # ---------- Overlay（限频重绘） ----------
                now = time.perf_counter()
                if now - last_paint >= 0.05:
                    fps = 1.0 / max(1e-6, now - t0)
                    info = {
                        "status": "RUNNING",
                        "fps": fps,
                        "mode": ctrl_mode_name(),
                        "ui": ui_rect,
                        "dbg": (f"e={pid.err:+.0f} vb={pid.v_bar:+.0f} "
                                f"vf={pid.v_fish:+.0f} vt={pid.target_v:+.0f} "
                                f"key={'HOLD' if is_holding else 'rel '} "
                                f"fs={int(fish_seen_now)} bs={int(bar_seen_now)}"),
                    }
                    if fish_pt is not None:
                        info["fish_pt"] = (ox + fish_pt[0], oy + fish_pt[1])
                    if bar_box is not None:
                        info["bar_box"] = last_bar
                    update_overlay(info)
                    last_paint = now
                    win32gui.PumpWaitingMessages()

                # ---------- 每 2 秒打印实际 FPS ----------
                fps_frames += 1
                if now - stats_t >= 2.0:
                    fps = fps_frames / (now - stats_t)
                    tag = f" ROI={ui_rect[2]}x{ui_rect[3]}" if ui_rect else " FULL"
                    print(f"[{fps:5.1f} FPS]{tag}  {engine_name}")
                    fps_frames = 0
                    stats_t = now

                time.sleep(max(0.0, 1.0 / FPS_TARGET - (time.perf_counter() - t0)))
    finally:
        flush_dbg_log()
        if is_holding:
            keyboard.release(HOLD_KEY)
        destroy_overlay()
        listener.stop()
        print("已退出")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stardew Valley Auto Fishing Bot

检测:
  - UI 模板: 未锁定时全屏定位钓鱼 UI；连续几次稳定命中后锁定，
    锁定后只截取 UI 框区域，并按固定间隔复检 UI 是否还在
  - mixed（默认）: 鱼用 YOLO，条用传统绿色矩形检测（缺条时回退 YOLO）

控制:
  - 参考 PD：u = kp*e + kd*(vf-vb) + kff*vf，死区/迟滞/最短切换

性能:
  - 自动 CUDA 推理（可用时 FP16），支持同目录 best.onnx
  - 暂停时不截图、不推理，几乎不占 CPU
  - Overlay 限频重绘

热键: F8 开始/暂停, F6 切换 CV/mixed/YOLO 检测, F9 退出
依赖:
  pip install ultralytics opencv-python numpy mss pynput pywin32

权重优先级:
  1) yolo_runs/stardew_fish/weights/best.pt (或同目录 best.onnx)
  2) weights/best.pt
  3) yolov8n.pt (仅演示，请先训练)
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import cv_detect
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
DETECT_MODE = "mixed"     # "mixed"=鱼YOLO+条传统(推荐)  "cv"/"yolo" 备用
DETECT_CHOICES = ("cv", "mixed", "yolo")

HOLD_KEY = "c"
FPS_TARGET = 40           # 运行帧率上限
PAUSE_POLL_S = 0.05       # 暂停时轮询间隔（不截图，CPU 占用几乎为 0）

CONF = 0.25               # YOLO 置信度
IMGSZ = 640               # YOLO 输入尺寸
IMGSZ_ROI = 416           # UI 锁定后的 YOLO 输入尺寸（更小=更快）

ROI_PAD = 60              # UI 框外扩，避免鱼/条贴边被裁
UI_RECHECK_EVERY = 40     # 锁定后每隔多少帧复检一次 UI 是否还在
UI_MIN_SCORE = 0.55       # UI 模板匹配阈值（调高更不容易误锁）
UI_FISH_TIMEOUT_S = 1.5   # UI 锁定后，鱼持续这么久检测不到就解锁

FISH_MEMORY_FRAMES = 4    # 鱼短暂漏检时沿用上一帧位置
BAR_MEMORY_FRAMES = 4     # bar 短暂漏检时的容错
MAX_FISH_STEP_PX = 160.0  # 鱼相邻帧最大位移，超过视为误检(跳变)忽略
MAX_BAR_STEP_PX = 220.0   # bar 相邻帧最大位移，超过视为误检(跳变)忽略

# ====================== 控制参数（参考 stardew-valley-fishing-assistant） ======================
# 坐标约定: y 向下为正；e = 鱼 - bar 中心，鱼在上方时 e<0。
# 控制律: u = kp*e + kd*(v_fish - v_bar) + kff*v_fish
#         误差死区 -> 施密特迟滞 -> 最短切换，量化成 按住/松开
KP = 0.03                # 位置增益（延迟鲁棒性扫描最优）
KD = 0.010               # 对 bar 自身速度的轻微阻尼（防过冲，不预测鱼）
KFF = 0.0                # 关闭鱼速前馈（鱼速抖动会把 bar 带到鱼不会去的地方）
DEADBAND_PX = 4.0        # 误差死区（延迟鲁棒性扫描最优）
HYST_U = 0.02            # 最小按压力度阈值
SMOOTH_ALPHA = 0.50      # 位置低通
LARGE_ERR_PX = 55.0      # 大误差阈值
LARGE_ERR_BOOST = 1.5    # 大误差时控制量放大
PDM_CYCLE_S = 0.36       # PDM 周期：按压力度在周期内转成占空比
PDM_GAIN = 0.60          # 控制量 → 按压力度的比例（gym 实测最优）
VEL_WINDOW_S = 0.15      # bar 速度估计窗口
FISH_WINDOW_S = 0.15     # 鱼速度估计窗口
VEL_ALPHA = 0.80         # 速度低通滤波系数
MAX_SPEED = 1600.0       # 速度估计上限，防止单帧抖动带偏
START_HOTKEY = Key.f8
STOP_HOTKEY = Key.f9
TOGGLE_DETECT_HOTKEY = Key.f6   # 运行时切换 cv / YOLO 检测

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
cv_fish_tpl = None
detect_mode = DETECT_MODE
yolo_engine_name = "CUDA"


def on_press(key):
    global running, exit_flag, detect_mode, engine_name
    try:
        if key == START_HOTKEY:
            running = not running
            print(f"[{'RUNNING' if running else 'PAUSED'}]")
        elif key == STOP_HOTKEY:
            exit_flag = True
            print("退出中...")
            return False
        elif key == TOGGLE_DETECT_HOTKEY:
            i = DETECT_CHOICES.index(detect_mode)
            detect_mode = DETECT_CHOICES[(i + 1) % len(DETECT_CHOICES)]
            labels = {"cv": "CV", "mixed": "CV+YOLO", "yolo": yolo_engine_name}
            engine_name = labels[detect_mode]
            print(f"检测模式: {detect_mode.upper()}")
    except Exception:
        pass


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
    global model, device, is_torch_model, engine_name, tpl_ui, cv_fish_tpl
    global yolo_engine_name

    # YOLO 与 CV 检测器都加载，F6 可随时切换
    cv_fish_tpl = cv_detect.load_fish_template(TEMPLATE_DIR / "fish.png")
    if cv_fish_tpl is not None:
        print("传统检测: fish 模板已加载")
    else:
        print("警告: templates/fish.png 不存在，CV 模式鱼将无法检测")

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
            yolo_engine_name = "CUDA"
            print(f"YOLO 推理: GPU {torch.cuda.get_device_name(0)} (FP16)")
        else:
            yolo_engine_name = "CPU"
            print("YOLO 推理: CPU（未检测到 CUDA，帧率会较低）")
    else:
        yolo_engine_name = "ONNX"
        print(f"YOLO 推理: ONNX ({'CUDA' if device != 'cpu' else 'CPU'})")
    labels = {"cv": "CV", "mixed": "CV+YOLO", "yolo": yolo_engine_name}
    engine_name = labels[detect_mode]

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

    if best is None or best[0] < UI_MIN_SCORE:
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
        text(16, 16, f"[F8] {st}  [F9] Quit  {engine_name}{tag}", col)

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
        max_det=2,
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


def detect_objects(frame, imgsz=IMGSZ):
    """按当前 detect_mode 在 frame(局部坐标) 上检测，返回 (fish_pt, bar_box)。"""
    if detect_mode == "cv":
        return cv_detect.detect(frame, cv_fish_tpl)
    yolo_fish, yolo_bar = yolo_detect(frame, imgsz)
    if detect_mode == "mixed":
        cv_bar = cv_detect.detect_bar(frame)
        return yolo_fish, (cv_bar if cv_bar is not None else yolo_bar)
    return yolo_fish, yolo_bar


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

    def reset(self):
        self.bar_hist.clear()
        self.fish_hist.clear()
        self.last_t = None
        self.v_bar = 0.0
        self.v_fish = 0.0
        self.last_action = None
        self.fish_filt = None
        self.bar_filt = None

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

    def control(self, t, bar_cy, fish_y, bar_seen, fish_seen):
        """每帧调用一次。返回 0~1 的按压力度（PDM 占空比）。"""
        self.observe(t, bar_cy, fish_y, bar_seen, fish_seen)

        if not bar_seen or not fish_seen:   # 与参考一致：丢一帧就松开
            return 0.0

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
        u = KP * e_p - KD * self.v_bar
        if LARGE_ERR_PX > 0 and abs(e) > LARGE_ERR_PX:
            u *= LARGE_ERR_BOOST
        self.err = e
        self.target_v = u

        # PDM：u<0(鱼在上方/需抬升) 时按压力度 = -u*GAIN，越不匹配按越久
        if u < -HYST_U:
            duty = min(1.0, -u * PDM_GAIN)
        else:
            duty = 0.0
        self.last_action = duty
        return duty


# ====================== 主循环 ======================
def main():
    global running, exit_flag

    print("=" * 52)
    print("  星露谷自动钓鱼 - YOLO 版")
    print("  F8 开始/暂停 | F6 切换检测 | F9 退出")
    print("=" * 52)

    load_models()
    print("预热模型（首次推理会初始化 CUDA/ONNX，稍等）...")
    yolo_detect(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), IMGSZ)
    create_overlay()
    listener = Listener(on_press=on_press)
    listener.start()

    is_holding = False
    was_running = False
    pid = FishPID()
    print(f"控制: PID   检测: {detect_mode.upper()}（F6 可切换）")
    dbg_log = []
    dbg_log_path = SCRIPT_DIR / "debug_log.csv"

    def flush_dbg_log():
        if not dbg_log:
            return
        first = not dbg_log_path.exists()
        with open(dbg_log_path, "a", encoding="utf-8", newline="") as f:
            if first:
                f.write(
                    "t,fish_y,bar_y,vf_px_s,vb_px_s,action,holding,"
                    "ui_h,fish_seen,bar_seen\n"
                )
            for row in dbg_log:
                f.write(",".join(row) + "\n")
        dbg_log.clear()

    ui_rect = None       # 屏幕坐标 (x, y, w, h)；None = 未锁定，需全屏找 UI
    fish_seen_at = None  # 最近一次检测到鱼的时间(monotonic)
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
                    if time.perf_counter() - last_paint >= 0.2:
                        update_overlay({"status": "PAUSED"})
                        last_paint = time.perf_counter()
                        win32gui.PumpWaitingMessages()
                    time.sleep(PAUSE_POLL_S)
                    continue

                t0 = time.perf_counter()
                was_running = True
                frame_idx += 1

                # ---------- 截图范围 ----------
                if ui_rect is None:
                    # 未锁定：全屏截图，先找可疑 UI
                    frame, ox, oy = grab_screen(sct, None)
                else:
                    # 已锁定：只截 UI 框(外扩)内的画面
                    x, y, w, h = ui_rect
                    frame, ox, oy = grab_screen(
                        sct, (x - ROI_PAD, y - ROI_PAD, w + ROI_PAD * 2, h + ROI_PAD * 2)
                    )
                do_ui_check = (frame_idx % UI_RECHECK_EVERY == 0)

                # ---------- 检测与 UI 锁定 ----------
                fish_pt = bar_box = None
                if ui_rect is None:
                    # 思路：全屏模板找可疑 UI -> 只对可疑框做检测 -> 发现鱼才锁定
                    hit = locate_ui(frame) if tpl_ui is not None else None
                    if hit is None:
                        # 无模板/没命中：退化为全图检测（不锁定）
                        fish_pt, bar_box = detect_objects(frame)
                    else:
                        hx, hy, hw, hh = hit
                        pad = 20
                        x1 = max(0, hx - pad)
                        y1 = max(0, hy - pad)
                        x2 = min(frame.shape[1], hx + hw + pad)
                        y2 = min(frame.shape[0], hy + hh + pad)
                        crop = frame[y1:y2, x1:x2]
                        f, b = detect_objects(crop)
                        if f is not None:
                            # 候选框里真有鱼 -> 锁这个 UI
                            ui_rect = (ox + hx, oy + hy, hw, hh)
                            print("UI 锁定（候选框内发现鱼）")
                            fish_pt = (x1 + f[0], y1 + f[1])
                            if b is not None:
                                bar_box = (x1 + b[0], y1 + b[1],
                                           x1 + b[2], y1 + b[3])
                        # 候选框内没鱼：不锁定，也不浪费全图检测
                else:
                    # 锁定后：整帧已经是 UI 区域，直接检测
                    fish_pt, bar_box = detect_objects(frame, IMGSZ_ROI)
                    if do_ui_check and tpl_ui is not None:
                        hit = locate_ui(frame)
                        if hit is not None:
                            ui_rect = (ox + hit[0], oy + hit[1],
                                       hit[2], hit[3])
                        # 复检没命中先不处理；鱼长时间消失才解锁

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
                        fish_seen_at = t0
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

                # UI 是否还在：鱼持续检测不到就算 UI 结束
                if (ui_rect is not None and fish_seen_at is not None and
                        time.perf_counter() - fish_seen_at > UI_FISH_TIMEOUT_S):
                    ui_rect = None
                    last_fish = None
                    last_bar = None
                    fish_mem = bar_mem = 0
                    print("UI 释放（长时间未检测到鱼）")

                # ---------- PID 按键控制 ----------
                ctrl_t = time.perf_counter()
                if last_fish is not None and last_bar is not None:
                    bar_cy = (last_bar[1] + last_bar[3]) / 2
                    act = pid.control(
                        ctrl_t,
                        bar_cy,
                        last_fish[1],
                        bar_seen=bar_seen_now,
                        fish_seen=fish_seen_now,
                    )
                    dbg_log.append(
                        [
                            f"{ctrl_t:.3f}",
                            f"{last_fish[1]:.1f}",
                            f"{bar_cy:.1f}",
                            f"{pid.v_fish:.1f}",
                            f"{pid.v_bar:.1f}",
                            "1" if act > 0 and (ctrl_t % PDM_CYCLE_S) < act * PDM_CYCLE_S else "0",
                            "1" if is_holding else "0",
                            f"{ui_rect[3]:.0f}" if ui_rect else "0",
                            "1" if fish_seen_now else "0",
                            "1" if bar_seen_now else "0",
                        ]
                    )
                    if len(dbg_log) >= 500:
                        flush_dbg_log()
                    # PDM：按压力度 act 在 PDM_CYCLE_S 周期内转成占空比
                    press_now = (act > 0 and
                                 (ctrl_t % PDM_CYCLE_S) < act * PDM_CYCLE_S)
                    if press_now and not is_holding:
                        keyboard.press(HOLD_KEY)
                        is_holding = True
                    elif not press_now and is_holding:
                        keyboard.release(HOLD_KEY)
                        is_holding = False
                elif is_holding:
                    keyboard.release(HOLD_KEY)
                    is_holding = False

                # ---------- Overlay（限频重绘） ----------
                now = time.perf_counter()
                if now - last_paint >= 0.10:
                    fps = 1.0 / max(1e-6, now - t0)
                    info = {
                        "status": "RUNNING",
                        "fps": fps,
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

#!/usr/bin/env python3
"""
Stardew Valley Auto Fishing Bot - YOLO 版

检测:
  - UI 模板: 未锁定时全屏定位钓鱼 UI；连续几次稳定命中后锁定，
    锁定后只截取 UI 框区域给 YOLO，并按固定间隔复检 UI 是否还在
  - YOLO: 0=bar, 1=fish（与 yolo_dataset/data.yaml 一致）

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

import sys
import time
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
HOLD_KEY = "c"
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

START_HOTKEY = Key.f8
STOP_HOTKEY = Key.f9

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


def on_press(key):
    global running, exit_flag
    try:
        if key == START_HOTKEY:
            running = not running
            print(f"[{'RUNNING' if running else 'PAUSED'}]")
        elif key == STOP_HOTKEY:
            exit_flag = True
            print("退出中...")
            return False
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
        text(16, 16, f"[F8] {st}  [F9] Quit  {engine_name}{tag}", col)

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


def should_hold(bar_box, fish_y):
    """鱼在条内 -> 不按键；鱼在条上方 -> 按住(条上升)；在下方 -> 松开(条下降)。"""
    if bar_box is None or fish_y is None:
        return False
    x1, y1, x2, y2 = bar_box
    top, bottom, center = y1, y2, (y1 + y2) / 2
    margin = max(3.0, (bottom - top) * 0.10)
    if top + margin <= fish_y <= bottom - margin:
        return False
    return fish_y < center


# ====================== 主循环 ======================
def main():
    global running, exit_flag

    print("=" * 52)
    print("  星露谷自动钓鱼 - YOLO 版")
    print("  F8 开始/暂停 | F9 退出")
    print("=" * 52)

    load_models()
    print("预热模型（首次推理会初始化 CUDA/ONNX，稍等）...")
    yolo_detect(np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8), IMGSZ)
    create_overlay()
    listener = Listener(on_press=on_press)
    listener.start()

    is_holding = False

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

                # ---------- 短时记忆：漏检 1~2 帧不丢位置 ----------
                if fish_pt is not None:
                    last_fish = (ox + fish_pt[0], oy + fish_pt[1])
                    fish_mem = 0
                else:
                    fish_mem += 1
                    if fish_mem > FISH_MEMORY_FRAMES:
                        last_fish = None

                if bar_box is not None:
                    bx1, by1, bx2, by2 = bar_box
                    last_bar = (ox + bx1, oy + by1, ox + bx2, oy + by2)
                    bar_mem = 0
                else:
                    bar_mem += 1
                    if bar_mem > BAR_MEMORY_FRAMES:
                        last_bar = None

                # ---------- 按键控制 ----------
                if last_fish is not None and last_bar is not None:
                    hold = should_hold(last_bar, last_fish[1])
                    if hold and not is_holding:
                        keyboard.press(HOLD_KEY)
                        is_holding = True
                    elif not hold and is_holding:
                        keyboard.release(HOLD_KEY)
                        is_holding = False
                elif is_holding:
                    keyboard.release(HOLD_KEY)
                    is_holding = False

                # ---------- Overlay（限频重绘） ----------
                now = time.perf_counter()
                if now - last_paint >= 0.05:
                    fps = 1.0 / max(1e-6, now - t0)
                    info = {
                        "status": "RUNNING",
                        "fps": fps,
                        "ui": ui_rect,
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
        if is_holding:
            keyboard.release(HOLD_KEY)
        destroy_overlay()
        listener.stop()
        print("已退出")


if __name__ == "__main__":
    main()

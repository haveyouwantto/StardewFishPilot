#!/usr/bin/env python3
"""
传统视觉检测（替代 YOLO）：
  - fish: 精确模板匹配（templates/fish.png）
  - bar:  UI 里的绿色矩形（HSV 颜色分割 + 形状筛选）

返回的坐标均为局部图像坐标，接口与 yolo_detect 一致：
  fish_pt = (cx, cy) 或 None
  bar_box = (x1, y1, x2, y2) 或 None
"""

from pathlib import Path

import cv2
import numpy as np

# ---------------- fish 模板匹配参数 ----------------
# fish.png 是 1:1 小模板(19x20)，实际画面里鱼可能被 UI 缩放放大数倍，
# 所以匹配时要按多档放大。若你的窗口缩放固定，可只保留对应档位提速。
FISH_SCALES = (0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0)
FISH_MIN_SCORE = 0.40          # 匹配分阈值，太低会误检

# ---------------- bar 绿色矩形参数 ----------------
GREEN_HSV_LOW = np.array([35, 80, 80], np.uint8)
GREEN_HSV_HIGH = np.array([95, 255, 255], np.uint8)
BAR_MIN_AREA = 800.0           # 最小面积(px²)
BAR_MIN_AREA_RATIO = 0.006     # 相对整帧的最小面积比
BAR_MAX_W_RATIO = 0.60         # 最大宽度占帧宽比例
BAR_ASPECT_MIN = 1.6           # 高/宽比下限（竖长绿条）
BAR_ASPECT_MAX = 10.0          # 上限（排除又高又窄的进度条）
BAR_Y_BAND_RATIO = 0.45        # 以鱼 y 为中心的搜索带（占帧高比例）


def load_fish_template(path):
    p = Path(path)
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def detect_fish(frame_bgr, tpl_bgr):
    """模板匹配鱼图标，返回 (cx, cy) 或 None。"""
    if tpl_bgr is None or frame_bgr is None or frame_bgr.size == 0:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    tpl = cv2.cvtColor(tpl_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = tpl.shape[:2]
    fh, fw = frame_bgr.shape[:2]

    best = None
    for s in FISH_SCALES:
        nw, nh = int(tw * s), int(th * s)
        if nw < 6 or nh < 6 or nw >= fw or nh >= fh:
            continue
        # 像素画模板放大用最近邻，保持锯齿轮廓，避免插值模糊掉特征
        interp = (cv2.INTER_NEAREST if nw > tw
                  else cv2.INTER_AREA)
        r = cv2.resize(tpl, (nw, nh), interpolation=interp)
        res = cv2.matchTemplate(gray, r, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_loc = cv2.minMaxLoc(res)
        if best is None or max_v > best[0]:
            best = (max_v, max_loc[0] + nw / 2, max_loc[1] + nh / 2)

    if best is None or best[0] < FISH_MIN_SCORE:
        return None
    return (float(best[1]), float(best[2]))


def detect_bar(frame_bgr, fish=None, band=None):
    """只在给定 y 附近找绿条（默认整帧搜索），返回 (x1,y1,x2,y2) 或 None。
    返回坐标始终是整帧坐标。"""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    fh, fw = frame_bgr.shape[:2]
    y0, y1 = 0, fh
    if fish is not None:
        center_y = fish[1]
        band = band if band is not None else max(80.0, fh * BAR_Y_BAND_RATIO)
        y0 = max(0, int(center_y - band))
        y1 = min(fh, int(center_y + band))
        if y1 - y0 < 40:
            y0, y1 = 0, fh
    search = frame_bgr[y0:y1]
    sh = y1 - y0
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(BAR_MIN_AREA, fw * sh * BAR_MIN_AREA_RATIO)
    best_area = 0.0
    best_box = None
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area < min_area or w <= 1 or h <= 1:
            continue
        if w > fw * BAR_MAX_W_RATIO:
            continue
        aspect = h / w
        if aspect < BAR_ASPECT_MIN or aspect > BAR_ASPECT_MAX:
            continue
        candidates.append((area, x, y, w, h))
        if fish is not None:
            # 硬约束：绿条垂直范围必须与鱼 y 有重叠（±60px），
            # 水平中心也必须接近鱼 x，排除进度条/侧边绿块
            y_overlap = (y - 60 <= fish[1] <= y + h + 60)
            x_close = abs((x + w / 2) - fish[0]) <= max(60.0, fw * 0.3)
            if not (y_overlap and x_close):
                continue
        if area > best_area:
            best_area = area
            best_box = (float(x), float(y0 + y), float(x + w),
                        float(y0 + y + h))
    if best_box is None and fish is not None and candidates:
        # 鱼离条太远时退而求其次：选垂直距离最近的绿块，保证还能追
        cx = fish[1]
        best_d = None
        best = None
        for area, x, y, w, h in candidates:
            dist = min(abs(cx - y), abs(cx - (y + h)))
            if best_d is None or dist < best_d:
                best_d = dist
                best = (x, y0 + y, x + w, y0 + y + h)
        best_box = best
    return best_box


def detect(frame_bgr, fish_tpl):
    """一次返回 (fish_pt, bar_box)，与 yolo_detect 输出一致。"""
    fish = detect_fish(frame_bgr, fish_tpl)
    bar = detect_bar(frame_bgr, fish=fish)
    return fish, bar

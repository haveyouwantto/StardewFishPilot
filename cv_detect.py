#!/usr/bin/env python3
"""
传统视觉检测（替代 YOLO）：
  - fish: 精确模板匹配（templates/fish.png）
  - bar:  只扫 fish 中心列上的连续绿色段

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

# ---------------- bar 检测参数（单列扫描） ----------------
GREEN_HSV_LOW = np.array([38, 90, 90], np.uint8)
GREEN_HSV_HIGH = np.array([82, 255, 255], np.uint8)


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


def detect_bar(frame_bgr, fish_box=None):
    """简化版：只扫 fish 中心那一列。

    沿该列从上到下找连续绿色段；相邻两段之间的空隙如果落在 fish 框内
    （鱼遮挡导致），就把上下段拼成一条。绿色段高度需不小于 fish 高度。
    bar 宽度直接沿用 fish 框宽度。返回整帧坐标 (x1,y1,x2,y2) 或 None。
    """
    if frame_bgr is None or frame_bgr.size == 0 or fish_box is None:
        return None
    fh, fw = frame_bgr.shape[:2]
    fx1, fy1, fx2, fy2 = [int(round(v)) for v in fish_box]
    fx1 = max(0, fx1)
    fy1 = max(0, fy1)
    fx2 = min(fw - 1, fx2)
    fy2 = min(fh - 1, fy2)
    if fx2 <= fx1 or fy2 <= fy1:
        return None

    col = max(0, min(fw - 1, int(round((fx1 + fx2) / 2))))
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    col_mask = cv2.inRange(hsv[:, col:col + 1, :],
                           GREEN_HSV_LOW, GREEN_HSV_HIGH)[:, 0]

    runs = []
    start = None
    for y in range(fh):
        if col_mask[y] > 0:
            if start is None:
                start = y
        else:
            if start is not None:
                runs.append((start, y - 1))
                start = None
    if start is not None:
        runs.append((start, fh - 1))

    fish_h = fy2 - fy1
    cands = []
    i = 0
    while i < len(runs):
        top, bot = runs[i]
        j = i
        while j + 1 < len(runs):
            gap_s = runs[j][1] + 1
            gap_e = runs[j + 1][0] - 1
            gap_len = gap_e - gap_s + 1
            # 只有空隙落在 fish 框内（鱼遮挡）才拼接上下两段
            if (0 < gap_len <= fish_h and gap_s <= fy2 and gap_e >= fy1):
                bot = runs[j + 1][1]
                j += 1
            else:
                break
        if bot - top + 1 >= fish_h:
            cands.append((top, bot))
        i = j + 1

    # 严格从上到下：取第一段高度达标的连续绿
    if not cands:
        return None
    top, bot = cands[0]
    return (float(fx1), float(top), float(fx2), float(bot))


def detect(frame_bgr, fish_tpl):
    """模板模式只输出 fish（bar 需要 YOLO fish 框，见 detect_bar）。"""
    return detect_fish(frame_bgr, fish_tpl), None

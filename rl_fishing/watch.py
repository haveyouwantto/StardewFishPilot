#!/usr/bin/env python3
"""
带调试仪表盘的观看/评估工具。

用法:
  uv run python rl_fishing/watch.py
  uv run python rl_fishing/watch.py --no-window --episodes 200
  uv run python rl_fishing/watch.py --height 700
"""

import argparse

import cv2
import numpy as np
from stable_baselines3 import PPO

from env import BEHAVIORS, FishingEnv, POND

# 深色仪表盘配色 (BGR)
BG = (26, 30, 38)
CARD = (40, 46, 56)
CARD_EDGE = (60, 68, 80)
TEXT = (235, 238, 242)
TEXT_DIM = (150, 158, 168)
ACCENT = (255, 170, 0)
GREEN = (60, 210, 120)
RED = (70, 70, 230)
CYAN = (235, 200, 0)
MAGENTA = (220, 90, 200)


def _text(img, x, y, s, color=TEXT, scale=0.52, thick=1, font=None):
    font = font or cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _card(img, x, y, w, h, fill=CARD, edge=CARD_EDGE):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), edge, 1)


def _label_value(img, x, y, label, value, vcolor=TEXT, vscale=0.52):
    _text(img, x, y, label, TEXT_DIM, 0.46)
    _text(img, x + 165, y, value, vcolor, vscale)


def _f(v):
    return f"{v:+.1f}"


def _progress_bar(img, x, y, w, h, ratio, good_color=GREEN, bad_color=RED):
    ratio = float(np.clip(ratio, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 24, 30), -1)
    color = good_color if ratio >= 0.5 else bad_color
    if ratio > 0:
        cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), CARD_EDGE, 1)


def _mini_history(img, x, y, results, w=16, h=16, gap=6):
    for i, win in enumerate(results[-12:]):
        px = x + i * (w + gap)
        cv2.rectangle(img, (px, y), (px + w, y + h),
                      GREEN if win else RED, -1)
        cv2.rectangle(img, (px, y), (px + w, y + h), BG, 1)


def build_dashboard(track_img, d: dict, run: dict) -> np.ndarray:
    """把钓鱼画面 + 完整调试信息拼成一张仪表盘。"""
    W, H = 940, 780
    img = np.full((H, W, 3), BG, np.uint8)
    status = run.get("status", "RUNNING")

    # ---------- 顶部标题 ----------
    _card(img, 12, 10, W - 24, 46, fill=(32, 38, 48))
    _text(img, 26, 40, "STARDEW FISHING - RL DASHBOARD", ACCENT, 0.68, 2)
    stcol = GREEN if status == "RUNNING" else (GREEN if status == "WIN" else RED)
    _text(img, W - 180, 40, f"[ {status} ]", stcol, 0.62, 2)

    # ---------- 左侧: 钓鱼池塘 ----------
    track_h = H - 74
    tw = int(track_img.shape[1] * track_h / track_img.shape[0])
    track = cv2.resize(track_img, (tw, track_h), interpolation=cv2.INTER_AREA)
    x0, y0 = 16, 66
    _card(img, x0 - 5, y0 - 5, tw + 10, track_h + 10, fill=CARD, edge=CARD_EDGE)
    img[y0:y0 + track_h, x0:x0 + tw] = track

    # ---------- 右侧面板 ----------
    px = x0 + tw + 24
    pw = W - px - 16

    # 当前局
    _card(img, px, 66, pw, 74)
    _text(img, px + 12, 90, f"EPISODE {run['ep'] + 1}", TEXT, 0.62, 2)
    _text(img, px + 12, 122,
          f"difficulty {d['difficulty']}   behavior {d['behavior']}   "
          f"bar {d['bar_size']:.0f}px", TEXT_DIM, 0.5)
    _text(img, px + pw - 120, 122,
          f"tick {d['steps']}  ({d['steps'] / 60:.1f}s)", TEXT_DIM, 0.48)

    y = 158
    _card(img, px, y, pw, 178)
    _text(img, px + 12, y + 24, "PROGRESS", TEXT_DIM, 0.5)
    ratio = d["progress"] / 1000.0
    _progress_bar(img, px + 12, y + 36, pw - 24, 16, ratio)
    _text(img, px + 12, y + 78,
          f"{d['progress']:.0f} / 1000   ({ratio * 100:.0f}%)",
          GREEN if ratio >= 0.5 else RED, 0.58)

    acc = run["in_ticks"] / max(1, run["ticks"])
    _label_value(img, px + 14, y + 118, "episode accuracy", f"{acc * 100:.1f}%",
                 ACCENT, 0.56)
    _label_value(img, px + 14, y + 152, "run win rate",
                 f"{run['wins']}/{run['eps']} ({run['win_rate'] * 100:.0f}%)",
                 GREEN if run["win_rate"] > 0.5 else TEXT, 0.56)

    # 位置 / 速度
    y += 194
    _card(img, px, y, pw, 190)
    _text(img, px + 12, y + 24, "STATE (px / px-per-tick)", TEXT_DIM, 0.5)
    bar_ctr = d["bar_pos"] + d["bar_size"] / 2
    err = bar_ctr - d["fish_pos"]           # >0: bar 中心在鱼下方
    _label_value(img, px + 14, y + 60, "fish y", f"{d['fish_pos']:6.1f} "
                 f"({d['fish_pos'] / POND * 100:4.1f}%)", CYAN)
    _label_value(img, px + 14, y + 94, "bar y", f"{bar_ctr:6.1f} "
                 f"({bar_ctr / POND * 100:4.1f}%)", GREEN)
    _label_value(img, px + 14, y + 128, "err (bar-fish)", f"{err:+7.1f}",
                 RED if abs(err) > 30 else TEXT)
    _label_value(img, px + 14, y + 162, "in bar", "YES" if d["in_bar"] else "NO",
                 GREEN if d["in_bar"] else RED)

    y2 = y
    _label_value(img, px + pw // 2 + 8, y2 + 60, "fish vel",
                 f"{d['fish_vel']:+.2f} ({_f(d['fish_vel'] * 60)}px/s)", CYAN)
    _label_value(img, px + pw // 2 + 8, y2 + 94, "bar vel",
                 f"{d['bar_vel']:+.2f} ({_f(d['bar_vel'] * 60)}px/s)", GREEN)
    _label_value(img, px + pw // 2 + 8, y2 + 128, "fish drift",
                 f"{d['fish_base_vel']:+.2f}", MAGENTA)
    _label_value(img, px + pw // 2 + 8, y2 + 162, "action",
                 "HOLD  [C]" if d["last_action"] else "release",
                 ACCENT if d["last_action"] else TEXT_DIM)

    # 按键统计 / 鱼目标
    y += 206
    _card(img, px, y, pw, 112)
    _text(img, px + 12, y + 24, "DEBUG", TEXT_DIM, 0.5)
    _label_value(img, px + 14, y + 58, "key switches", f"{run['keys']}", ACCENT)
    _label_value(img, px + pw // 2 + 8, y + 58, "fish target",
                 f"{d['fish_target']:.0f}" if d["fish_target"] >= 0 else "none",
                 MAGENTA)
    _label_value(img, px + 14, y + 92, "reward/sum", f"{run['reward']:+.2f}", TEXT)
    _label_value(img, px + pw // 2 + 8, y + 92, "hold ratio",
                 f"{run['hold_ticks'] / max(1, run['ticks']) * 100:.0f}%", TEXT)

    # 最近战绩
    _card(img, px, y + 128, pw, 62)
    _text(img, px + 12, y + 152, "recent results", TEXT_DIM, 0.5)
    _mini_history(img, px + 14, y + 164, run["results"])
    _text(img, px + pw - 190, y + 168,
          f"avg_acc {run['acc_sum'] / max(1, run['eps']):.1%}", TEXT, 0.46)
    _text(img, 16, H - 6, "ESC / Q quit", TEXT_DIM, 0.42)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rl_fishing/models/best_model.zip")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--difficulty", type=int, default=None)
    ap.add_argument("--behavior", type=str, default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--obs-noise", type=float, default=2.5)
    ap.add_argument("--control-hz", type=int, default=20)
    ap.add_argument("--latency-ticks", type=int, default=2)
    ap.add_argument("--no-window", action="store_true", help="不弹窗，纯统计")
    ap.add_argument("--height", type=int, default=780, help="仪表盘高度 px")
    args = ap.parse_args()
    if args.behavior is not None and args.behavior not in BEHAVIORS:
        ap.error(f"--behavior 可选 {BEHAVIORS}")

    model = PPO.load(args.model, device="cpu")
    env = FishingEnv(
        difficulty=args.difficulty,
        behavior=args.behavior,
        level=args.level,
        obs_noise=args.obs_noise,
        control_hz=args.control_hz,
        latency_ticks=args.latency_ticks,
        render_mode="rgb_array",
    )

    run = {
        "ep": 0, "ticks": 0, "in_ticks": 0, "keys": 0, "reward": 0.0,
        "hold_ticks": 0, "wins": 0, "eps": 0, "results": [], "acc_sum": 0.0,
        "win_rate": 0.0, "status": "RUNNING", "prev_action": 0,
    }
    win = "RL Fishing Dashboard"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 940, max(320, min(args.height, 940)))

    def reset_run():
        run["ticks"] = 0
        run["in_ticks"] = 0
        run["keys"] = 0
        run["reward"] = 0.0
        run["hold_ticks"] = 0
        run["status"] = "RUNNING"

    try:
        for ep in range(args.episodes):
            run["ep"] = ep
            reset_run()
            obs, _ = env.reset()
            done = False
            episode_acc = []
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(int(act))
                d = env.debug_state()
                run["ticks"] += 1
                run["reward"] += reward
                run["hold_ticks"] += int(d["last_action"])
                run["in_ticks"] += int(info["in_bar"])
                if int(act) != run["prev_action"]:
                    run["keys"] += 1
                    run["prev_action"] = int(act)
                episode_acc.append(info["in_bar"])
                done = term or trunc
                if done:
                    run["status"] = "WIN" if info["success"] else "LOSS"

                if not args.no_window:
                    frame = build_dashboard(env.render(), d, run)
                    cv2.imshow(win, frame)
                    key = cv2.waitKey(16) & 0xFF
                    if key in (27, ord("q")):
                        return
                    if done:
                        cv2.waitKey(700)

            # 结算
            run["eps"] += 1
            run["results"].append(info["success"])
            run["acc_sum"] += np.mean(episode_acc)
            run["wins"] += int(info["success"])
            run["win_rate"] = run["wins"] / run["eps"]
            print(
                f"ep{ep}: {'WIN ' if info['success'] else 'FAIL'} "
                f"difficulty={info['difficulty']} behavior={info['behavior']} "
                f"acc={np.mean(episode_acc):.2f} len={run['ticks']}"
            )
    finally:
        if not args.no_window:
            cv2.destroyAllWindows()

    if args.episodes > 1:
        print(f"\n共 {args.episodes} 局: win_rate={run['win_rate']:.2f} "
              f"avg_acc={run['acc_sum'] / max(1, run['eps']):.2f} "
              f"avg_len={run['ticks'] / max(1, run['eps']):.0f}")


if __name__ == "__main__":
    main()

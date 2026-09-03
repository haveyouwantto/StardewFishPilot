#!/usr/bin/env python3
"""
用训练好的策略观看/评估虚拟钓鱼。

用法:
  uv run python rl_fishing/watch.py                          # 默认 best_model.zip
  uv run python rl_fishing/watch.py --no-window --episodes 200
"""

import argparse

import cv2
import numpy as np
from stable_baselines3 import PPO

from env import BEHAVIORS, FishingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rl_fishing/models/best_model.zip")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--difficulty", type=int, default=None)
    ap.add_argument("--behavior", type=str, default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--no-window", action="store_true", help="不弹窗，纯统计")
    ap.add_argument("--height", type=int, default=640, help="窗口高度上限 px")
    args = ap.parse_args()
    if args.behavior is not None and args.behavior not in BEHAVIORS:
        ap.error(f"--behavior 可选 {BEHAVIORS}")

    model = PPO.load(args.model, device="cpu")
    env = FishingEnv(
        difficulty=args.difficulty,
        behavior=args.behavior,
        level=args.level,
        obs_noise=args.obs_noise,
        render_mode="rgb_array",
    )

    win = "RL Fishing (ESC quit)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # 保持池塘 66:579 的窄长比例，但限制高度，避免超出屏幕
        first_frame = env.render()
        disp_h = min(first_frame.shape[0], max(120, args.height))
        disp_w = max(1, round(first_frame.shape[1] * disp_h / first_frame.shape[0]))
        cv2.resizeWindow(win, disp_w, disp_h)

    wins = fails = 0
    accs = []
    lens = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            acc = []
            steps = 0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(int(act))
                done = term or trunc
                acc.append(info["in_bar"])
                steps += 1
                if not args.no_window:
                    img = env.render()
                    cv2.putText(
                        img,
                        f"ep{ep} prog={info['progress']:.0f} diff={info['difficulty']} "
                        f"bh={info['behavior']} acc={np.mean(acc[-120:]):.2f}",
                        (8, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )
                    cv2.imshow(win, img)
                    key = cv2.waitKey(16) & 0xFF
                    if key in (27, ord("q")):
                        return
            wins += info["success"]
            fails += not info["success"]
            accs.append(np.mean(acc))
            lens.append(steps)
            print(
                f"ep{ep}: {'WIN ' if info['success'] else 'FAIL'} "
                f"difficulty={info['difficulty']} behavior={info['behavior']} "
                f"acc={np.mean(acc):.2f} len={steps}"
            )
    finally:
        if not args.no_window:
            cv2.destroyAllWindows()

    if args.episodes > 1:
        print(f"\n共 {args.episodes} 局: win_rate={wins / args.episodes:.2f} "
              f"avg_acc={np.mean(accs):.2f} avg_len={np.mean(lens):.0f}")


if __name__ == "__main__":
    main()

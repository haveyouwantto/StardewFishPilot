"""用 gym 环境测试 main.py 当前 PDM-PID 控制器（可加测量噪声/输入延迟）。"""
import random
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as m  # noqa: E402
from env import FishingEnv, POND  # noqa: E402


def run_one(difficulty, behavior, eps=15, every=1, noise=0.0, delay=0,
            seed=42):
    pid = m.FishPID()
    env = FishingEnv(difficulty=difficulty, behavior=behavior,
                     max_steps=1200, seed=seed)
    rng = random.Random(seed + difficulty)
    wins = 0
    accs = []
    keys = 0
    cmd_q = deque([0] * delay)      # 输入延迟：命令 delay 个 tick 后才生效
    press = False
    for _ in range(eps):
        obs, _ = env.reset()
        done = False
        in_t = tot = 0
        tick = 0
        while not done:
            cmd = cmd_q.popleft() if cmd_q else 0
            obs, _, term, trunc, info = env.step(cmd)
            tick += 1
            if tick % every == 0:           # 每 every 个 tick 做一次控制决策
                d = env.debug_state()
                bar_cy = d["bar_pos"] + d["bar_size"] / 2
                fish_y = d["fish_pos"]
                bar_obs = bar_cy + rng.gauss(0.0, noise)   # 测量误差
                fish_obs = fish_y + rng.gauss(0.0, noise)
                duty = pid.control(env.steps / 60.0,
                                   bar_obs, fish_obs, True, True)
                t = env.steps / 60.0
                new_press = (duty > 0 and
                             (t % m.PDM_CYCLE_S) < duty * m.PDM_CYCLE_S)
                if new_press != press:
                    keys += 1
                press = new_press
            cmd_q.append(1 if press else 0)
            in_t += int(env._was_in_bar())
            tot += 1
            done = term or trunc
        cmd_q = deque([0] * delay)
        press = False
        wins += int(info["success"])
        accs.append(in_t / max(1, tot))
    return wins / eps, sum(accs) / len(accs), keys / eps


if __name__ == "__main__":
    m.KP, m.PDM_GAIN, m.DEADBAND_PX = 0.03, 0.6, 4.0
    m.KD = 0.010
    for kff in (0.0, 0.002, 0.005, 0.010):
        m.KFF = kff
        print(f"KFF={kff}:")
        for delay in (0, 2, 4):
            row = []
            for d, b in ((50, "mixed"), (50, "dart")):
                win, acc, _ = run_one(d, b, eps=10, every=3,
                                      noise=3.0, delay=delay)
                row.append(f"{b[0]}50={win:.2f}/{acc:.2f}")
            print(f"  delay={delay}: " + "  ".join(row))

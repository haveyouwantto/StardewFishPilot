"""One-off: 用 gym 环境测试 main.py 当前 PDM-PID 控制器（跑完删除）。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rl_fishing"))

import main as m  # noqa: E402
from env import FishingEnv, POND  # noqa: E402


def run_one(difficulty, behavior, eps=15, every=1):
    pid = m.FishPID()
    env = FishingEnv(difficulty=difficulty, behavior=behavior,
                     max_steps=1200, seed=42)
    wins = 0
    accs = []
    keys = 0
    holding = False
    for _ in range(eps):
        obs, _ = env.reset()
        done = False
        in_t = tot = 0
        prev_hold = False
        while not done:
            for _ in range(every):          # every>1 模拟低帧率控制
                obs, _, term, trunc, info = env.step(int(holding))
                if term or trunc:
                    done = True
                    break
            if done:
                break
            d = env.debug_state()
            bar_cy = d["bar_pos"] + d["bar_size"] / 2
            fish_y = d["fish_pos"]
            duty = pid.control(env.steps / 60.0, bar_cy, fish_y, True, True)
            t = env.steps / 60.0
            press = (duty > 0 and (t % m.PDM_CYCLE_S) < duty * m.PDM_CYCLE_S)
            if press and not holding:
                holding = True
                keys += 1
            elif not press and holding:
                holding = False
                keys += 1
            if env._was_in_bar():
                in_t += 1
            tot += 1
        holding = False
        wins += int(info["success"])
        accs.append(in_t / max(1, tot))
    return wins / eps, sum(accs) / len(accs), keys / eps


if __name__ == "__main__":
    m.KP = 0.04
    m.PDM_GAIN = 0.6
    for every, tag in ((1, "60Hz"), (3, "20Hz")):
        print(f"=== {tag} KP={m.KP} gain={m.PDM_GAIN} ===")
        for d, b in ((30, "mixed"), (50, "mixed"), (50, "dart"),
                     (70, "dart"), (50, "smooth")):
            win, acc, k = run_one(d, b, eps=15, every=every)
            print(f"difficulty={d:3d} behavior={b:6s}: "
                  f"win={win:.2f} acc={acc:.2f} keys/ep={k:.0f}")

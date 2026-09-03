#!/usr/bin/env python3
"""
在虚拟钓鱼小游戏上训练 PPO。

用法:
  uv run python rl_fishing/train.py --timesteps 300000
  uv run python rl_fishing/train.py --difficulty 50 --behavior dart  # 固定一种鱼
"""

import argparse
import itertools
import random
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env import BEHAVIORS, FishingEnv

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


class EvalCallback(BaseCallback):
    """定期跑 N 局，报告成功率 / 准度，并按成功率保存最好模型。"""

    def __init__(self, eval_episodes=50, eval_freq=10_000, verbose=1):
        super().__init__(verbose=verbose)
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq
        self.best = -np.inf
        self.last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        env = FishingEnv(seed=12345, control_hz=20,
                         obs_noise=2.5, latency_ticks=2)
        wins = fails = 0
        accs = []
        lens = []
        for _ in range(self.eval_episodes):
            obs, _ = env.reset()
            done = False
            acc = []
            steps = 0
            while not done:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(int(act))
                done = term or trunc
                acc.append(info["in_bar"])
                steps += 1
            wins += info["success"]
            fails += not info["success"]
            accs.append(np.mean(acc))
            lens.append(steps)
        win_rate = wins / self.eval_episodes
        score = win_rate
        if score > self.best:
            self.best = score
            path = MODEL_DIR / "best_model.zip"
            self.model.save(path)
            if self.verbose:
                print(f"[eval] {self.n_calls} steps: 新最佳 win_rate={win_rate:.2f} -> {path}")
        if self.verbose:
            print(
                f"[eval] steps={self.n_calls} win_rate={win_rate:.2f} "
                f"avg_acc={np.mean(accs):.2f} avg_len={np.mean(lens):.0f}"
            )
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--difficulty", type=int, default=None)
    ap.add_argument("--behavior", type=str, default=None)
    ap.add_argument("--difficulty-min", type=int, default=25)
    ap.add_argument("--difficulty-max", type=int, default=90)
    ap.add_argument("--control-hz", type=int, default=20)
    ap.add_argument("--noise-min", type=float, default=1.0)
    ap.add_argument("--noise-max", type=float, default=4.0)
    ap.add_argument("--latency-min", type=int, default=1)
    ap.add_argument("--latency-max", type=int, default=3)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--save-freq", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=50)
    args = ap.parse_args()

    if args.behavior is not None and args.behavior not in BEHAVIORS:
        ap.error(f"--behavior 可选 {BEHAVIORS}")
    MODEL_DIR.mkdir(exist_ok=True)

    rand_iter = itertools.count(1000)

    def make_env():
        s = next(rand_iter)
        rng = random.Random(s)
        return FishingEnv(
            difficulty=args.difficulty,
            behavior=args.behavior,
            level=args.level,
            obs_noise=rng.uniform(args.noise_min, args.noise_max),
            control_hz=args.control_hz,
            latency_ticks=rng.randint(args.latency_min, args.latency_max),
            difficulty_range=(args.difficulty_min, args.difficulty_max),
            seed=s,
        )

    env = DummyVecEnv([make_env for _ in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        verbose=1,
        device="cpu",
        seed=0,
    )
    eval_cb = EvalCallback(
        eval_episodes=args.eval_episodes, eval_freq=args.save_freq, verbose=1
    )

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    elapsed = time.time() - t0

    model.save(MODEL_DIR / "final_model.zip")
    print(f"\n训练完成: {args.timesteps} steps, {elapsed:.0f}s")
    print(f"模型: {MODEL_DIR / 'final_model.zip'}")
    print(f"观看: uv run python rl_fishing/watch.py")


if __name__ == "__main__":
    main()

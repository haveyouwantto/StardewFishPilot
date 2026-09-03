#!/usr/bin/env python3
"""
Stardew Valley 钓鱼小游戏 RL 环境。

物理与行为参数按 ramenchef.net/fishing-game/fishing.js 逐条移植：
  - 按住: bar 加速度 -0.25/tick，松开: +0.25/tick，鱼在条内时加速度乘 0.6
  - bar 触顶/底: 按住时速度清零；松开时按 -2/3 反弹
  - 鱼: 目标点平滑趋近 + 随机换目标（难度/行为决定频率与幅度）
  - 进度: 初始 300，在条内 +2/tick，脱离 -3/tick，1000 成功 / 0 失败
  - 1 tick = 1/60 秒

动作: 0=松开, 1=按住
观测: [bar_pos, bar_vel, fish_pos, fish_vel, progress, prev_action]（归一化）
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

POND = 568.0                 # 池塘高度(px)，与原版一致
TICK_S = 1 / 60.0
BEHAVIORS = ("mixed", "dart", "smooth", "floater", "sinker")

VEL_SCALE = 6.0              # 观测中速度的归一化除数


@dataclass
class FishConfig:
    """一条鱼/一局的参数。difficulty=None 或 behavior=None 表示随机。"""

    difficulty: Optional[int] = None          # 0~110+，越高越难
    behavior: Optional[str] = None            # mixed/dart/smooth/floater/sinker
    level: int = 0                            # 钓鱼等级，决定 bar 长度
    max_steps: int = 1500
    obs_noise: float = 0.0                    # 观测噪声 px（模拟 YOLO）
    control_hz: int = 60                      # 决策频率（物理仍 60Hz）
    latency_ticks: int = 0                    # 输入延迟（60Hz tick 数）


class FishingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(
        self,
        difficulty: Optional[int] = None,
        behavior: Optional[str] = None,
        level: int = 0,
        max_steps: int = 1500,
        obs_noise: float = 0.0,
        control_hz: int = 60,
        latency_ticks: int = 0,
        difficulty_range: tuple[int, int] = (25, 90),
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if behavior is not None and behavior not in BEHAVIORS:
            raise ValueError(f"behavior 必须是 {BEHAVIORS}")
        if 60 % control_hz != 0 or not (10 <= control_hz <= 60):
            raise ValueError("control_hz 必须是 10~60 且能整除 60")
        self.cfg = FishConfig(
            difficulty=difficulty,
            behavior=behavior,
            level=level,
            max_steps=max_steps,
            obs_noise=obs_noise,
            control_hz=control_hz,
            latency_ticks=latency_ticks,
        )
        self._substeps = 60 // control_hz
        self.difficulty_range = difficulty_range
        self.render_mode = render_mode
        self._rng = random.Random(seed)

        # 0=松开, 1=按住
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # 每局状态
        self.difficulty = 30
        self.behavior = "mixed"
        self.bar_size = 96.0
        self.bar_pos = 0.0
        self.bar_vel = 0.0
        self.fish_pos = 0.0
        self.fish_vel = 0.0
        self.fish_base_vel = 0.0
        self.fish_target = -1.0
        self.progress = 300.0
        self.steps = 0
        self.last_action = 0
        self.prev_noisy_bar = None
        self.prev_noisy_fish = None
        self.prev_true_bar = None
        self.prev_true_fish = None
        self._schedule = {}
        self._last_cmd = 0
        self._prev_action = 0
        self.prev_true_bar = None
        self.prev_true_fish = None
        self._schedule: dict[int, int] = {}
        self._last_cmd = 0
        self._prev_action = 0

    # ---------------- 工具 ----------------
    def _rand(self, lo: float, hi: float) -> float:
        """移植 JS 的 rand(min, max)：hi 为开区间。"""
        if hi <= lo:
            return float(lo)
        return lo + self._rng.random() * (hi - lo)

    def _randint(self, lo: int, hi: int) -> int:
        if hi <= lo:
            return lo
        return self._rng.randint(lo, hi - 1)

    def _new_fish(self):
        """按难度/行为设置开始新的一局。"""
        if self.cfg.difficulty is not None:
            self.difficulty = self.cfg.difficulty
        else:
            self.difficulty = self._rng.randint(*self.difficulty_range)
        if self.cfg.behavior is not None:
            self.behavior = self.cfg.behavior
        else:
            self.behavior = self._rng.choice(BEHAVIORS)
        self.bar_size = 96.0 + 8.0 * self.cfg.level
        self.bar_pos = POND - self.bar_size
        self.bar_vel = 0.0
        self.fish_pos = 508.0
        self.fish_vel = 0.0
        self.fish_base_vel = 0.0
        self.fish_target = (1 - self.difficulty / 100.0) * 548.0
        self.progress = 300.0
        self.steps = 0
        self.last_action = 0
        self.prev_noisy_bar = None
        self.prev_noisy_fish = None

    def _was_in_bar(self) -> bool:
        fish_top = self.fish_pos + 16
        fish_bottom = self.fish_pos + 44
        bar_top = self.bar_pos
        bar_bottom = self.bar_pos + self.bar_size
        bar_at_bottom = bar_top >= POND - self.bar_size - 4
        return (fish_bottom <= bar_bottom or bar_at_bottom) and fish_top >= bar_top

    def _observe(self, bar_y, fish_y, bar_vel, fish_vel):
        return np.array(
            [
                bar_y / POND,
                float(np.clip(bar_vel, -VEL_SCALE, VEL_SCALE)) / VEL_SCALE,
                fish_y / POND,
                float(np.clip(fish_vel, -VEL_SCALE, VEL_SCALE)) / VEL_SCALE,
                self.progress / 1000.0,
                float(self.last_action),
            ],
            dtype=np.float32,
        )

    def debug_state(self) -> dict:
        """供 watch/调试界面读取的本帧完整内部状态。"""
        return {
            "difficulty": self.difficulty,
            "behavior": self.behavior,
            "bar_size": self.bar_size,
            "bar_pos": self.bar_pos,
            "bar_vel": self.bar_vel,
            "fish_pos": self.fish_pos,
            "fish_vel": self.fish_vel,
            "fish_base_vel": self.fish_base_vel,
            "fish_target": self.fish_target,
            "progress": self.progress,
            "steps": self.steps,
            "in_bar": self._was_in_bar(),
            "last_action": self.last_action,
        }

    # ---------------- Gym API ----------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        self._new_fish()
        return self._observe(self.bar_pos, self.fish_pos, 0.0, 0.0), {}

    def _tick_60(self, action: int):
        action = int(action)
        hold = action == 1
        r = self._rng

        # ---------- 鱼的目标更新（原版逐行移植） ----------
        smooth = self.behavior == "smooth"
        retarget_p = self.difficulty * (0.005 if smooth else 0.00025)
        if r.random() < retarget_p and (self.fish_target == -1 or not smooth):
            percent = min(99, self.difficulty + self._randint(10, 45)) * 0.01
            lo = float(math.ceil(-self.fish_pos))
            hi = float(math.floor(POND - 20 - self.fish_pos)) * percent
            self.fish_target = self.fish_pos + self._rand(lo, hi)

        if self.fish_target != -1 and abs(self.fish_pos - self.fish_target) > 3:
            denom = self._randint(10, 30) + max(0, 100 - self.difficulty)
            fish_accel = (self.fish_target - self.fish_pos) / denom
            self.fish_vel += (fish_accel - self.fish_vel) / 5
        elif not smooth and r.random() < 0.0005 * self.difficulty:
            up = r.random() < 0.5
            self.fish_target = self.fish_pos + (
                self._randint(-100, -51) if up else self._randint(50, 101)
            )
        else:
            self.fish_target = -1

        if self.behavior == "dart" and r.random() < 0.001 * self.difficulty:
            up = r.random() < 0.5
            if up:
                self.fish_target = self.fish_pos + self._randint(
                    -100 - self.difficulty * 2, -51
                )
            else:
                self.fish_target = self.fish_pos + self._randint(
                    50, 101 + self.difficulty * 2
                )
        if self.behavior == "floater":
            self.fish_base_vel = max(self.fish_base_vel - 0.01, -1.5)
        elif self.behavior == "sinker":
            self.fish_base_vel = min(self.fish_base_vel + 0.01, 1.5)

        self.fish_target = max(-1.0, min(self.fish_target, POND - 20))
        self.fish_pos = max(
            0.0, min(self.fish_pos + self.fish_vel + self.fish_base_vel, POND - 36)
        )

        # ---------- 判定与进度 ----------
        in_bar = self._was_in_bar()
        if in_bar:
            self.progress += 2
        else:
            self.progress -= 3
        self.progress = max(0.0, min(self.progress, 1000.0))

        # ---------- bar 物理 ----------
        bacc = -0.25 if hold else 0.25
        if in_bar:
            bacc *= 0.6
        self.bar_vel += bacc
        self.bar_pos += self.bar_vel
        if self.bar_pos < 0:
            self.bar_pos = 0
            self.bar_vel = 0.0 if hold else -2 / 3 * self.bar_vel
        elif self.bar_pos + self.bar_size > POND:
            self.bar_pos = POND - self.bar_size
            self.bar_vel = 0.0 if hold else -2 / 3 * self.bar_vel

        # ---------- 奖励 ----------
        reward = (2.0 if in_bar else -3.0) / 60.0
        self.steps += 1

        terminated = self.progress >= 1000 or self.progress <= 0
        if self.progress >= 1000:
            reward += 5.0
        elif self.progress <= 0:
            reward -= 5.0
        return reward, terminated, self.progress >= 1000, in_bar

    def step(self, action: int):
        """一个决策步 = 推进 60/control_hz 个内部 tick（如 20Hz=3 tick），
        动作经 latency_ticks 延迟生效；观测带测量噪声。"""
        action = int(action)
        # 输入延迟：本次动作在 latency_ticks 个内部 tick 后才生效
        self._schedule[self.steps + self.cfg.latency_ticks] = action

        total_reward = 0.0
        terminated = truncated = False
        success = False
        for _ in range(self._substeps):
            if self.steps >= self.cfg.max_steps:
                truncated = True
                break
            cmd = self._schedule.get(self.steps, self._last_cmd)
            self._last_cmd = cmd
            r, term, success, _in_bar = self._tick_60(cmd)
            total_reward += r
            if term:
                terminated = True
                break
        if action != self._prev_action:
            total_reward -= 0.002
        self._prev_action = action
        self.last_action = action

        # ---------- 观测：每 control 步采样一次 + 测量噪声 ----------
        n = self.cfg.obs_noise
        if n > 0:
            bar_obs = self.bar_pos + self._rng.gauss(0.0, n)
            fish_obs = self.fish_pos + self._rng.gauss(0.0, n)
            if self.prev_noisy_bar is None:
                bar_vel_obs = fish_vel_obs = 0.0
            else:
                bar_vel_obs = (bar_obs - self.prev_noisy_bar) / self._substeps
                fish_vel_obs = (fish_obs - self.prev_noisy_fish) / self._substeps
            self.prev_noisy_bar = bar_obs
            self.prev_noisy_fish = fish_obs
        else:
            bar_obs, fish_obs = self.bar_pos, self.fish_pos
            if self.prev_true_bar is None:
                bar_vel_obs = fish_vel_obs = 0.0
            else:
                bar_vel_obs = (bar_obs - self.prev_true_bar) / self._substeps
                fish_vel_obs = (fish_obs - self.prev_true_fish) / self._substeps
            self.prev_true_bar = bar_obs
            self.prev_true_fish = fish_obs

        obs = self._observe(bar_obs, fish_obs, bar_vel_obs, fish_vel_obs)
        info = {
            "progress": self.progress,
            "in_bar": self._was_in_bar(),
            "difficulty": self.difficulty,
            "behavior": self.behavior,
            "success": success,
        }
        return obs, total_reward, terminated, truncated, info

    # ---------------- 渲染 ----------------
    def _frame_rgb(self, scale: int = 3):
        w, h = 66 * scale, 579 * scale
        img = np.full((h, w, 3), 255, np.uint8)
        s = scale
        # 池塘
        cv2.rectangle(img, (6 * s, 1 * s), (38 * s, int(POND) * s), (255, 224, 130), -1)
        cv2.rectangle(img, (6 * s, 1 * s), (38 * s, int(POND) * s), (0, 0, 0), 1)
        # bar
        x0, y0 = 6 * s, int(1 + self.bar_pos) * s
        bh = int(self.bar_size) * s
        color = (0, 255, 0) if self._was_in_bar() else (0, 180, 0)
        cv2.rectangle(img, (x0, y0), (x0 + 32 * s, y0 + bh), color, -1)
        cv2.rectangle(img, (x0, y0), (x0 + 32 * s, y0 + bh), (0, 0, 0), 1)
        # 鱼
        fy = int(25 + self.fish_pos) * s
        cv2.ellipse(
            img,
            (22 * s, fy),
            (26 * s, 13 * s),
            45,
            0,
            360,
            (70, 130, 255),
            -1,
        )
        # 进度条
        ph = int(self.progress * 0.57) * s
        py = int((1000 - self.progress) * 0.57) * s
        pcol = (0, 200, 0) if self.progress >= 1000 else (
            (0, 0, 255) if self.progress <= 0 else (0, 255, 255)
        )
        cv2.rectangle(img, (50 * s, py), (66 * s, py + ph), pcol, -1)
        return img

    def render(self):
        return self._frame_rgb()

    def close(self):
        pass

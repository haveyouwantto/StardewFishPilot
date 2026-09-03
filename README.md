# Stardew Valley Auto Fishing

## 项目介绍

星露谷钓鱼小游戏的自动化辅助程序。程序对屏幕上的钓鱼界面做实时检测，并根据鱼的位置控制游戏内的绿色条，尝试把鱼保持在有效区域内直至捕获。

项目处于实验阶段：控制策略、视觉检测和运行帧率都还有明显限制，不保证在任意环境下稳定工作，也不建议用于长期挂机。

整体流程分为三个阶段：

1. UI 定位：在全屏画面中通过模板匹配寻找钓鱼面板；只有模板命中且候选区域内检测到鱼时，才锁定该区域。
2. 目标检测：UI 锁定后只截取 ROI，在鱼和条两个目标上进行逐帧检测（详见“检测方案”）。
3. 控制：根据检测结果决定按住或松开游戏按键。支持两套控制方案：PDM-PID 与强化学习策略。

## 技术选型

| 模块 | 方案 |
| --- | --- |
| 运行环境 | Python 3.13，uv 管理虚拟环境 |
| 鱼检测 | 自训练的 ultralytics YOLO（fish/bar 二分类），CUDA + FP16 |
| bar 检测 | OpenCV HSV 绿色分割 + 形状筛选，仅在鱼检测框 y 范围内搜索 |
| 鱼模板（备用） | OpenCV 模板匹配（templates/fish.png） |
| 截屏 | mss（仅截取锁定 ROI） |
| 输入模拟 | pynput 键盘控制器 |
| Overlay | pywin32 分层窗口（GDI 绘制，限频重绘） |
| RL 训练 | gymnasium 环境 + stable-baselines3 PPO |

## 检测方案

默认检测模式为 `mixed`：

- fish 使用 YOLO，返回中心点与完整检测框；
- bar 使用传统视觉，在 fish 检测框的 y 范围内寻找绿色矩形；
- bar 检测失败时回退到 YOLO 的 bar 结果。

可在运行时用 F6 在 `cv` / `mixed` / `yolo` 之间切换。

## 控制方案

### PDM-PID

控制器输出一个连续的“按压力度”值，再由 PDM 在固定周期内转换为占空比按键：

```
u = kp * e - kd * v_bar
duty = clamp(-u * PDM_GAIN, 0, 1)
```

其中 `e` 为鱼与 bar 中心的纵向误差（y 向下为正），`v_bar` 为 bar 速度估计。参数位于 `main.py` 顶部配置区。

### 强化学习

RL 环境按 ramenchef.net 的钓鱼小游戏实现移植，内部物理固定 60Hz；训练时加入以下真实化设定：

- 决策频率 20Hz（每 3 个物理 tick 决策一次）；
- 位置观测加 1~4px 高斯噪声；
- 动作延迟 1~3 个 tick 生效；
- 难度与行为（mixed/dart/smooth/floater/sinker）逐局随机。

策略为 PPO（MLP 128×128），训练代码见 `rl_fishing/train.py`。模型保存在 `rl_fishing/models/`，默认不提交 git。

## 运行

```bash
uv run python main.py
```

热键：

| 按键 | 功能 |
| --- | --- |
| F6 | 切换检测模式（cv / mixed / yolo） |
| F7 | 预切控制模式（RL / PID），暂停或当前局结束后生效 |
| F8 | 开始 / 暂停 |
| F9 | 退出 |

默认配置写在 `main.py` 顶部：

- `CONTROL_MODE`：`"rl"` 或 `"pid"`；
- `DETECT_MODE`：`"cv"`、`"mixed"` 或 `"yolo"`。

## 训练

### YOLO 检测模型

数据集位于 `yolo_dataset/`，类别为 `bar`(0) 与 `fish`(1)。训练入口：

```bash
uv run python train_yolo.py
```

### RL 策略

```bash
uv run python rl_fishing/train.py --timesteps 200000
```

评估当前 PDM-PID 在模拟环境中的表现：

```bash
uv run python rl_fishing/test_pid_gym.py
```

## 已知限制

- 游戏进度条尚未读取，RL 观测中的 `progress` 为固定估计值；
- RL 仅按钓鱼等级 0 训练，等级变化导致 bar 长度不同时行为可能偏差；
- 低帧率（约 17~20Hz）下控制性能明显下降；
- CV 鱼模板与具体鱼种外观绑定，换鱼种可能漏检；
- ROI 与模板参数依赖固定分辨率/窗口布局，更换环境需重新校准。

## 免责声明

本仓库仅用于自动化与控制算法学习。使用过程中可能违反游戏条款，作者不对账号风险或使用后果负责。

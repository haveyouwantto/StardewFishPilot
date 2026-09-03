# 虚拟钓鱼 RL

按 ramenchef.net/fishing-game 的物理与鱼行为移植的 Stardew 钓鱼小游戏环境，
用于离线训练 RL 按键策略（按住/松开），暂不涉及 YOLO 视觉。

## 安装（进入项目 uv 环境）

```bash
uv pip install -r rl_fishing/requirements.txt
```

## 训练

```bash
uv run python rl_fishing/train.py --timesteps 300000
```

默认每局随机抽取难度 25~90、行为 mixed/dart/smooth/floater/sinker。
想固定练某类鱼：

```bash
uv run python rl_fishing/train.py --difficulty 60 --behavior dart --timesteps 200000
```

观测/奖励细节见 `env.py` 文件头。

## 观看 / 评估

```bash
uv run python rl_fishing/watch.py                  # 弹窗实时观看
uv run python rl_fishing/watch.py --no-window --episodes 200   # 纯统计胜率
```

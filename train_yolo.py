#!/usr/bin/env python3
"""
训练星露谷钓鱼检测模型

类别（必须一致）:
  0 = bar
  1 = fish

用法:
  uv pip install ultralytics
  uv run python scripts/train_yolo.py
"""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[0]
DATA = ROOT / "yolo_dataset" / "data.yaml"
OUT = ROOT / "yolo_runs"


def main():
    # 若 data.yaml 里 path 是相对路径，写成绝对路径更稳
    yaml_text = DATA.read_text(encoding="utf-8")
    if "path: yolo_dataset" in yaml_text:
        DATA.write_text(
            yaml_text.replace(
                "path: yolo_dataset",
                f"path: {ROOT / 'yolo_dataset'}",
            ),
            encoding="utf-8",
        )

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=640,
        batch=4,
        project=str(OUT),
        name="stardew_fish",
        exist_ok=True,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.4,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,
        mosaic=0.8,
        mixup=0.05,
    )
    best = OUT / "stardew_fish" / "weights" / "best.pt"
    print("\n训练完成")
    print("权重路径:", best)
    print("运行主程序时会自动加载这个 best.pt")


if __name__ == "__main__":
    main()

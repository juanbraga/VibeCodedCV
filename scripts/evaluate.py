#!/usr/bin/env python3
"""Evaluate a trained YOLO model on the test split and print metrics."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a YOLO model")
    p.add_argument("--weights", required=True, help="Path to trained weights (.pt)")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.45)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_yaml = str(Path(cfg["dataset"]["yaml"]).resolve())

    model = YOLO(args.weights)

    metrics = model.val(
        data=dataset_yaml,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
        project="outputs/evaluation",
        name=Path(args.weights).stem,
        exist_ok=True,
    )

    print("\n=== Evaluation Results ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    if hasattr(metrics.box, "ap_class_index"):
        print("\nPer-class AP50:")
        names = model.names
        for idx, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
            print(f"  {names[idx]}: {ap:.4f}")


if __name__ == "__main__":
    main()

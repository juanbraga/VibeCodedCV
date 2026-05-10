#!/usr/bin/env python3
"""Train a YOLOv8 model on the synthetic dataset."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on synthetic dataset")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--weights", default=None, help="Override base weights (e.g. yolov8s.pt)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu / 0 / cuda")
    p.add_argument("--name", default="yolo_synthetic", help="Run name under outputs/training/")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    weights = args.weights or model_cfg["weights"]
    epochs = args.epochs or train_cfg["epochs"]
    batch = args.batch or train_cfg["batch_size"]
    device = args.device or train_cfg["device"]

    dataset_yaml = str(Path(cfg["dataset"]["yaml"]).resolve())

    model = YOLO(weights)

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=model_cfg["input_size"],
        lr0=train_cfg["learning_rate"],
        patience=train_cfg["patience"],
        save_period=train_cfg["save_period"],
        workers=train_cfg["workers"],
        device=device,
        project="outputs/training",
        name=args.name,
        augment=cfg["dataset"]["augment"],
        exist_ok=True,
    )

    print(f"\nTraining complete. Results saved to: outputs/training/{args.name}/")
    print(f"Best weights: outputs/training/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    main()

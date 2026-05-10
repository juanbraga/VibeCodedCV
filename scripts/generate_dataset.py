#!/usr/bin/env python3
"""Generate synthetic dataset with balls, boxes, and triangles."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import DatasetGenerator  # noqa: import only what's needed (no ultralytics dep)


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic YOLO dataset")
    p.add_argument("--output", default="data", help="Output directory")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--train", type=int, default=80, help="Number of training images")
    p.add_argument("--val", type=int, default=20, help="Number of validation images")
    p.add_argument("--test", type=int, default=10, help="Number of test images")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    gen = DatasetGenerator(
        output_dir=args.output,
        img_size=args.img_size,
        n_train=args.train,
        n_val=args.val,
        n_test=args.test,
        seed=args.seed,
    )
    gen.generate()
    print(f"\nDataset ready at: {args.output}/")
    print("Classes: 0=ball, 1=box, 2=triangle")


if __name__ == "__main__":
    main()

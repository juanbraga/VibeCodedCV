#!/usr/bin/env python3
"""Preview dataset images with their ground-truth annotations overlaid."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


CLASS_NAMES = {0: "ball", 1: "box", 2: "triangle"}
COLORS = [(220, 80, 80), (80, 180, 80), (80, 80, 220)]


def draw_yolo_labels(image: np.ndarray, label_path: Path) -> np.ndarray:
    H, W = image.shape[:2]
    if not label_path.exists():
        return image

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        color = COLORS[cls_id % len(COLORS)]
        name = CLASS_NAMES.get(cls_id, str(cls_id))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, name, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return image


def parse_args():
    p = argparse.ArgumentParser(description="Visualize dataset annotations")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--data", default="data", help="Dataset root dir")
    p.add_argument("--max", type=int, default=20, dest="max_images")
    p.add_argument("--save", default=None, help="Save grid to this path instead of displaying")
    return p.parse_args()


def main():
    args = parse_args()
    img_dir = Path(args.data) / "images" / args.split
    lbl_dir = Path(args.data) / "labels" / args.split

    images_paths = sorted(img_dir.glob("*.jpg"))[: args.max_images]
    if not images_paths:
        print(f"No images found in {img_dir}")
        sys.exit(1)

    annotated = []
    for p in images_paths:
        img = cv2.imread(str(p))
        lbl = lbl_dir / (p.stem + ".txt")
        annotated.append(draw_yolo_labels(img, lbl))

    thumb_size = 200
    cols = 5
    rows = (len(annotated) + cols - 1) // cols
    grid = np.full((rows * thumb_size, cols * thumb_size, 3), 50, dtype=np.uint8)

    for idx, img in enumerate(annotated):
        r, c = divmod(idx, cols)
        thumb = cv2.resize(img, (thumb_size, thumb_size))
        grid[r * thumb_size:(r + 1) * thumb_size, c * thumb_size:(c + 1) * thumb_size] = thumb

    if args.save:
        cv2.imwrite(args.save, grid)
        print(f"Grid saved to {args.save}")
    else:
        cv2.imshow(f"Dataset: {args.split} ({len(annotated)} images)", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

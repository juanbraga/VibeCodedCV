from __future__ import annotations

import cv2
import numpy as np
import random
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Annotation:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


CLASS_BALL = 0
CLASS_BOX = 1
CLASS_TRIANGLE = 2

PALETTE = {
    CLASS_BALL: [(220, 80, 80), (180, 60, 60), (255, 100, 100)],
    CLASS_BOX: [(80, 180, 80), (60, 140, 60), (100, 220, 100)],
    CLASS_TRIANGLE: [(80, 80, 220), (60, 60, 180), (100, 100, 255)],
}


def _draw_ball(img: np.ndarray, cx: int, cy: int, r: int, color: tuple) -> tuple[int, int, int, int]:
    cv2.circle(img, (cx, cy), r, color, -1)
    cv2.circle(img, (cx, cy), r, tuple(max(0, c - 40) for c in color), 2)
    return cx - r, cy - r, 2 * r, 2 * r


def _draw_box(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple) -> tuple[int, int, int, int]:
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), tuple(max(0, c - 40) for c in color), 2)
    return x, y, w, h


def _draw_triangle(img: np.ndarray, cx: int, cy: int, size: int, color: tuple) -> tuple[int, int, int, int]:
    pts = np.array([
        [cx, cy - size],
        [cx - size, cy + size],
        [cx + size, cy + size],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, tuple(max(0, c - 40) for c in color), 2)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


def _random_color(class_id: int) -> tuple:
    return random.choice(PALETTE[class_id])


class DatasetGenerator:
    def __init__(
        self,
        output_dir: str | Path,
        img_size: int = 640,
        n_train: int = 80,
        n_val: int = 20,
        n_test: int = 10,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        random.seed(seed)
        np.random.seed(seed)

    def _make_background(self) -> np.ndarray:
        bg_type = random.choice(["solid", "gradient", "noise"])
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if bg_type == "solid":
            color = [random.randint(200, 255) for _ in range(3)]
            img[:] = color

        elif bg_type == "gradient":
            base = random.randint(180, 220)
            for i in range(self.img_size):
                val = min(255, int(base + (i / self.img_size) * 30))
                img[i, :] = [val, val, val]

        else:
            img = np.random.randint(200, 240, (self.img_size, self.img_size, 3), dtype=np.uint8)

        return img

    def _generate_image(self) -> tuple[np.ndarray, list[Annotation]]:
        img = self._make_background()
        annotations = []
        n_objects = random.randint(1, 5)
        H, W = self.img_size, self.img_size

        for _ in range(n_objects):
            class_id = random.randint(0, 2)
            color = _random_color(class_id)

            if class_id == CLASS_BALL:
                r = random.randint(20, 60)
                cx = random.randint(r, W - r)
                cy = random.randint(r, H - r)
                bx, by, bw, bh = _draw_ball(img, cx, cy, r, color)

            elif class_id == CLASS_BOX:
                bw = random.randint(40, 120)
                bh = random.randint(40, 120)
                bx = random.randint(0, W - bw)
                by = random.randint(0, H - bh)
                bx, by, bw, bh = _draw_box(img, bx, by, bw, bh, color)

            else:
                size = random.randint(25, 60)
                cx = random.randint(size, W - size)
                cy = random.randint(size, H - size)
                bx, by, bw, bh = _draw_triangle(img, cx, cy, size, color)

            norm_cx = (bx + bw / 2) / W
            norm_cy = (by + bh / 2) / H
            norm_w = bw / W
            norm_h = bh / H

            norm_cx = max(0.0, min(1.0, norm_cx))
            norm_cy = max(0.0, min(1.0, norm_cy))
            norm_w = max(0.001, min(1.0, norm_w))
            norm_h = max(0.001, min(1.0, norm_h))

            annotations.append(Annotation(class_id, norm_cx, norm_cy, norm_w, norm_h))

        return img, annotations

    def _write_split(self, split: str, count: int):
        img_dir = self.output_dir / "images" / split
        lbl_dir = self.output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img, annotations = self._generate_image()
            stem = f"{split}_{i:04d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
            label_path = lbl_dir / f"{stem}.txt"
            label_path.write_text("\n".join(a.to_yolo_line() for a in annotations))

        print(f"  {split}: {count} images → {img_dir}")

    def generate(self):
        print("Generating synthetic dataset...")
        self._write_split("train", self.n_train)
        self._write_split("val", self.n_val)
        self._write_split("test", self.n_test)
        print("Done.")

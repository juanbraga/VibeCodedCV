from __future__ import annotations

import cv2
import json
import numpy as np
import yaml
from pathlib import Path


COLORS = [
    (220, 80, 80),
    (80, 180, 80),
    (80, 80, 220),
    (220, 180, 80),
    (80, 220, 180),
    (180, 80, 220),
    (220, 120, 60),
]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        class_id = det["class_id"]
        color = COLORS[class_id % len(COLORS)]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = y1 - 8 if y1 - 8 > th else y1 + th + 8
        cv2.rectangle(image, (x1, label_y - th - baseline), (x1 + tw + 4, label_y + baseline), color, -1)
        cv2.putText(image, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

    return image


def save_results(
    image: np.ndarray,
    detections: list[dict],
    output_dir: str | Path,
    stem: str,
    save_image: bool = True,
    save_txt: bool = True,
    save_json: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_image:
        annotated = draw_detections(image.copy(), detections)
        cv2.imwrite(str(output_dir / f"{stem}_detected.jpg"), annotated)

    if save_txt:
        lines = [
            f"{d['class_name']} {d['confidence']:.4f} "
            f"{d['bbox'][0]:.1f} {d['bbox'][1]:.1f} {d['bbox'][2]:.1f} {d['bbox'][3]:.1f}"
            for d in detections
        ]
        (output_dir / f"{stem}.txt").write_text("\n".join(lines))

    if save_json:
        (output_dir / f"{stem}.json").write_text(
            json.dumps({"detections": detections}, indent=2)
        )


def letterbox(image: np.ndarray, target: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

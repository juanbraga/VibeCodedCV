#!/usr/bin/env python3
"""Run YOLO detection on images, a directory, or a video file."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from src.detector import YOLODetector
from src.utils import draw_detections, save_results, load_config


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO object detection")
    p.add_argument("source", help="Image path, directory, or video file")
    p.add_argument("--weights", default="yolov8n.pt", help="Model weights (.pt file)")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--output", default="outputs/detections", help="Output directory")
    p.add_argument("--no-save", action="store_true", help="Do not save annotated images")
    p.add_argument("--show", action="store_true", help="Display results with OpenCV")
    p.add_argument("--config", default=None, help="Optional config.yaml (overrides other flags)")
    return p.parse_args()


def process_image(detector: YOLODetector, image_path: Path, output_dir: Path, show: bool, save: bool, imgsz: int):
    image, detections = detector.detect_image(image_path, imgsz)

    print(f"  {image_path.name}: {len(detections)} detection(s)")
    for d in detections:
        print(f"    - {d['class_name']} ({d['confidence']:.2f})")

    if save:
        save_results(image, detections, output_dir, image_path.stem)

    if show:
        annotated = draw_detections(image.copy(), detections)
        cv2.imshow(image_path.name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
        weights = cfg["model"]["weights"]
        conf = cfg["model"]["confidence_threshold"]
        iou = cfg["model"]["iou_threshold"]
        imgsz = cfg["model"]["input_size"]
    else:
        weights, conf, iou, imgsz = args.weights, args.conf, args.iou, args.imgsz

    detector = YOLODetector(weights=weights, conf=conf, iou=iou)
    output_dir = Path(args.output)
    source = Path(args.source)
    save = not args.no_save

    if not source.exists():
        print(f"Error: source not found: {source}")
        sys.exit(1)

    if source.is_dir():
        images = [p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not images:
            print("No images found in directory.")
            sys.exit(1)
        print(f"Detecting in {len(images)} image(s) from {source}/")
        for img_path in sorted(images):
            process_image(detector, img_path, output_dir, args.show, save, imgsz)

    elif source.suffix.lower() in IMAGE_EXTS:
        print(f"Detecting in: {source}")
        process_image(detector, source, output_dir, args.show, save, imgsz)

    elif source.suffix.lower() in VIDEO_EXTS:
        out_video = str(output_dir / f"{source.stem}_detected.mp4") if save else None
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing video: {source}  →  {out_video or '(not saving)'}")
        detector.detect_video(str(source), out_video, imgsz)

    else:
        print(f"Unsupported file type: {source.suffix}")
        sys.exit(1)

    if save:
        print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()

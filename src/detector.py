from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.5, iou: float = 0.45):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.class_names = self.model.names

    def detect(self, source, imgsz: int = 640) -> list[dict]:
        results = self.model(source, conf=self.conf, iou=self.iou, imgsz=imgsz, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": self.class_names[int(box.cls[0])],
                })
        return detections

    def detect_image(self, image_path: str | Path, imgsz: int = 640) -> tuple[np.ndarray, list[dict]]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        detections = self.detect(str(image_path), imgsz)
        return image, detections

    def detect_frame(self, frame: np.ndarray, imgsz: int = 640) -> list[dict]:
        results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=imgsz, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": self.class_names[int(box.cls[0])],
                })
        return detections

    def detect_video(self, video_path: str, output_path: str | None = None, imgsz: int = 640):
        cap = cv2.VideoCapture(video_path)
        writer = None

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        from .utils import draw_detections

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detect_frame(frame, imgsz)
            annotated = draw_detections(frame.copy(), detections)

            if writer:
                writer.write(annotated)

            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

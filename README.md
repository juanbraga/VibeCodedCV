# VibeCodedCV — OpenCV + YOLOv8 Object Detection

End-to-end object detection project using OpenCV and YOLOv8. Includes a synthetic dataset generator, training pipeline, inference scripts, and evaluation tools.

## Classes

| ID | Name     | Shape              |
|----|----------|--------------------|
| 0  | ball     | Red circle         |
| 1  | box      | Green rectangle    |
| 2  | triangle | Red triangle       |

## Project Structure

```
VibeCodedCV/
├── Makefile
├── requirements.txt
├── config/
│   ├── config.yaml          # model & training hyperparameters
│   └── dataset.yaml         # YOLO dataset definition (3 classes)
├── data/
│   ├── images/
│   │   ├── train/           # 80 generated images
│   │   ├── val/             # 20 generated images
│   │   └── test/            # 10 generated images
│   └── labels/
│       ├── train/           # YOLO-format .txt annotations
│       └── val/
├── src/
│   ├── dataset.py           # synthetic image generator
│   ├── detector.py          # YOLODetector wrapper class
│   └── utils.py             # draw boxes, save results, IoU helper
├── scripts/
│   ├── generate_dataset.py  # regenerate the dataset
│   ├── train.py             # fine-tune YOLOv8
│   ├── detect.py            # inference on image / directory / video
│   ├── evaluate.py          # mAP / precision / recall metrics
│   └── visualize_dataset.py # preview annotated image grid
└── outputs/
    ├── detections/          # saved inference results
    ├── training/            # training runs & weights
    └── evaluation/          # evaluation reports
```

## Requirements

- Python 3.9+
- PyTorch 2.0+

## Installation

```bash
pip install -r requirements.txt
```

Or use the Makefile shortcut:

```bash
make install
```

## Interactive Inference Notebook

[`inference_demo.ipynb`](inference_demo.ipynb) is a Jupyter notebook that walks through the complete inference workflow with visualizations:

- **Load a detector** with configurable confidence/IoU thresholds
- **Single image inference** — detect objects in one image and display results
- **Confidence tuning** — compare detection counts at different thresholds
- **Batch inference** — run on a directory and show a grid of results
- **Filter by class** — keep only specific object types
- **NumPy frame input** — detect from in-memory arrays (for camera/video pipelines)
- **Save results** — export annotated images and JSON metadata
- **Statistics** — visualize class distribution and confidence scores

**Launch the notebook:**

```bash
python3 -m jupyterlab inference_demo.ipynb
```

Or use the Makefile shortcut (after adding `~/Library/Python/3.9/bin` to PATH):

```bash
make notebook
```

## Workflow

### 1. Generate the dataset

The dataset is already generated. To regenerate it at any time:

```bash
make dataset

# or with custom options:
python3 scripts/generate_dataset.py --train 200 --val 50 --test 20 --img-size 640
```

### 2. Preview annotated images

```bash
make visualize
# saves a grid to outputs/dataset_preview.jpg

# or display interactively:
python3 scripts/visualize_dataset.py --split train --max 20
```

### 3. Train

Fine-tunes YOLOv8n on the synthetic dataset:

```bash
make train

# or with custom options:
python3 scripts/train.py --weights yolov8n.pt --epochs 50 --batch 16
```

Trained weights are saved to `outputs/training/yolo_synthetic/weights/best.pt`.

### 4. Run detection

```bash
# on test split using trained weights
make detect

# on test split using pretrained weights (no fine-tuning)
make detect-pretrained

# on a single image
python3 scripts/detect.py path/to/image.jpg --weights yolov8n.pt --show

# on a directory
python3 scripts/detect.py data/images/test/ --weights outputs/training/yolo_synthetic/weights/best.pt

# on a video
python3 scripts/detect.py myvideo.mp4 --weights yolov8n.pt

# with custom confidence threshold
python3 scripts/detect.py data/images/test/ --weights yolov8n.pt --conf 0.4 --iou 0.5
```

Results (annotated images, `.txt`, and `.json`) are saved to `outputs/detections/`.

### 5. Evaluate

```bash
make evaluate

# or against a specific weights file:
python3 scripts/evaluate.py --weights outputs/training/yolo_synthetic/weights/best.pt --split test
```

Prints mAP50, mAP50-95, precision, recall, and per-class AP50.

## Configuration

Edit [`config/config.yaml`](config/config.yaml) to change model, training, and output settings:

```yaml
model:
  weights: yolov8n.pt           # base weights to start from
  confidence_threshold: 0.5
  iou_threshold: 0.45
  input_size: 640

training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.01
  patience: 10                  # early stopping
  device: cpu                   # use "0" for GPU
```

Edit [`config/dataset.yaml`](config/dataset.yaml) to add classes or point to a different dataset.

## Adding your own images

1. Place images in `data/images/train/` and `data/images/val/`.
2. Create matching YOLO-format label files in `data/labels/train/` and `data/labels/val/`.  
   Each `.txt` file has one row per object: `class_id cx cy w h` (all values normalized 0–1).
3. Run `make train`.

## Makefile reference

| Command              | Description                                      |
|----------------------|--------------------------------------------------|
| `make install`       | Install Python dependencies                      |
| `make notebook`      | Launch Jupyter notebook for inference demo       |
| `make dataset`       | Generate synthetic dataset                       |
| `make visualize`     | Save annotated preview grid to `outputs/`        |
| `make train`         | Train YOLOv8n on the dataset                     |
| `make detect`        | Run inference on test split (trained weights)    |
| `make detect-pretrained` | Run inference using base YOLOv8n weights     |
| `make evaluate`      | Print mAP / precision / recall on test split     |
| `make clean`         | Delete all outputs and generated data            |

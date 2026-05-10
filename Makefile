PYTHON     = python3
WEIGHTS    = yolov8n.pt
BEST       = outputs/training/yolo_synthetic/weights/best.pt
TEST_IMG   = data/images/test

.PHONY: install dataset visualize train detect evaluate notebook clean

install:
	$(PYTHON) -m pip install -r requirements.txt

dataset:
	$(PYTHON) scripts/generate_dataset.py

visualize:
	$(PYTHON) scripts/visualize_dataset.py --split train --save outputs/dataset_preview.jpg

train:
	$(PYTHON) scripts/train.py --weights $(WEIGHTS)

detect:
	$(PYTHON) scripts/detect.py $(TEST_IMG) --weights $(BEST) --show

detect-pretrained:
	$(PYTHON) scripts/detect.py $(TEST_IMG) --weights $(WEIGHTS) --show

evaluate:
	$(PYTHON) scripts/evaluate.py --weights $(BEST)

notebook:
	jupyter lab inference_demo.ipynb

clean:
	rm -rf outputs/detections outputs/training outputs/evaluation
	find data -name "*.jpg" -o -name "*.txt" | xargs rm -f

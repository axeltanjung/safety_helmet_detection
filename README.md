# Safety Helmet Detection using YOLOv8

Real-time computer vision system for detecting **helmet compliance** in industrial and construction environments using **YOLOv8**.

---

## Problem Statement

According to OSHA & ILO statistics, head injuries remain one of the leading causes of fatal accidents in industrial sites. Manual supervision is:

- Error-prone  
- Non-scalable  
- Reactive (not preventive)

This project introduces automated real-time helmet compliance monitoring.

---

## Objectives

- Detect persons and helmets simultaneously  
- Classify each person as:
  - 🟢 Wearing Helmet  
  - 🔴 Not Wearing Helmet  
- Support CCTV / RTSP / video inference  
- Provide visual and structured alerts  

---

## Model Architecture

| Component | Description |
|----------|------------|
| Backbone | YOLOv8 |
| Detection Head | Multi-class object detection |
| Classes | `person`, `helmet` |
| Framework | Ultralytics YOLOv8 |
| Inference Speed | 40–80 FPS (GPU) |

---

## 🗂 Dataset Structure

datasets/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/


YOLO label format:

<class_id> <x_center> <y_center> <width> <height>


Classes:
- 0 → person
- 1 → helmet


---

## 🏋️ Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="helmet.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)

Output

- Bounding box visualization
- Compliance labels
- Violation count
- Timestamped alert logs

Applications

- Construction sites
- Mining
- Manufacturing
- Oil & Gas
- Smart City CCTV

Performance
Metric	Value
mAP@0.5	> 90%
Precision	> 92%
Recall	> 90%
FPS (GPU)	60+
🛡 Limitations
Issue	Cause
Missed helmets	Occlusion
False positives	Helmet-colored objects
Low-light errors	Night conditions
🛠 Future Work

Multi-PPE detection

Person-helmet tracking

Alert automation (Telegram/Email)

Edge deployment (Jetson / RK3588)

📦 Requirements
ultralytics
opencv-python
torch
numpy

✨ Author

Axel Ivanda Tanjung
Data Scientist – Industrial AI & Computer Vision
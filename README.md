# 🌿 AgroAI

This project is an intelligent agricultural diagnostic solution that combines the power of **MobileNetV2** for classification and **YOLOv2** for object detection. It identifies plant diseases and insect pests through a modern Graphical User Interface (GUI).

## 🚀 Features
- **Accurate Classification**: Uses MobileNetV2 to identify over 58 categories of plants and insects.
- **Object Detection**: Integration of YOLOv2 to localize affected areas on leaves.
- **User Interface (GUI)**: Developed with Tkinter for a professional and user-friendly experience.
- **Real-Time Analysis**: Supports live camera feeds via OpenCV.
- **Automated Reporting**: Generates a detailed analysis report including confidence levels and agricultural recommendations.

---

## 🧠 Models Used

### 1. MobileNetV2 (Training & Classification)
The classification model is designed to be lightweight and efficient:
- **Architecture**: MobileNetV2 utilizing Transfer Learning (pre-trained ImageNet weights).
- **Optimization**: Includes a Dropout layer (0.5) and a Dense layer (512 units) for specialized feature extraction.
- **Training**: Built with TensorFlow/Keras using Data Augmentation (Rotation, Zoom, Shift, Flip).
- **Performance**: Evaluated via Confusion Matrix and Classification Reports (Precision/Recall/F1-Score).

### 2. YOLOv2 (Detection & Inference)
The YOLO model is used within the final application to:
- Dynamically detect diseased leaves in images or live video streams.
- Provide bounding boxes around specific infections or pests.

---

## 🛠️ Installation and Configuration

### 1. Prerequisites
- Python 3.8 or higher
- A virtual environment is recommended: `python -m venv env`

### 2. Install Dependencies
```bash
pip install tensorflow opencv-python pillow ultralytics torch scikit-learn seaborn matplotlib

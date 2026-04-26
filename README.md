# 🚗 Driver Drowsiness Detection & Alertness System

[![CI/CD Pipeline](https://github.com/OmerKhan33/driver-drowsiness-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/OmerKhan33/driver-drowsiness-detection-system/actions/workflows/ci.yml)

> A real-time, end-to-end driver drowsiness detection system using YOLO face detection, CNN-based classification, and physiological signal analysis (EAR/MAR) to alert drowsy drivers and prevent accidents.

![Demo GIF](models/results/demo_placeholder.gif)

---

## 📑 Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Model Comparison Results](#model-comparison-results)
- [GitHub Actions Pipeline](#github-actions-pipeline)
- [File Descriptions](#file-descriptions)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## 🎯 Project Overview

**Problem Statement:** Driver drowsiness is a leading cause of road accidents worldwide. The National Highway Traffic Safety Administration (NHTSA) estimates that drowsy driving causes over 100,000 crashes annually in the US alone.

**Solution:** This system provides a multi-signal approach to drowsiness detection:

1. **Face Detection** — Uses YOLO (v8, v9, v11, v12) to locate the driver's face in real-time video frames.
2. **CNN Classification** — Classifies detected faces as ALERT or DROWSY using fine-tuned deep learning architectures (ResNet, EfficientNet, MobileNet, VGG, CustomCNN).
3. **Physiological Signals** — Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) via MediaPipe face landmarks for secondary drowsiness confirmation.
4. **Alert System** — Triggers audio and visual alerts when drowsiness is detected over consecutive frames.

---

## 🔄 Pipeline Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Webcam /    │     │   YOLO Face      │     │  CNN Classifier   │
│  Video Feed  │────▶│   Detector       │────▶│  (Alert/Drowsy)   │
│              │     │  (v8/v9/v11/v12) │     │  ResNet/EfficNet  │
└─────────────┘     └──────────────────┘     └───────────────────┘
                            │                          │
                            ▼                          ▼
                    ┌──────────────────┐     ┌───────────────────┐
                    │  MediaPipe Face  │     │  Drowsiness Score │
                    │  Landmarks       │     │  Computation      │
                    │  (468 points)    │     │  (Weighted Combo) │
                    └──────────────────┘     └───────────────────┘
                            │                          │
                            ▼                          ▼
                    ┌──────────────────┐     ┌───────────────────┐
                    │  EAR + MAR       │     │  Alert System     │
                    │  Calculation     │────▶│  (Audio + Visual) │
                    └──────────────────┘     └───────────────────┘
```

---

## 🛠️ Tech Stack

| Category          | Technology                                      |
|-------------------|------------------------------------------------|
| Language          | Python 3.12                                     |
| Deep Learning     | PyTorch, Torchvision                            |
| Face Detection    | Ultralytics YOLOv8, v9, v11, v12                |
| Face Landmarks    | MediaPipe                                       |
| Image Processing  | OpenCV (cv2)                                    |
| Metrics           | Scikit-learn                                    |
| Visualization     | Matplotlib, Seaborn                             |
| Web App           | Streamlit                                       |
| Testing           | pytest, pytest-cov                              |
| Linting           | flake8, black, isort                            |
| CI/CD             | GitHub Actions                                  |
| Containerization  | Docker                                          |
| Experiment Track  | MLflow                                          |

---

## 📁 Repository Structure

```
driver-drowsiness-detection-system/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  ← CI/CD pipeline (lint → test → model-check)
│
├── data/
│   ├── raw/
│   │   └── sample_frames/          ← Sample frames for testing
│   ├── processed/
│   │   ├── train/
│   │   │   ├── alert/              ← Alert class training images
│   │   │   └── drowsy/             ← Drowsy class training images
│   │   ├── val/
│   │   │   ├── alert/
│   │   │   └── drowsy/
│   │   └── test/
│   │       ├── alert/
│   │       └── drowsy/
│   └── scripts/
│       └── prepare_dataset.py      ← Dataset preparation from Kaggle download
│
├── notebooks/
│   └── model_comparison.ipynb      ← Full model comparison (YOLO + CNN)
│
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py        ← Image transforms & frame utilities
│   │   ├── drowsiness_utils.py     ← EAR/MAR computation & scoring
│   │   └── sanity_check.py         ← CI model verification script
│   ├── detection/
│   │   ├── __init__.py
│   │   └── face_detector.py        ← YOLO & Haar face detectors
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── model_builder.py        ← CNN architecture builder
│   │   ├── train.py                ← Training loop with AMP & early stopping
│   │   └── predict.py              ← Inference predictor class
│   └── alert/
│       ├── __init__.py
│       └── alert_system.py         ← Alert system with sound & visuals
│
├── models/
│   ├── weights/                    ← Saved model weights (.pt)
│   └── results/                    ← Metrics, plots, CSVs
│
├── app/
│   └── main.py                     ← Streamlit real-time demo app
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py            ← Comprehensive pytest suite
│
├── .gitignore
├── requirements.txt
├── README.md
└── Dockerfile
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/OmerKhan33/driver-drowsiness-detection-system.git
cd driver-drowsiness-detection-system
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Prepare Dataset (after Kaggle download)

```bash
python data/scripts/prepare_dataset.py
```

### 4. Run Smoke Tests

```bash
python src/utils/preprocessing.py          # All ✓ checks
python src/utils/drowsiness_utils.py       # All ✓ checks
python src/classification/model_builder.py # Model parameter table
```

### 5. Run Sanity Check

```bash
python src/utils/sanity_check.py           # All checks PASSED
```

### 6. Run Test Suite

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 7. Train Models

```bash
python src/classification/train.py --epochs 15 --batch_size 32 --lr 0.0001
```

### 8. Launch Demo App

```bash
streamlit run app/main.py
```

### 9. Run Comparison Notebook

```bash
jupyter notebook notebooks/model_comparison.ipynb
```

---

## 📊 Dataset Setup

This project uses the [Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) by dheerajperumandla on Kaggle.

### Download Instructions

1. **Option A — Kaggle CLI:**
   ```bash
   pip install kaggle
   kaggle datasets download -d dheerajperumandla/drowsiness-dataset
   unzip drowsiness-dataset.zip -d data/raw/
   ```

2. **Option B — Manual Download:**
   - Visit: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
   - Download and extract to `data/raw/drowsiness-dataset/`

3. **Run Preparation Script:**
   ```bash
   python data/scripts/prepare_dataset.py
   ```

The script maps:
- `Open_Eyes` + `no_yawn` → **ALERT** class
- `Closed_Eyes` + `Yawn` → **DROWSY** class

And splits into: **Train (70%)** / **Val (15%)** / **Test (15%)**

---

## 📈 Model Comparison Results

### Face Detection Models

| Model     | Avg Time (ms) | FPS    | Parameters |
|-----------|---------------|--------|------------|
| YOLOv8n   | TBD           | TBD    | 3.2M       |
| YOLOv9c   | TBD           | TBD    | 25.3M      |
| YOLOv11n  | TBD           | TBD    | 2.6M       |
| YOLOv12n  | TBD           | TBD    | 2.6M       |
| Haar      | TBD           | TBD    | N/A        |

### Classification Models

| Model          | Accuracy | F1 Score | Precision | Recall | AUC   | Params  |
|----------------|----------|----------|-----------|--------|-------|---------|
| CustomCNN      | TBD      | TBD      | TBD       | TBD    | TBD   | ~0.5M   |
| ResNet18       | TBD      | TBD      | TBD       | TBD    | TBD   | 11.2M   |
| ResNet50       | TBD      | TBD      | TBD       | TBD    | TBD   | 23.5M   |
| VGG16          | TBD      | TBD      | TBD       | TBD    | TBD   | 134.3M  |
| EfficientNet-B0| TBD      | TBD      | TBD       | TBD    | TBD   | 4.0M    |
| MobileNetV2    | TBD      | TBD      | TBD       | TBD    | TBD   | 2.2M    |

*Results will be populated after running the model comparison notebook.*

---

## ⚙️ GitHub Actions Pipeline

The CI/CD pipeline runs automatically on every push and pull request with 3 sequential jobs:

```
lint → test → model-check
```

1. **Lint** — Runs flake8, black, and isort checks on all Python code
2. **Test** — Executes the full pytest suite with coverage reporting
3. **Model Check** — Runs `sanity_check.py` to verify all model architectures load and produce correct outputs

---

## 📄 File Descriptions

| File | Description |
|------|-------------|
| `src/utils/preprocessing.py` | Image preprocessing, transforms, and frame utilities |
| `src/utils/drowsiness_utils.py` | EAR/MAR calculation, drowsiness scoring functions |
| `src/utils/sanity_check.py` | Model architecture verification for CI pipeline |
| `src/detection/face_detector.py` | YOLO and Haar Cascade face detector wrappers |
| `src/classification/model_builder.py` | CNN model builder (6 architectures) |
| `src/classification/train.py` | Training loop with AMP, early stopping, checkpointing |
| `src/classification/predict.py` | Inference predictor for single/batch predictions |
| `src/alert/alert_system.py` | Drowsiness alert system with audio/visual feedback |
| `data/scripts/prepare_dataset.py` | Dataset restructuring from Kaggle format |
| `app/main.py` | Streamlit real-time demo application |
| `tests/test_pipeline.py` | Comprehensive pytest test suite |
| `notebooks/model_comparison.ipynb` | Full YOLO + CNN model comparison notebook |

---

## 📊 Results

Results from model training and comparison will be saved to `models/results/`:

- `detection_comparison.csv` — Face detection speed/accuracy comparison
- `classification_comparison.csv` — CNN model performance metrics
- `training_curves.png` — Loss and accuracy curves for all models
- `confusion_matrices.png` — Confusion matrices for all classifiers
- `roc_curves.png` — ROC curves with AUC scores
- `speed_vs_accuracy.png` — Accuracy vs inference speed scatter plot

---

## 🔮 Future Work

- [ ] Add driver identity verification (face recognition)
- [ ] Implement attention tracking using gaze estimation
- [ ] Add support for night-time / infrared cameras
- [ ] Deploy as edge application on Raspberry Pi / NVIDIA Jetson
- [ ] Integrate with vehicle CAN bus for automatic speed reduction
- [ ] Add multi-driver support for fleet monitoring
- [ ] Implement temporal modeling (LSTM/Transformer) for sequence-based detection
- [ ] Add phone usage detection as additional distraction signal

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ for road safety
</p>

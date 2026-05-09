# Driver Drowsiness Detection & Alertness System

[![CI/CD Pipeline](https://github.com/OmerKhan33/driver-drowsiness-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/OmerKhan33/driver-drowsiness-detection-system/actions/workflows/ci.yml)

> A real-time, end-to-end driver drowsiness detection system combining CNN-based classification, MediaPipe facial-landmark physiological signals (EAR / MAR), and a piercing audio alarm. Ships with a role-based Streamlit web app, SQLite event logging, and a Docker image for one-command deployment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Running with Docker](#running-with-docker)
- [Dataset Setup](#dataset-setup)
- [Model Comparison Results](#model-comparison-results)
- [GitHub Actions Pipeline](#github-actions-pipeline)
- [File Descriptions](#file-descriptions)
- [Future Work](#future-work)
- [License](#license)

---

## Project Overview

**Problem.** Drowsy driving causes over 100,000 crashes annually in the US alone (NHTSA). A reliable, low-latency in-cabin monitor that runs on commodity hardware can flag the dangerous state before the driver loses control.

**Solution — multi-signal fusion:**

1. **Face landmarks** — MediaPipe FaceLandmarker (Tasks API, 478 points) locates eyes and mouth on every frame.
2. **Physiological signals** — Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) are computed from landmarks and exponentially smoothed.
3. **CNN classification** — A fine-tuned CNN (selectable from 6 backbones) produces an independent drowsy probability from the cropped face.
4. **Drowsiness scoring** — EAR, MAR, and CNN confidence are fused into a single score; a state machine over consecutive frames triggers the alarm.
5. **Alert system** — Browser-side WebAudio yelp siren (frequency-sweeping 800 → 1600 Hz) — the same pattern emergency vehicles use, designed to disrupt drowsy attention.

A **Haar cascade fallback** kicks in automatically when MediaPipe is unavailable, so the app degrades gracefully.

---

## Pipeline Architecture

```
┌──────────────────┐     ┌──────────────────────┐
│  Browser Webcam  │     │   MediaPipe Face     │
│  (streamlit-     │────▶│   Landmarker         │
│   webrtc)        │     │   (478 landmarks)    │
└──────────────────┘     └──────────┬───────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
      ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
      │  EAR (eyes)  │     │  MAR (mouth) │     │  CNN Classifier │
      │  per-frame   │     │  per-frame   │     │  (6 backbones)  │
      └──────┬───────┘     └──────┬───────┘     └────────┬────────┘
             └─────────────┬──────┴───────────────┬──────┘
                           ▼                      ▼
                  ┌──────────────────┐  ┌────────────────────┐
                  │  EMA smoothing   │  │ Drowsiness score   │
                  │  + state machine │  │ (weighted fusion)  │
                  └────────┬─────────┘  └────────┬───────────┘
                           ▼                     ▼
                  ┌──────────────────────────────────────┐
                  │  Alert: yelp siren + visual overlay  │
                  │       SQLite event logging           │
                  └──────────────────────────────────────┘
```

---

## Tech Stack

| Category          | Technology                                      |
|-------------------|-------------------------------------------------|
| Language          | Python 3.12                                     |
| Deep Learning     | PyTorch, Torchvision                            |
| Face Landmarks    | MediaPipe Tasks API (FaceLandmarker)            |
| Image Processing  | OpenCV, CLAHE                                   |
| Web App           | Streamlit, streamlit-webrtc, streamlit-autorefresh |
| Storage           | SQLite (auth, sessions, events)                 |
| Audio Alarm       | Browser WebAudio API (yelp siren)               |
| Remote Access     | pyngrok                                         |
| Testing           | pytest, pytest-cov                              |
| Linting           | flake8, black, isort                            |
| CI/CD             | GitHub Actions (lint → test → model-check + docker-build) |
| Containerization  | Docker (multi-stage, CPU-only PyTorch)          |
| Experiment Track  | MLflow                                          |

---

## Repository Structure

```
driver-drowsiness-detection-system/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  ← lint → (test → model-check) + docker-build
│
├── app/
│   ├── streamlit_app.py            ← Main Streamlit web app (role-based UI)
│   ├── run_app.py                  ← Launcher with optional ngrok tunnel
│   ├── database.py                 ← SQLite layer (auth, sessions, events)
│   └── _shared.py                  ← Cross-thread Events for the alarm flag
│
├── src/
│   ├── alert/
│   │   └── alert_system.py         ← State machine for drowsy / yawn detection
│   ├── classification/
│   │   ├── model_builder.py        ← 6 CNN backbones + classifier head
│   │   ├── train.py                ← Training loop (AMP, early stopping)
│   │   └── predict.py              ← Inference predictor
│   ├── detection/
│   │   └── face_detector.py        ← Haar cascade fallback
│   └── utils/
│       ├── preprocessing.py        ← Image transforms & frame utilities
│       ├── drowsiness_utils.py     ← EAR / MAR / score computation
│       └── sanity_check.py         ← CI architecture verification
│
├── data/
│   ├── raw/                        ← Kaggle dataset extraction target
│   ├── processed/                  ← train / val / test splits (alert / drowsy)
│   ├── driver_drowsiness.db        ← SQLite DB (created at runtime)
│   └── scripts/
│       ├── prepare_dataset.py      ← Kaggle → train/val/test splitter
│       ├── build_manifest.py       ← Manifest builder for processed splits
│       └── validate_dataset.py     ← Dataset integrity checker
│
├── models/
│   ├── face_landmarker.task        ← MediaPipe Tasks landmark model
│   ├── MODEL_CARD.md               ← Trained model documentation
│   ├── weights/                    ← Trained .pt weights for all 6 backbones
│   └── results/
│       ├── training_summary.json   ← Best val acc + epoch + training time
│       └── sanity_check.json       ← CI sanity-check output
│
├── notebooks/
│   └── 01_model_training.ipynb     ← End-to-end training notebook
│
├── tests/
│   └── test_pipeline.py            ← pytest suite (unit + integration)
│
├── Dockerfile                      ← Multi-stage CPU image
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/OmerKhan33/driver-drowsiness-detection-system.git
cd driver-drowsiness-detection-system
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
python -m streamlit run app/streamlit_app.py
```

Then open <http://localhost:8501> in your browser.

> **Default admin login** — create a driver account on the **Create Account** tab; the admin role is provisioned by the database layer (see `app/database.py`).

### 4. (Optional) Expose via ngrok

```bash
python app/run_app.py --ngrok-token YOUR_NGROK_AUTH_TOKEN
```

### 5. Run the test suite

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 6. Run the sanity check

```bash
python src/utils/sanity_check.py
```

### 7. Train models (optional — pretrained weights are committed)

```bash
python src/classification/train.py --epochs 15 --batch_size 32 --lr 0.0001
```

---

## Running with Docker

The project ships a multi-stage Dockerfile that uses CPU-only PyTorch wheels for a lean image.

```bash
# Build
docker build -t drowsiness-detector .

# Run (port 8501, persist SQLite DB to a host volume)
docker run -p 8501:8501 -v drowsi-data:/app/data drowsiness-detector
```

The container has a `HEALTHCHECK` that verifies the PyTorch import, and Streamlit is started in headless mode bound to `0.0.0.0:8501`.

---

## Dataset Setup

This project uses the [Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) on Kaggle.

### Download & prepare

```bash
# Option A — Kaggle CLI
pip install kaggle
kaggle datasets download -d dheerajperumandla/drowsiness-dataset
unzip drowsiness-dataset.zip -d data/raw/

# Option B — manual
# Download from the URL above and extract to data/raw/

# Then run the splitter
python data/scripts/prepare_dataset.py
python data/scripts/validate_dataset.py
```

The splitter maps Kaggle classes to the project taxonomy:
- `Open_Eyes` + `no_yawn` → **ALERT**
- `Closed_Eyes` + `Yawn` → **DROWSY**

with a **70 / 15 / 15** train / val / test split.

---

## Model Comparison Results

Pretrained weights for all six classifiers are committed under [`models/weights/`](models/weights/). Results are taken from [`models/results/training_summary.json`](models/results/training_summary.json) (15-epoch budget, early stopping on val loss).

| Model            | Best Val Accuracy | Best Epoch | Training Time | Approx. Params |
|------------------|------------------:|-----------:|--------------:|---------------:|
| **ResNet50**     |        **100.0%** |         11 |        7m 07s |          23.5M |
| ResNet18         |             98.6% |          9 |        5m 24s |          11.2M |
| MobileNetV2      |             97.2% |          9 |        5m 24s |           2.2M |
| EfficientNet-B0  |             96.8% |         13 |        6m 04s |           4.0M |
| VGG16            |             94.5% |          9 |        6m 18s |         134.3M |
| CustomCNN        |             56.0% |          6 |        4m 35s |          ~0.5M |

> **In production** the app defaults to MobileNetV2 — best accuracy-to-latency trade-off. Switch backbones in the driver dashboard sidebar.

---

## GitHub Actions Pipeline

Every push and pull request triggers four jobs in [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

```
lint ─┬─▶ test ─▶ model-check
      └─▶ docker-build
```

1. **lint** — `flake8` + `black --check` + `isort --check` over `src/`, `tests/`, `app/`.
2. **test** — installs CPU PyTorch wheels and runs `pytest` with coverage.
3. **model-check** — runs `src/utils/sanity_check.py`, uploads results as artifact.
4. **docker-build** — builds the production image with Buildx + GHA cache, then smoke-tests `python -c "import torch, cv2, mediapipe, streamlit"` inside the container.

---

## File Descriptions

| File | Description |
|------|-------------|
| `app/streamlit_app.py` | Main web app — role-based login, driver dashboard with live webcam, admin control panel |
| `app/run_app.py` | Launcher that starts Streamlit and opens an ngrok tunnel for remote access |
| `app/database.py` | SQLite layer for users, sessions, and drowsiness events |
| `app/_shared.py` | Module-cached `threading.Event` flags so the alarm survives Streamlit re-runs |
| `src/alert/alert_system.py` | Consecutive-frame state machine — drowsy / yawn / alert decisions |
| `src/classification/model_builder.py` | Six CNN backbones with a unified two-class head |
| `src/classification/train.py` | Training loop with AMP, early stopping, checkpointing |
| `src/classification/predict.py` | Inference predictor for cropped face frames |
| `src/detection/face_detector.py` | Haar cascade fallback when MediaPipe is unavailable |
| `src/utils/preprocessing.py` | Transforms, CLAHE, resize / normalize utilities |
| `src/utils/drowsiness_utils.py` | EAR, MAR, drowsiness-score formulas, level classifier |
| `src/utils/sanity_check.py` | CI architecture verification |
| `data/scripts/prepare_dataset.py` | Kaggle dataset → 70/15/15 splits in `data/processed/` |
| `data/scripts/build_manifest.py` | Generates a CSV manifest of the processed splits |
| `data/scripts/validate_dataset.py` | Verifies split integrity (counts, no leakage) |
| `notebooks/01_model_training.ipynb` | End-to-end training & comparison notebook |
| `tests/test_pipeline.py` | pytest suite — preprocessing, EAR/MAR, models, alert state machine |
| `Dockerfile` | Multi-stage build, CPU-only PyTorch, ~2 GB image |

---

## Future Work

- [ ] Driver identity verification via face recognition
- [ ] Gaze-direction estimation as a secondary attention signal
- [ ] Infrared / night-time camera support
- [ ] Edge deployment on Raspberry Pi 5 / NVIDIA Jetson Orin Nano
- [ ] Vehicle CAN-bus integration for haptic / speed-limit response
- [ ] Multi-driver fleet view in the admin panel
- [ ] Temporal modelling (LSTM / Transformer) over landmark sequences
- [ ] Phone-usage / hands-off-wheel detection

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
# Model Card — Driver Drowsiness Classification

## Overview

Binary CNN classifier predicting `alert` vs `drowsy` from a face crop.
Used as one of three signals in the live pipeline alongside MediaPipe-derived
EAR (eye aspect ratio) and MAR (mouth aspect ratio).

- **Task:** binary image classification
- **Inputs:** RGB face crop, 224×224, ImageNet-normalized
- **Outputs:** softmax over `["alert", "drowsy"]`
- **Frameworks:** PyTorch 2.x, torchvision

## Architectures evaluated

Built via `src/classification/model_builder.py`:

| Name | Backbone | Pretrained | Trainable head |
|---|---|---|---|
| CustomCNN | from scratch | no | full |
| ResNet18 | ImageNet | yes | layer4 + fc |
| ResNet50 | ImageNet | yes | layer3 + layer4 + fc |
| VGG16 | ImageNet | yes | classifier head |
| EfficientNet-B0 | ImageNet | yes | last 2 blocks + classifier |
| MobileNetV2 | ImageNet | yes | last 3 blocks + classifier |

Default classifier head: `Dropout(0.4) → Linear(num_classes)`
(ResNet50 / VGG16 use a wider 2-layer head — see `model_builder.py`).

## Training procedure

Implemented in `src/classification/train.py`.

| Setting | Value |
|---|---|
| Optimizer | AdamW (`weight_decay=1e-4`) |
| LR schedule | CosineAnnealingLR (`eta_min=1e-6`) |
| Loss | CrossEntropyLoss |
| Batch size | 32 (default, CLI overridable) |
| Epochs | 15 (default, CLI overridable) |
| Image size | 224×224 |
| Mixed precision | `torch.amp` autocast on CUDA |
| Gradient clip | max_norm = 1.0 |
| Best-checkpoint metric | val accuracy |
| Early stopping | patience = 5 |

Per-run metrics (`best_val_acc`, `best_epoch`, `training_time_s`) land in
[`models/results/training_summary.json`](results/training_summary.json).
When MLflow is enabled, the same metrics + final weights are logged to
the local `mlruns/` tracking store.

## Data

Source: [Drowsiness Dataset by dheerajperumandla on Kaggle](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset).

Class mapping applied in [`data/scripts/prepare_dataset.py`](../data/scripts/prepare_dataset.py):

| Raw folder | Mapped class |
|---|---|
| `Open_Eyes`, `no_yawn` | `alert` |
| `Closed_Eyes`, `Yawn` | `drowsy` |

Splits: 70% train / 15% val / 15% test (`random_seed=42`).
Validate with `python data/scripts/validate_dataset.py` and snapshot with
`python data/scripts/build_manifest.py` (writes `data_manifest.json` —
two runs with the same `dataset_hash` describe the same data).

## Intended use

- **In-cabin driver-monitoring research** — alongside EAR/MAR + alert system
  hysteresis (5 consecutive frames before flagging drowsy).
- **NOT** intended as a standalone safety-critical decision maker. The live
  app combines CNN confidence with physiological signals via
  `compute_drowsiness_score()` and applies temporal smoothing before alerting.

## Known limitations

- Trained on a single public dataset — generalization to night-time, glasses,
  diverse skin tones, and partial face occlusion has **not** been audited.
- The CNN sees only a tight face crop; head pose and gaze direction are not
  modeled (pipeline supplements with EAR/MAR/tilt instead).
- No per-driver calibration. Some users have a closed-eye EAR above the
  default 0.23 threshold and require admin-side adjustment.

## Reproducing a run

```bash
# 1. Prepare data
python data/scripts/prepare_dataset.py
python data/scripts/validate_dataset.py
python data/scripts/build_manifest.py

# 2. Train (single architecture)
python src/classification/train.py --models ResNet18 --epochs 15

# 3. Sanity check
python src/utils/sanity_check.py
```

## Files

- Architectures: [`src/classification/model_builder.py`](../src/classification/model_builder.py)
- Training loop: [`src/classification/train.py`](../src/classification/train.py)
- Inference: [`src/classification/predict.py`](../src/classification/predict.py)
- Saved weights: `models/weights/<name>_best.pt` (gitignored — too large for the repo)
- Sanity check report: [`models/results/sanity_check.json`](results/sanity_check.json)
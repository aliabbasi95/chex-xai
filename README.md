# CheX-XAI 🩺
**Explainable AI for Chest X-ray Diagnosis**

---

## 📘 Overview
**CheX-XAI** is a modular PyTorch-based framework designed for training and evaluating deep learning models on the **CheXpert** chest X-ray dataset.
It supports configurable training pipelines, explainable evaluation, mixed datasets (raw + masked), and standardized logging and reproducibility features.

This repository implements **Stage 1–3** of the baseline pipeline, including:
- Data loading and preprocessing
- Model training with mixed precision (AMP)
- Evaluation with AUROC and per-class thresholds (F1 optimization)
- Automated artifact logging (metrics, config, environment)

---

## ⚙️ Project Structure
```
chex-xai/
│
├── configs/
│   ├── train.yaml        # Training hyperparameters and setup
│   └── paths.yaml        # Data and output paths
│
├── src/chex_xai/         # Main Python package
│   ├── data/             # Dataset loaders and preprocessing
│   ├── models/           # Model architectures (DenseNet121, etc.)
│   ├── engine/           # Training and evaluation loops
│   ├── metrics/          # AUROC and F1 computation utilities
│   └── utils/            # Helper functions (logging, seeding, checkpointing)
│
├── scripts/              # Standalone executable scripts
│   ├── train.py          # Training entrypoint
│   ├── eval.py           # Evaluation with AUROC and F1
│   ├── compute_thresholds.py  # Compute optimal F1 thresholds on dev set
│   └── dump_class_support.py  # Generate class statistics (pos/neg)
│
└── outputs/
    └── chexpert_baseline_v1/
        ├── best.pt / last.pt             # Saved model weights
        ├── metrics_eval.json             # AUROC + F1 metrics
        ├── thresholds.json               # Per-class threshold values
        ├── class_support.json            # Class frequency summary
        ├── config_resolved.yaml          # Resolved training config
        └── env.txt                       # Python/Torch environment info
```

---

## 🚀 Quick Start

### 1️⃣ Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ Data Preparation
Organize your dataset as follows:
```
data/
├── splits/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── CheXpert-v1.0/
│   ├── train/
│   ├── valid/
│   └── test/
└── data_masked/   # (optional) preprocessed masked versions
```

Update paths in `configs/paths.yaml` if necessary.

---

### 3️⃣ Training
Train the baseline model (DenseNet121, 320×320):
```bash
python scripts/train.py
```

This saves checkpoints and logs to:
```
outputs/chexpert_baseline_v1/
```

---

### 4️⃣ Evaluation
Run full evaluation with AUROC and per-class F1:

```bash
python scripts/compute_thresholds.py --ckpt outputs/chexpert_baseline_v1/best.pt --exp outputs/chexpert_baseline_v1
python scripts/dump_class_support.py
python scripts/eval.py --ckpt outputs/chexpert_baseline_v1/best.pt --amp \
  --thresholds outputs/chexpert_baseline_v1/thresholds.json --exp outputs/chexpert_baseline_v1
```

Results will be stored in:
```
outputs/chexpert_baseline_v1/metrics_eval.json
```

---

## 📊 Baseline Results

| Metric | Dev | Test |
|:-------|:----|:----|
| AUROC (macro) | **0.712** | **0.720** |
| AUROC (micro) | 0.722 | 0.683 |
| F1 (macro, thresholds) | 0.357 | 0.388 |
| F1 (micro, thresholds) | 0.540 | 0.534 |

> Model: **DenseNet121** (ImageNet pretrained, 320×320)
> Hardware: **NVIDIA A100 40GB**, CUDA 12.8
> Dataset: **CheXpert**, 13 pathology labels.

---

## 🧬 Reproducibility
Each experiment automatically logs:
- `config_resolved.yaml` → complete training configuration
- `env.txt` → environment info (Python, Torch, CUDA)
- `metrics_eval.json` → metrics summary
- `thresholds.json` → per-class optimal thresholds

All random seeds are fixed (`seed: 42`) for reproducible results.


---

## 📜 License
This repository is provided for academic and research purposes only.
© 2025 — Developed by **Ali Abbasi** and contributors.

# src/data/chexpert.py

"""
CheXpert Dataset Loader (chex_xai.data.chexpert)
------------------------------------------------
- Reads split CSVs (train/dev/test) with a 'Path' column (relative to a root dir).
- Maps uncertainty labels to numeric values via `_map_value`.
- Returns grayscale images (PIL -> tensor) with torchvision-style transforms.
- Exposes `build_loaders(cfg)` to create train/dev/test DataLoaders.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .transforms import build_transforms

CHEXPERT_LABELS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    ]
PATIENT_RE = re.compile(r"(patient\d+)", re.IGNORECASE)

def extract_patient_id(path_str: str) -> str:
    m = PATIENT_RE.search(path_str.replace("\\", "/"))
    if m: return m.group(1)
    parts = path_str.replace("\\","/").split("/")
    cand = [p for p in parts if p.lower().startswith("patient")]
    return cand[0] if cand else "patientUNK"

@dataclass
class CheXpertConfig:
    data_root: str = "data/chexpert_small"
    splits_dir: str = "data/splits"
    img_size: int = 320
    batch_size: int = 32
    num_workers: int = 8
    u_policy: str = "zeros"  # 'zeros' | 'ones' | 'ignore' (baseline: zeros)

class CheXpertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, img_size: int, is_train: bool, u_policy: str = "zeros"):
        self.df = df.reset_index(drop=True).copy()
        self.root = Path(root)
        self.is_train = is_train
        self.u_policy = u_policy
        self.labels = [c for c in CHEXPERT_LABELS if c in df.columns]
        self.tfm = build_transforms(img_size=img_size, is_train=is_train)

        self.Y = self._build_targets(self.df[self.labels])

    def _map_value(self, v):
        # v ∈ {1, 0, -1, NaN}
        if pd.isna(v): return 0.0
        if v == 1: return 1.0
        if v == 0: return 0.0
        if v == -1:
            return 0.0 if self.u_policy == "zeros" else (1.0 if self.u_policy == "ones" else 0.0)
        return float(v)

    def _build_targets(self, subdf: pd.DataFrame) -> torch.Tensor:
        mapped = (
             subdf.apply(lambda col: col.map(self._map_value))
             .astype("float32")
             .values
        )
        return torch.from_numpy(mapped)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root / row["Path"]
        if not img_path.exists():
            img_path = self.root / Path(row["Path"]).name  # کمکی
        img = Image.open(img_path).convert("L")  # خاکستری
        x = self.tfm(img)                        # → 3×H×W

        y = self.Y[idx]
        pid = row.get("patient_id", extract_patient_id(str(row["Path"])))
        return {"image": x, "target": y, "path": str(img_path), "patient_id": pid}

def build_loaders(cfg: CheXpertConfig):
    splits = Path(cfg.splits_dir)
    root   = cfg.data_root

    df_train = pd.read_csv(splits / "train.csv")
    df_dev   = pd.read_csv(splits / "dev.csv")
    df_test  = pd.read_csv(splits / "test.csv")

    ds_train = CheXpertDataset(df_train, root=root, img_size=cfg.img_size, is_train=True,  u_policy=cfg.u_policy)
    ds_dev   = CheXpertDataset(df_dev,   root=root, img_size=cfg.img_size, is_train=False, u_policy=cfg.u_policy)
    ds_test  = CheXpertDataset(df_test,  root=root, img_size=cfg.img_size, is_train=False, u_policy=cfg.u_policy)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, pin_memory=True)
    dl_dev   = DataLoader(ds_dev,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return ds_train, ds_dev, ds_test, dl_train, dl_dev, dl_test


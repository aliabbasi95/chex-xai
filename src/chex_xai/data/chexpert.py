# src/chex_xai/data/chexpert.py

"""
CheXpert Dataset Loader (chex_xai.data.chexpert)
------------------------------------------------
- Reads split CSVs (train/dev/test) with a 'Path' column (relative to a root dir).
- Maps uncertainty labels to numeric values via `_map_value`.
- Returns grayscale images (PIL -> tensor) with torchvision-style transforms.
- Exposes `build_loaders(cfg)` to create train/dev/test DataLoaders.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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
    """Extract 'patientXXXX' from a CheXpert path; fallback to 'patientUNK'."""
    s = path_str.replace("\\", "/")
    m = PATIENT_RE.search(s)
    if m:
        return m.group(1)
    parts = s.split("/")
    cand = [p for p in parts if p.lower().startswith("patient")]
    return cand[0] if cand else "patientUNK"


@dataclass
class CheXpertConfig:
    """
    Configuration for building CheXpert datasets/loaders.
    """

    root: str
    csv_train: str
    csv_dev: str
    csv_test: str
    img_size: int = 320
    batch_size: int = 32
    num_workers: int = 8
    u_policy: str = "zeros"  # 'zeros' | 'ones' | 'ignore'
    augment: bool = True  # whether to use train-time augmentations


class CheXpertDataset(Dataset):
    """Torch Dataset for CheXpert with grayscale reading and basic transforms."""

    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        img_size: int,
        is_train: bool,
        u_policy: str = "zeros",
    ):
        self.df = df.reset_index(drop=True).copy()
        self.root = Path(root)
        self.is_train = is_train
        self.u_policy = u_policy
        self.labels = [c for c in CHEXPERT_LABELS if c in self.df.columns]
        self.tfm = build_transforms(img_size=img_size, is_train=is_train)
        self.Y = self._build_targets(self.df[self.labels])

    def _map_value(self, v):
        """Map CheXpert targets {1, 0, -1, NaN} -> float in [0,1]."""
        if pd.isna(v):
            return 0.0
        if v == 1:
            return 1.0
        if v == 0:
            return 0.0
        if v == -1:
            if self.u_policy == "zeros":
                return 0.0
            if self.u_policy == "ones":
                return 1.0
            # 'ignore': map to 0.0; masking can be handled in the loss if needed
            return 0.0
        return float(v)

    def _build_targets(self, subdf: pd.DataFrame) -> torch.Tensor:
        mapped = (
            subdf.apply(lambda col: col.map(self._map_value)).astype("float32").values
        )
        return torch.from_numpy(mapped)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = str(row["Path"])
        img_path = self.root / rel_path
        if not img_path.exists():
            img_path = self.root / Path(rel_path).name  # small fallback

        img = Image.open(img_path).convert("L")
        x = self.tfm(img)
        y = self.Y[idx]
        pid = row.get("patient_id", extract_patient_id(rel_path))
        return {"image": x, "target": y, "path": str(img_path), "patient_id": pid}


def _read_csv(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        p = Path(os.getcwd()) / csv_path
    return pd.read_csv(p)


def build_loaders(
    cfg: CheXpertConfig,
) -> Tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    """
    Build train/dev/test datasets & dataloaders.
    - Train dataset uses cfg.augment to toggle train-time augmentations.
    - Dev/Test datasets are always non-augmented.
    """
    df_train = _read_csv(cfg.csv_train)
    df_dev = _read_csv(cfg.csv_dev)
    df_test = _read_csv(cfg.csv_test)

    ds_train = CheXpertDataset(
        df_train,
        root=cfg.root,
        img_size=cfg.img_size,
        is_train=cfg.augment,
        u_policy=cfg.u_policy,
    )
    ds_dev = CheXpertDataset(
        df_dev,
        root=cfg.root,
        img_size=cfg.img_size,
        is_train=False,
        u_policy=cfg.u_policy,
    )
    ds_test = CheXpertDataset(
        df_test,
        root=cfg.root,
        img_size=cfg.img_size,
        is_train=False,
        u_policy=cfg.u_policy,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    dl_dev = DataLoader(
        ds_dev,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return ds_train, ds_dev, ds_test, dl_train, dl_dev, dl_test

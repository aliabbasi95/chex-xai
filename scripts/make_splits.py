# scripts/make_splits.py

from __future__ import annotations
import os, re, argparse, random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import re

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
    if not m:
        parts = path_str.replace("\\","/").split("/")
        cand = [p for p in parts if p.lower().startswith("patient")]
        return cand[0] if cand else "patientUNK"
    return m.group(1)

def _to_rel_from_root(p: str) -> str:
    p = p.replace("\\", "/").lstrip("/")
    for pref in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
        if p.startswith(pref):
            p = p[len(pref):]
    m = re.search(r"(train|valid)/", p)
    if m:
        p = p[m.start():]
    return p

def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Path" not in df.columns:
        raise ValueError(f"'Path' column not found in {csv_path}")
    df["Path"] = df["Path"].astype(str).map(_to_rel_from_root)  # ← این خط مهمه
    df["patient_id"] = df["Path"].apply(extract_patient_id)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g., data/chexpert_small")
    ap.add_argument("--out_dir", type=str, default="data/splits")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--frontal_only", action="store_true", help="فقط تصاویر frontal")
    args = ap.parse_args()

    random.seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_root / "train.csv"
    valid_csv = data_root / "valid.csv"

    df_train_full = load_csv(train_csv)
    df_test = load_csv(valid_csv)

    if args.frontal_only:
        if "Frontal/Lateral" in df_train_full.columns:
            df_train_full = df_train_full[df_train_full["Frontal/Lateral"] == "Frontal"]
        if "Frontal/Lateral" in df_test.columns:
            df_test = df_test[df_test["Frontal/Lateral"] == "Frontal"]

    patients = df_train_full["patient_id"].unique()
    train_p, dev_p = train_test_split(
        patients, test_size=args.dev_ratio, random_state=args.seed, shuffle=True
    )
    df_train = df_train_full[df_train_full["patient_id"].isin(train_p)].reset_index(drop=True)
    df_dev   = df_train_full[df_train_full["patient_id"].isin(dev_p)].reset_index(drop=True)

    keep_cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + CHEXPERT_LABELS
    keep_cols = [c for c in keep_cols if c in df_train.columns]
    keep_cols = list(dict.fromkeys(keep_cols + ["patient_id"]))  # unique

    df_train[keep_cols].to_csv(out_dir / "train.csv", index=False)
    df_dev[keep_cols].to_csv(out_dir / "dev.csv", index=False)
    df_test[keep_cols].to_csv(out_dir / "test.csv", index=False)

    print(f"Saved splits to {out_dir} | train={len(df_train)} dev={len(df_dev)} test={len(df_test)}")

if __name__ == "__main__":
    main()

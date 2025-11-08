# scripts/dump_class_support.py

import json
from pathlib import Path

import pandas as pd

from chex_xai.data.chexpert import CHEXPERT_LABELS


def counts(csv_path):
    df = pd.read_csv(csv_path)
    labs = [c for c in CHEXPERT_LABELS if c in df.columns]
    Y = df[labs].values
    pos = (Y == 1).sum(axis=0).tolist()
    neg = (Y == 0).sum(axis=0).tolist()
    return {"labels": labs, "pos": pos, "neg": neg}


def main():
    out_dir = Path("outputs/chexpert_baseline_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dev": counts("data/splits/dev.csv"),
        "test": counts("data/splits/test.csv"),
    }
    with open(out_dir / "class_support.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_dir / 'class_support.json'}")


if __name__ == "__main__":
    main()

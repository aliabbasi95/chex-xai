import pandas as pd
import torch

from chex_xai.data.chexpert import CHEXPERT_LABELS

"""
Compute per-class pos_weight for BCEWithLogitsLoss:
    pos_weight[c] = N_neg[c] / max(N_pos[c], 1)
Using train split only. Saves a tensor to outputs/chexpert_baseline_v1/pos_weight.pt
"""


def main():
    train_csv = "data/splits/train.csv"
    df = pd.read_csv(train_csv)

    # Keep only label columns that exist
    labels = [c for c in CHEXPERT_LABELS if c in df.columns]
    Y = df[labels].copy()

    # CheXpert convention: {-1,0,1,NaN}. Treat positives as == 1
    is_pos = (Y.values == 1).astype("int64")
    is_neg = (Y.values == 0).astype("int64")

    pos = is_pos.sum(axis=0)
    neg = is_neg.sum(axis=0)

    pos_weight = []
    for i, name in enumerate(labels):
        pw = float(neg[i] / max(pos[i], 1))  # avoid div-by-zero
        pos_weight.append(pw)
        print(
            f"{i:02d} {name:24s}  pos={pos[i]:6d}  neg={neg[i]:6d}  pos_weight={pw:.2f}"
        )

    pw_tensor = torch.tensor(pos_weight, dtype=torch.float32)
    out_path = "outputs/chexpert_baseline_v1/pos_weight.pt"
    import os

    os.makedirs("outputs/chexpert_baseline_v1", exist_ok=True)
    torch.save(pw_tensor, out_path)
    print(f"\nSaved pos_weight tensor to: {out_path}")


if __name__ == "__main__":
    main()

# scripts/compute_thresholds.py

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from chex_xai.data.chexpert import CHEXPERT_LABELS, CheXpertConfig, build_loaders
from chex_xai.models.classifier import MultiLabelClassifier


def collect_probs_targets(model, loader, device):
    Ps, Ys = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["target"].float().to(device)
            p = torch.sigmoid(model(x))
            Ps.append(p.cpu())
            Ys.append(y.cpu())
    return torch.cat(Ps), torch.cat(Ys)


def best_thresh(p, y):
    ths = np.linspace(0.0, 1.0, 101)
    best, best_t = 0.0, 0.5
    y = y.numpy().astype("int32")
    p = p.numpy()
    for t in ths:
        pred = (p >= t).astype("int32")
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best:
            best, best_t = f1, t
    return float(best_t), float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", required=True, help="Path to trained checkpoint (best.pt)"
    )
    ap.add_argument("--exp", default="outputs/chexpert_baseline_v1")
    args = ap.parse_args()

    cfg_train = OmegaConf.load("configs/train.yaml")
    cfg_paths = OmegaConf.load("configs/paths.yaml")

    dcfg = CheXpertConfig(
        root=cfg_paths.data_root,
        csv_train="data/splits/train.csv",
        csv_dev="data/splits/dev.csv",
        csv_test="data/splits/test.csv",
        img_size=cfg_train.data.img_size,
        batch_size=cfg_train.data.batch_size,
        num_workers=cfg_train.data.num_workers,
    )
    _, _, _, _, dl_dev, _ = build_loaders(dcfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelClassifier(
        name=cfg_train.model.name,
        num_classes=len(CHEXPERT_LABELS),
        pretrained=False,
        dropout=cfg_train.model.dropout,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    P_dev, Y_dev = collect_probs_targets(model, dl_dev, device)
    ths = {}
    for i, name in enumerate(CHEXPERT_LABELS):
        t, f1 = best_thresh(P_dev[:, i], Y_dev[:, i])
        ths[name] = {"threshold": t, "dev_f1": f1}

    out_dir = Path(args.exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(ths, f, indent=2)
    print(f"saved -> {out_dir / 'thresholds.json'}")


if __name__ == "__main__":
    main()

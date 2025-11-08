# scripts/eval.py

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from chex_xai.data.chexpert import CHEXPERT_LABELS, CheXpertConfig, build_loaders
from chex_xai.engine.train import evaluate
from chex_xai.models.classifier import MultiLabelClassifier


def collect_probs_targets(model, loader, device):
    Ps, Ys, Paths = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["target"].float().to(device)
            p = torch.sigmoid(model(x))
            Ps.append(p.cpu())
            Ys.append(y.cpu())
            Paths += batch["path"]
    return torch.cat(Ps), torch.cat(Ys), Paths


def binarize_with_thresholds(probs, thresholds_by_idx):
    # probs: [N, C], thresholds_by_idx: length C floats
    thr = np.array(thresholds_by_idx, dtype=np.float32)[None, :]
    return (probs >= thr).astype("int32")


def f1_macro_micro(pred, y_true):
    C = y_true.shape[1]
    f1s = []
    tp = fp = fn = 0
    for c in range(C):
        yc = y_true[:, c]
        pc = pred[:, c]
        tp_c = ((pc == 1) & (yc == 1)).sum()
        fp_c = ((pc == 1) & (yc == 0)).sum()
        fn_c = ((pc == 0) & (yc == 1)).sum()
        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        tp += tp_c
        fp += fp_c
        fn += fn_c
    macro = float(np.mean(f1s) if len(f1s) else 0.0)
    prec_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    micro = float(
        2 * prec_micro * rec_micro / (prec_micro + rec_micro)
        if (prec_micro + rec_micro) > 0
        else 0.0
    )
    return macro, micro, f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument(
        "--thresholds", default="", help="JSON with per-class thresholds (optional)"
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
    _, _, _, dl_train, dl_dev, dl_test = build_loaders(dcfg)  # we use dev/test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLabelClassifier(
        name=cfg_train.model.name,
        num_classes=len(CHEXPERT_LABELS),
        pretrained=False,
        dropout=cfg_train.model.dropout,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    dev_metrics = evaluate(model, dl_dev, device, amp=args.amp)
    test_metrics = evaluate(model, dl_test, device, amp=args.amp)

    print("\n=== DEV METRICS (AUROC) ===")
    print(f"val_loss: {dev_metrics['val_loss']:.4f}")
    print(
        f"auroc_macro: {dev_metrics['auroc_macro']:.4f}  auroc_micro: {dev_metrics['auroc_micro']:.4f}"
    )

    print("\n=== TEST METRICS (AUROC) ===")
    print(f"val_loss: {test_metrics['val_loss']:.4f}")
    print(
        f"auroc_macro: {test_metrics['auroc_macro']:.4f}  auroc_micro: {test_metrics['auroc_micro']:.4f}"
    )

    ths_map = None
    if args.thresholds and Path(args.thresholds).exists():
        with open(args.thresholds) as f:
            raw = json.load(f)
        ths_map = [
            float(raw.get(name, {}).get("threshold", 0.5)) for name in CHEXPERT_LABELS
        ]

        P_dev, Y_dev, _ = [], [], []
        with torch.no_grad():
            P_dev, Y_dev, _ = collect_probs_targets(model, dl_dev, device)
            P_te, Y_te, _ = collect_probs_targets(model, dl_test, device)

        Pd = P_dev.numpy()
        Yd = Y_dev.numpy().astype("int32")
        Pt = P_te.numpy()
        Yt = Y_te.numpy().astype("int32")

        pred_d = binarize_with_thresholds(Pd, ths_map)
        pred_t = binarize_with_thresholds(Pt, ths_map)

        f1d_macro, f1d_micro, f1d_pc = f1_macro_micro(pred_d, Yd)
        f1t_macro, f1t_micro, f1t_pc = f1_macro_micro(pred_t, Yt)

        print("\n=== DEV (F1 with thresholds) ===")
        print(f"F1_macro: {f1d_macro:.4f}  F1_micro: {f1d_micro:.4f}")
        print("Per-class F1 (dev):")
        for i, name in enumerate(CHEXPERT_LABELS):
            print(f"{i:02d} {name:24s}  F1={f1d_pc[i]:.4f}  thr={ths_map[i]:.2f}")

        print("\n=== TEST (F1 with thresholds) ===")
        print(f"F1_macro: {f1t_macro:.4f}  F1_micro: {f1t_micro:.4f}")
        for i, name in enumerate(CHEXPERT_LABELS):
            print(f"{i:02d} {name:24s}  F1={f1t_pc[i]:.4f}  thr={ths_map[i]:.2f}")

    out_dir = Path(args.exp)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config_resolved.yaml", "w") as f:
        OmegaConf.save(config=cfg_train, f=f.name)
    with open(out_dir / "env.txt", "w") as f:
        f.write(f"python: {platform.python_version()}\n")
        f.write(f"torch:  {torch.__version__}\n")
        f.write(f"cuda_available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"cuda_device_count: {torch.cuda.device_count()}\n")
            f.write(f"device_name: {torch.cuda.get_device_name(0)}\n")

    dump = {
        "ckpt": args.ckpt,
        "thresholds": args.thresholds if ths_map is not None else None,
        "dev": {k: float(v) for k, v in dev_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
    }
    (out_dir / "metrics_eval.json").write_text(json.dumps(dump, indent=2))
    print(f"\nsaved -> {out_dir / 'metrics_eval.json'}")
    print(f"saved -> {out_dir / 'config_resolved.yaml'}")
    print(f"saved -> {out_dir / 'env.txt'}")


if __name__ == "__main__":
    main()

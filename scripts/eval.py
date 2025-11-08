# scripts/eval.py

import argparse
import warnings

import torch
from omegaconf import OmegaConf
from sklearn.exceptions import UndefinedMetricWarning

from chex_xai.data.chexpert import CHEXPERT_LABELS, CheXpertConfig, build_loaders
from chex_xai.engine.train import evaluate
from chex_xai.models.classifier import MultiLabelClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on CheXpert splits"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint .pt (best/last)"
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable AMP during evaluation"
    )
    args = parser.parse_args()

    # Optional: silence sklearn warning for single-class AUCs (we handle them safely anyway)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    cfg_train = OmegaConf.load("configs/train.yaml")
    cfg_paths = OmegaConf.load("configs/paths.yaml")

    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")

    # Build loaders (reuse same img_size, batch, workers)
    dcfg = CheXpertConfig(
        root=cfg_paths.data_root,
        csv_train="data/splits/train.csv",
        csv_dev="data/splits/dev.csv",
        csv_test="data/splits/test.csv",
        img_size=cfg_train.data.img_size,
        batch_size=cfg_train.data.batch_size,
        num_workers=cfg_train.data.num_workers,
        u_policy="zeros",
    )
    _, dl_dev, dl_test = build_loaders(dcfg)[3:]  # get only loaders

    # Model
    num_classes = len(CHEXPERT_LABELS)
    model = MultiLabelClassifier(
        name=cfg_train.model.name,
        num_classes=num_classes,
        pretrained=False,
        dropout=cfg_train.model.dropout,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    # Evaluate (DEV)
    print("\n=== DEV METRICS ===")
    dev = evaluate(model, dl_dev, device, amp=args.amp)
    print(f"val_loss: {dev['val_loss']:.4f}")
    print(
        f"auroc_macro: {dev['auroc_macro']:.4f}  auroc_micro: {dev['auroc_micro']:.4f}"
    )

    # Evaluate (TEST)
    print("\n=== TEST METRICS ===")
    te = evaluate(model, dl_test, device, amp=args.amp)
    print(f"val_loss: {te['val_loss']:.4f}")
    print(f"auroc_macro: {te['auroc_macro']:.4f}  auroc_micro: {te['auroc_micro']:.4f}")

    # Optional: print per-class AUCs where available
    print("\nPer-class AUROC (NaN means single-class in y_true):")
    for i, name in enumerate(CHEXPERT_LABELS):
        auc = te.get(f"auroc_c{i}")
        print(f"{i:02d} {name:24s}  {auc}")


if __name__ == "__main__":
    main()

# scripts/train.py

import json
import os
from datetime import datetime

import torch
from omegaconf import OmegaConf

from chex_xai.data.chexpert import CHEXPERT_LABELS, CheXpertConfig, build_loaders
from chex_xai.engine.train import evaluate, train_one_epoch
from chex_xai.models.classifier import MultiLabelClassifier
from chex_xai.utils.checkpoint import save_checkpoint
from chex_xai.utils.earlystop import EarlyStopping
from chex_xai.utils.ops import count_params
from chex_xai.utils.seed import set_seed


def build_optimizer(params, cfg):
    name = cfg.optim.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum,
            weight_decay=cfg.optim.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optim.name}")


def build_scheduler(optimizer, cfg):
    name = cfg.sched.name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.sched.t_max, eta_min=cfg.sched.min_lr
        )
    raise ValueError(f"Unknown scheduler: {cfg.sched.name}")


def _dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    # Load configs
    cfg_train = OmegaConf.load("configs/train.yaml")
    cfg_paths = OmegaConf.load("configs/paths.yaml")

    # Seed & device
    set_seed(cfg_train.seed)
    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")

    # Resolve output directory
    out_dir = os.path.join("outputs", cfg_train.experiment)
    os.makedirs(out_dir, exist_ok=True)

    # Persist resolved config for reproducibility
    resolved_yaml = OmegaConf.to_yaml(cfg_train, resolve=True)
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        f.write(resolved_yaml)

    # Data
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
    _, dl_dev, dl_test = build_loaders(dcfg)[0:3]  # datasets (unused) if needed later
    dl_train, dl_dev, dl_test = build_loaders(dcfg)[3:]  # loaders

    # Model
    num_classes = len(CHEXPERT_LABELS)
    model = MultiLabelClassifier(
        name=cfg_train.model.name,
        num_classes=num_classes,
        pretrained=cfg_train.model.pretrained,
        dropout=cfg_train.model.dropout,
    ).to(device)

    print(f"Model: {cfg_train.model.name} | params: {count_params(model):,}")

    # Optim/Sched
    optimizer = build_optimizer(model.parameters(), cfg_train)
    scheduler = build_scheduler(optimizer, cfg_train)

    # Early stopping (optional via config; defaults to disabled)
    patience = OmegaConf.select(cfg_train, "train.early_stopping.patience") or 0
    min_delta = OmegaConf.select(cfg_train, "train.early_stopping.min_delta") or 0.0
    es = EarlyStopping(patience=int(patience), mode="max", min_delta=float(min_delta))

    best_metric = -1.0
    history = []  # keep per-epoch metrics for metrics.json

    # Train loop
    for epoch in range(1, cfg_train.train.epochs + 1):
        print(f"\n=== Epoch {epoch}/{cfg_train.train.epochs} ===")
        tr = train_one_epoch(
            model,
            dl_train,
            optimizer,
            device,
            amp=cfg_train.train.amp,
            grad_clip=cfg_train.train.grad_clip,
            print_every=cfg_train.log.print_every,
        )
        print(f"train: loss={tr['loss']:.4f}")

        if scheduler is not None:
            scheduler.step()

        do_val = (epoch % cfg_train.train.val_interval) == 0
        if do_val:
            val = evaluate(model, dl_dev, device, amp=cfg_train.train.amp)
            print(
                f"valid: loss={val['val_loss']:.4f}  "
                f"auroc_macro={val['auroc_macro']:.4f}  auroc_micro={val['auroc_micro']:.4f}"
            )

            # Save last + best
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": OmegaConf.to_container(cfg_train, resolve=True),
                },
                out_dir=out_dir,
                is_best=(
                    cfg_train.train.save_best and val["auroc_macro"] > best_metric
                ),
                filename="last.pt",
            )

            # Track best
            if val["auroc_macro"] > best_metric:
                best_metric = val["auroc_macro"]

            # Append to history and persist metrics.json
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": tr["loss"],
                    "val_loss": val["val_loss"],
                    "val_auroc_macro": val["auroc_macro"],
                    "val_auroc_micro": val["auroc_micro"],
                    "time": datetime.utcnow().isoformat() + "Z",
                }
            )
            _dump_json(
                os.path.join(out_dir, "metrics.json"),
                {"best_auroc_macro": best_metric, "history": history},
            )

            # Early stopping check
            if es.patience > 0 and es.step(val["auroc_macro"]):
                print(
                    f"Early stopping: no improvement in {es.patience} val checks. "
                    f"Best auroc_macro={es.best:.4f}"
                )
                break

    # Final TEST evaluation with the current (last) model
    print("\nEvaluating on TEST set with the final model weights...")
    test = evaluate(model, dl_test, device, amp=cfg_train.train.amp)
    print(
        f"test: loss={test['val_loss']:.4f}  "
        f"auroc_macro={test['auroc_macro']:.4f}  auroc_micro={test['auroc_micro']:.4f}"
    )

    # Persist final test metrics (append into metrics.json)
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            obj = json.load(f)
    else:
        obj = {"best_auroc_macro": best_metric, "history": history}
    obj["test"] = {
        "loss": test["val_loss"],
        "auroc_macro": test["auroc_macro"],
        "auroc_micro": test["auroc_micro"],
        "time": datetime.utcnow().isoformat() + "Z",
    }
    _dump_json(metrics_path, obj)


if __name__ == "__main__":
    main()

# scripts/train.py

import os

import torch
from omegaconf import OmegaConf

from src.chex_xai.data.chexpert import (
    CHEXPERT_LABELS,
    CheXpertConfig,
    build_loaders,
)
from src.chex_xai.engine.train import evaluate, train_one_epoch
from src.chex_xai.models.classifier import MultiLabelClassifier
from src.chex_xai.utils.checkpoint import save_checkpoint
from src.chex_xai.utils.ops import count_params
from src.chex_xai.utils.seed import set_seed


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


def main():
    # Load configs
    cfg_train = OmegaConf.load("configs/train.yaml")
    cfg_paths = OmegaConf.load("configs/paths.yaml")

    set_seed(cfg_train.seed)
    device = torch.device(cfg_train.device if torch.cuda.is_available() else "cpu")

    # Data
    dcfg = CheXpertConfig(
        root=cfg_paths.data_root,
        csv_train="data/splits/train.csv",
        csv_dev="data/splits/dev.csv",
        csv_test="data/splits/test.csv",
        img_size=cfg_train.data.img_size,
        augment=cfg_train.data.augment,
        batch_size=cfg_train.data.batch_size,
        num_workers=cfg_train.data.num_workers,
    )
    dl_train, dl_dev, dl_test = build_loaders(dcfg)

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

    best_metric = -1.0
    out_dir = os.path.join("outputs", cfg_train.experiment)
    os.makedirs(out_dir, exist_ok=True)

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

        if (epoch % cfg_train.train.val_interval) == 0:
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
            if val["auroc_macro"] > best_metric:
                best_metric = val["auroc_macro"]

    print("\nEvaluating on TEST set with the final model weights...")
    test = evaluate(model, dl_test, device, amp=cfg_train.train.amp)
    print(
        f"test: loss={test['val_loss']:.4f}  "
        f"auroc_macro={test['auroc_macro']:.4f}  auroc_micro={test['auroc_micro']:.4f}"
    )


if __name__ == "__main__":
    main()

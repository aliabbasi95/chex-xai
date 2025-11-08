# scripts/check_dataloader.py

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from chex_xai.data.chexpert import CHEXPERT_LABELS, CheXpertConfig, build_loaders


def main():
    cfg_paths = OmegaConf.load("configs/paths.yaml")
    cfg_train = OmegaConf.load("configs/train.yaml")

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

    ds_train, ds_dev, ds_test, dl_train, dl_dev, dl_test = build_loaders(dcfg)
    print(f"train/dev/test sizes: {len(ds_train)}/{len(ds_dev)}/{len(ds_test)}")
    xb = next(iter(dl_train))
    x, y = xb["image"], xb["target"]
    print("Batch shapes:", x.shape, y.shape)  # [B,3,H,W], [B,C]
    print("Labels:", CHEXPERT_LABELS)
    print("Sample targets row 0:", y[0])


if __name__ == "__main__":
    torch.manual_seed(1337)
    main()

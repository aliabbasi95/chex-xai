# src/chex_xai/utils/checkpoint.py

import os

import torch


def save_checkpoint(
    state: dict, out_dir: str, is_best: bool = False, filename: str = "last.pt"
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, "best.pt")
        torch.save(state, best_path)


def load_checkpoint(path: str, map_location=None) -> dict:
    return torch.load(path, map_location=map_location)

# src/chex_xai/utils/ops.py

from typing import Any

import torch


def to_device(x: Any, device: torch.device):
    """Recursively move tensors to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(xx, device) for xx in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


class AverageMeter:
    """Keeps track of the running average of a scalar."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def count_params(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_pos_weight(targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.
    Formula: pos_weight_c = (neg_c + eps) / (pos_c + eps)
    targets: shape [N, C], values in {0,1}.
    """
    if targets.ndim != 2:
        raise ValueError(f"targets must be 2D [N, C], got shape={tuple(targets.shape)}")
    # ensure float
    t = targets.float()
    pos = t.sum(dim=0)  # [C]
    neg = t.shape[0] - pos  # [C]
    pw = (neg + eps) / (pos + eps)  # [C]
    return pw

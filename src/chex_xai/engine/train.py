# src/chex_xai/engine/train.py

from typing import Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from ..metrics.metrics import compute_auroc
from ..utils.ops import AverageMeter, to_device


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = True,
    grad_clip: float = 0.0,
    print_every: int = 50,
) -> Dict[str, float]:
    """
    Single training epoch. Expects dataloader batches shaped as dict:
      {"image": Tensor[B, C, H, W], "target": Tensor[B, K], ...}
    Returns dict with the average loss across the epoch.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=amp)

    loss_meter = AverageMeter()

    for step, batch in enumerate(loader):
        # Unpack dict batch
        xb, yb = batch["image"], batch["target"]
        xb, yb = to_device((xb, yb), device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=xb.size(0))

        if (step + 1) % print_every == 0:
            print(f"[train] step={step + 1:05d} loss={loss_meter.avg:.4f}")

    return {"loss": loss_meter.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    amp: bool = True,
) -> Dict[str, float]:
    """
    Validation loop. Consumes dict batches and reports loss + AUROC metrics.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    loss_meter = AverageMeter()
    all_logits = []
    all_targets = []

    for batch in loader:
        xb, yb = batch["image"], batch["target"]
        xb, yb = to_device((xb, yb), device)

        with autocast(enabled=amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        loss_meter.update(loss.item(), n=xb.size(0))
        all_logits.append(torch.sigmoid(logits).detach().cpu())
        all_targets.append(yb.detach().cpu())

    probs = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    auroc_macro, auroc_micro, per_class = compute_auroc(probs, targets)

    metrics = {
        "val_loss": loss_meter.avg,
        "auroc_macro": auroc_macro,
        "auroc_micro": auroc_micro,
    }
    # Expose per-class AUROC with stable keys
    for i, v in enumerate(per_class):
        metrics[f"auroc_c{i}"] = v

    return metrics

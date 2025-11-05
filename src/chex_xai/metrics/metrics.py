# src/chex_xai/metrics/metrics.py

from typing import List, Tuple

import torch
from sklearn.metrics import roc_auc_score


def compute_auroc(
    probs: torch.Tensor, targets: torch.Tensor  # [N, C], sigmoid-ed  # [N, C], {0,1}
) -> Tuple[float, float, List[float]]:
    """
    Compute per-class AUROC, and their macro and micro averages.
    """
    p = probs.numpy()
    y = targets.numpy()

    per_class = []
    # Handle the case where a class has only one label present → roc_auc_score may fail.
    for c in range(p.shape[1]):
        try:
            auc = roc_auc_score(y[:, c], p[:, c])
        except Exception:
            auc = float("nan")
        per_class.append(auc)

    # Macro = mean of per-class (ignoring NaNs)
    valid = [v for v in per_class if v == v]
    auroc_macro = float(sum(valid) / len(valid)) if valid else float("nan")

    # Micro: flatten across classes
    try:
        auroc_micro = roc_auc_score(y.reshape(-1), p.reshape(-1))
    except Exception:
        auroc_micro = float("nan")

    return auroc_macro, auroc_micro, per_class

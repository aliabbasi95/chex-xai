# src/chex_xai/metrics/metrics.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _to_numpy(x):
    """Convert Torch tensor or numpy-like to plain numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_auroc(
    probs: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float, List[float]]:
    """
    Compute per-class ROC AUC safely:
      - If a class has only a single label in y_true (all 0 or all 1), its AUC is NaN.
      - Macro AUC = nanmean(per-class AUC).
      - Micro AUC is computed by concatenating only valid classes (those with both labels).

    Parameters
    ----------
    probs : (N, C) tensor
        Sigmoid probabilities in [0, 1].
    targets : (N, C) tensor
        Binary ground-truth labels {0,1}.

    Returns
    -------
    macro_auc : float
    micro_auc : float
    per_class : list of float (len = C, NaN for invalid classes)
    """
    P = _to_numpy(probs)
    Y = _to_numpy(targets)
    N, C = Y.shape

    per_class = []
    valid_cols = []
    for c in range(C):
        y = Y[:, c]
        p = P[:, c]
        # Class is valid only if we have at least one 0 and one 1
        if np.unique(y).size == 2:
            try:
                auc = roc_auc_score(y, p)
            except Exception:
                auc = np.nan
        else:
            auc = np.nan
        per_class.append(auc)
        if not np.isnan(auc):
            valid_cols.append(c)

    # Macro over valid classes
    macro_auc = (
        float(np.nanmean(per_class)) if np.any(~np.isnan(per_class)) else float("nan")
    )

    # Micro over valid classes
    if len(valid_cols) > 0:
        y_micro = Y[:, valid_cols].ravel()
        p_micro = P[:, valid_cols].ravel()
        if np.unique(y_micro).size == 2:
            micro_auc = float(roc_auc_score(y_micro, p_micro))
        else:
            micro_auc = float("nan")
    else:
        micro_auc = float("nan")

    return macro_auc, micro_auc, per_class

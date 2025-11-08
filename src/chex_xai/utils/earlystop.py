# src/chex_xai/utils/earlystop.py

"""
Simple EarlyStopping utility.

- mode = "max": improvement if value > best + min_delta
- mode = "min": improvement if value < best - min_delta
Returns True from step(value) when patience exceeded (i.e., should stop).
"""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 0, mode: str = "max", min_delta: float = 0.0):
        self.patience = max(int(patience), 0)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_count = 0

    def step(self, value: float) -> bool:
        """Update with a new metric value. Return True if we should stop."""
        if self.best is None:
            self.best = value
            self.bad_count = 0
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.bad_count = 0
        else:
            self.bad_count += 1

        return self.bad_count > self.patience

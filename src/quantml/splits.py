"""Chronological train/validation/test splits and walk-forward folds.

Observations are never shuffled. Positions refer to rows of a chronologically ordered dataset.
Because a label at row ``i`` depends on prices up to ``i + lookahead``, the last ``purge`` rows
before each boundary are removed so that no training label overlaps a later evaluation period.
An optional ``embargo`` removes further rows at each boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HoldoutSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    purged: np.ndarray


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train: np.ndarray
    test: np.ndarray


def chronological_split(n: int, train: float = 0.6, validation: float = 0.2, test: float = 0.2,
                        purge: int = 1, embargo: int = 0, min_train: int = 50,
                        min_test: int = 20) -> HoldoutSplit:
    """Split ``n`` ordered rows into train / validation / test blocks with boundary purging."""
    if abs(train + validation + test - 1.0) > 1e-6:
        raise ValueError("Split proportions must sum to 1.")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative.")
    gap = purge + embargo
    n_train = int(np.floor(n * train))
    n_val = int(np.floor(n * validation))
    val_start, test_start = n_train, n_train + n_val

    train_idx = np.arange(0, max(n_train - gap, 0))
    val_idx = np.arange(val_start, max(test_start - gap, val_start))
    test_idx = np.arange(test_start, n)
    purged = np.setdiff1d(np.arange(n), np.concatenate([train_idx, val_idx, test_idx]))

    if len(train_idx) < min_train:
        raise ValueError(f"Training set has {len(train_idx)} rows (minimum {min_train}). Use a "
                         "longer date range or a larger training proportion.")
    if len(test_idx) < min_test:
        raise ValueError(f"Test set has {len(test_idx)} rows (minimum {min_test}). Use a longer "
                         "date range or a larger test proportion.")
    if validation > 0 and len(val_idx) == 0:
        raise ValueError("Validation set is empty after purging; increase its proportion.")
    return HoldoutSplit(train_idx, val_idx, test_idx, purged)


def walk_forward_splits(n: int, initial_train_size: int, test_size: int, purge: int = 1,
                        embargo: int = 0, window: str = "expanding",
                        max_train_size: int | None = None) -> list[WalkForwardFold]:
    """Generate walk-forward folds: train on the past only, predict the next block, roll forward."""
    if initial_train_size >= n:
        raise ValueError(f"initial_train_size ({initial_train_size}) must be smaller than the "
                         f"number of eligible observations ({n}).")
    if test_size < 1:
        raise ValueError("test_size must be >= 1")
    if window not in ("expanding", "rolling"):
        raise ValueError("window must be 'expanding' or 'rolling'")
    gap = purge + embargo
    rolling_size = max_train_size or initial_train_size
    folds: list[WalkForwardFold] = []
    test_start = initial_train_size
    while test_start < n:
        test_end = min(test_start + test_size, n)
        train_end = test_start - gap
        train_start = 0 if window == "expanding" else max(0, train_end - rolling_size)
        if train_end - train_start < 2:
            raise ValueError("Walk-forward training window is empty after purging.")
        folds.append(WalkForwardFold(len(folds), np.arange(train_start, train_end),
                                     np.arange(test_start, test_end)))
        test_start = test_end
    return folds


def assert_no_overlap(dates: pd.DatetimeIndex, label_end_dates: pd.Series,
                      train_idx: np.ndarray, later_idx: np.ndarray) -> None:
    """Raise if any training label depends on prices dated at or after the later block's start."""
    if len(train_idx) == 0 or len(later_idx) == 0:
        return
    later_start = dates[later_idx.min()]
    if train_idx.max() >= later_idx.min():
        raise AssertionError("Training rows must precede evaluation rows.")
    last_label_end = pd.Series(label_end_dates).iloc[train_idx].max()
    if pd.notna(last_label_end) and last_label_end >= later_start:
        raise AssertionError(f"Leakage: a training label uses prices up to {last_label_end:%Y-%m-%d}, "
                             f"on/after evaluation start {later_start:%Y-%m-%d}.")

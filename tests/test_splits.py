"""Chronological splitting, purging, walk-forward folds and train-only preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantml.models import fit_model
from quantml.splits import assert_no_overlap, chronological_split, walk_forward_splits


def test_chronological_split_order_and_sizes():
    s = chronological_split(1000, 0.6, 0.2, 0.2, purge=0)
    assert (len(s.train), len(s.validation), len(s.test)) == (600, 200, 200)
    assert s.train.max() < s.validation.min() < s.test.min()
    assert np.all(np.diff(np.concatenate([s.train, s.validation, s.test])) == 1), "never shuffled"


def test_purge_and_embargo_remove_boundary_rows():
    s = chronological_split(1000, 0.6, 0.2, 0.2, purge=2, embargo=3)
    assert s.train.max() == 600 - 5 - 1
    assert s.validation.min() == 600 and s.validation.max() == 800 - 5 - 1
    assert s.test.min() == 800 and s.test.max() == 999
    assert len(s.purged) == 10


def test_split_rejects_bad_inputs():
    with pytest.raises(ValueError, match="sum to 1"):
        chronological_split(1000, 0.6, 0.3, 0.3)
    with pytest.raises(ValueError, match="Training set"):
        chronological_split(60, 0.5, 0.25, 0.25)


def test_assert_no_overlap_detects_label_leakage():
    dates = pd.bdate_range("2024-01-01", periods=10)
    label_end = pd.Series(dates).shift(-1)           # 1-step-ahead labels
    train, test = np.arange(0, 5), np.arange(5, 10)
    with pytest.raises(AssertionError, match="Leakage"):
        assert_no_overlap(dates, label_end, train, test)   # row 4's label uses row 5's price
    assert_no_overlap(dates, label_end, np.arange(0, 4), test)


def test_walk_forward_trains_only_on_the_past():
    folds = walk_forward_splits(500, initial_train_size=200, test_size=50, purge=1)
    assert len(folds) == 6
    tested = np.concatenate([f.test for f in folds])
    np.testing.assert_array_equal(tested, np.arange(200, 500))    # contiguous, no overlap
    for f in folds:
        assert f.train.min() == 0                                 # expanding window
        assert f.train.max() + 1 < f.test.min()                   # purged boundary
    assert len(folds[-1].train) > len(folds[0].train)


def test_walk_forward_rolling_window_and_partial_last_fold():
    folds = walk_forward_splits(530, 200, 100, purge=1, window="rolling", max_train_size=150)
    assert all(len(f.train) == 150 for f in folds)
    assert len(folds[-1].test) == 30
    with pytest.raises(ValueError):
        walk_forward_splits(100, 100, 10)


def test_scaler_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": np.r_[rng.normal(0, 1, 300), rng.normal(50, 1, 100)],
                      "b": rng.normal(0, 1, 400)})
    y = pd.Series(rng.integers(0, 2, 400))
    model = fit_model("logistic_regression", {"C": 1.0}, X.iloc[:300], y.iloc[:300])
    scaler = model.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, X.iloc[:300].mean().to_numpy())
    assert abs(scaler.mean_[0] - X["a"].mean()) > 5, "statistics must exclude later rows"


def test_single_class_training_data_is_rejected():
    X = pd.DataFrame({"a": np.arange(10.0)})
    with pytest.raises(ValueError, match="only one class"):
        fit_model("logistic_regression", {}, X, pd.Series(np.ones(10)))

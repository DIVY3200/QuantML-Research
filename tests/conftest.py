"""Shared deterministic fixtures. No test touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantml.config import ResearchConfig, load_config
from quantml.data_loader import generate_synthetic_ohlcv, prepare_ohlcv, standardize_columns


def make_prices(close: list[float], open_: list[float] | None = None,
                start: str = "2024-01-01") -> pd.DataFrame:
    """Tiny analysis-ready price frame on business days (high/low bracket open and close)."""
    close_arr = np.asarray(close, dtype=float)
    open_arr = np.asarray(open_ if open_ is not None else close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close_arr), name="date")
    return pd.DataFrame({
        "open": open_arr,
        "high": np.maximum(open_arr, close_arr) * 1.01,
        "low": np.minimum(open_arr, close_arr) * 0.99,
        "close": close_arr,
        "volume": np.full(len(close_arr), 1_000.0),
    }, index=idx)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    raw = standardize_columns(generate_synthetic_ohlcv(n_days=1200, seed=7))
    prices, _ = prepare_ohlcv(raw, is_synthetic=True)
    return prices


@pytest.fixture
def small_config() -> ResearchConfig:
    """Fast configuration for integration tests on synthetic data."""
    return load_config(overrides={
        "data": {"source": "synthetic"},
        "validation": {"min_observations": 300},
        "walk_forward": {"initial_train_size": 400, "test_size": 100},
        "evaluation": {"bootstrap": {"n_resamples": 100}},
    })


@pytest.fixture
def valid_raw() -> pd.DataFrame:
    """Raw-format (standardised but unprepared) OHLCV with adjusted close."""
    df = generate_synthetic_ohlcv(n_days=120, seed=3)
    df["adj_close"] = df["close"] * 0.9
    return standardize_columns(df)

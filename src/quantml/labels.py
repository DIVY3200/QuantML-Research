"""Prediction targets aligned to the decision time.

A feature row dated ``t`` is known at the close of ``t``. Its label describes what happens
*after* that close:

* ``basis="close"`` (default): forward return ``close[t+h] / close[t] - 1``.
* ``basis="execution"``: the return a position decided at ``t`` can actually earn under the
  backtest's execution convention - ``open[t+1+h] / open[t+1] - 1`` for ``next_open`` or
  ``close[t+1+h] / close[t+1] - 1`` for ``next_close``.

The final rows whose forward window runs past the end of the data have NaN labels; they are
never used for training or classification metrics. The label is never used as a feature.
"""

from __future__ import annotations

import pandas as pd

from quantml.config import TargetConfig


def label_lookahead(cfg: TargetConfig) -> int:
    """How many future rows a label consumes; used to purge observations at split boundaries."""
    return cfg.horizon + (1 if cfg.basis == "execution" else 0)


def forward_returns(prices: pd.DataFrame, horizon: int = 1, basis: str = "close",
                    execution: str = "next_open") -> pd.Series:
    """Forward return for each row, using only prices strictly after the row's close."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if basis == "close":
        close = prices["close"]
        return close.shift(-horizon) / close - 1
    if basis == "execution":
        px = prices["open"] if execution == "next_open" else prices["close"]
        return px.shift(-(horizon + 1)) / px.shift(-1) - 1
    raise ValueError(f"Unknown target basis '{basis}'")


def make_labels(prices: pd.DataFrame, cfg: TargetConfig, execution: str = "next_open") -> pd.DataFrame:
    """Return ``fwd_return``, binary ``label`` (1.0 / 0.0 / NaN) and ``label_end_date``.

    ``label_end_date`` is the date of the last price the label depends on; it lets callers verify
    that no training label overlaps a later evaluation period.
    """
    fwd = forward_returns(prices, cfg.horizon, cfg.basis, execution)
    label = (fwd > cfg.positive_threshold).astype(float).where(fwd.notna())
    end_dates = pd.Series(prices.index, index=prices.index).shift(-label_lookahead(cfg))
    return pd.DataFrame({"fwd_return": fwd, "label": label, "label_end_date": end_dates},
                        index=prices.index)

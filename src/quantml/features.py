"""Feature engineering with explicit no-look-ahead guarantees.

Every feature in a row dated ``t`` is computed only from prices and volume observed at or before
the close of ``t`` (backward-looking rolling windows and non-negative shifts only). Rows inside a
lookback warm-up period are left as NaN and dropped later - values are never invented.

:func:`audit_lookahead` verifies this empirically: features recomputed on data truncated at ``t``
must equal the features computed on the full history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantml.config import FeatureConfig


def volume_usable(prices: pd.DataFrame, cfg: FeatureConfig) -> bool:
    """Whether volume features should be built for this dataset."""
    setting = str(cfg.use_volume).lower()
    if setting == "false" or "volume" not in prices.columns:
        return False
    vol = prices["volume"]
    has_data = bool(vol.notna().mean() > 0.95 and (vol > 0).mean() > 0.95)
    if setting == "true" and not has_data:
        raise ValueError("features.use_volume is true but volume data is missing or mostly zero.")
    return has_data


def feature_descriptions(cfg: FeatureConfig, use_volume: bool) -> dict[str, str]:
    """Human-readable definitions, keyed by feature column name (all known at close t)."""
    desc: dict[str, str] = {}
    for lag in range(cfg.return_lags):
        desc[f"ret_lag_{lag}"] = (f"Close-to-close return from t-{lag + 1} to t-{lag}"
                                  if lag else "Close-to-close return from t-1 to t")
    for w in cfg.momentum_windows:
        desc[f"mom_{w}"] = f"{w}-day momentum: close[t] / close[t-{w}] - 1"
    for w in cfg.volatility_windows:
        desc[f"vol_{w}"] = f"Standard deviation of the last {w} daily returns (not annualised)"
    for w in cfg.ma_distance_windows:
        desc[f"dist_ma_{w}"] = f"Distance from the {w}-day simple moving average: close / SMA{w} - 1"
    for s, lng in cfg.ma_ratio_pairs:
        desc[f"ma_ratio_{s}_{lng}"] = f"Moving-average ratio: SMA{s} / SMA{lng} - 1"
    desc["hl_range"] = "Daily range: (high - low) / close"
    w = cfg.range_window
    desc[f"range_pos_{w}"] = (f"Position of close within the {w}-day high-low range "
                              "(0 = at the low, 1 = at the high)")
    if use_volume:
        v = cfg.volume_window
        desc[f"volume_ratio_{v}"] = f"Volume relative to its {v}-day average: volume / SMA{v} - 1"
        desc["volume_change_1d"] = "One-day log change in volume: log(1 + V[t]) - log(1 + V[t-1])"
    return desc


def build_features(prices: pd.DataFrame, cfg: FeatureConfig,
                   use_volume: bool | None = None) -> pd.DataFrame:
    """Build the feature matrix (same index as ``prices``; warm-up rows contain NaN)."""
    if use_volume is None:
        use_volume = volume_usable(prices, cfg)
    close, high, low = prices["close"], prices["high"], prices["low"]
    ret = close.pct_change(fill_method=None)
    feats: dict[str, pd.Series] = {}

    for lag in range(cfg.return_lags):
        feats[f"ret_lag_{lag}"] = ret.shift(lag)
    for w in cfg.momentum_windows:
        feats[f"mom_{w}"] = close / close.shift(w) - 1
    for w in cfg.volatility_windows:
        feats[f"vol_{w}"] = ret.rolling(w, min_periods=w).std()
    for w in cfg.ma_distance_windows:
        feats[f"dist_ma_{w}"] = close / close.rolling(w, min_periods=w).mean() - 1
    for s, lng in cfg.ma_ratio_pairs:
        feats[f"ma_ratio_{s}_{lng}"] = (close.rolling(s, min_periods=s).mean()
                                        / close.rolling(lng, min_periods=lng).mean() - 1)
    feats["hl_range"] = (high - low) / close
    w = cfg.range_window
    hh = high.rolling(w, min_periods=w).max()
    ll = low.rolling(w, min_periods=w).min()
    feats[f"range_pos_{w}"] = (close - ll) / (hh - ll)

    if use_volume:
        vol = prices["volume"].astype(float)
        v = cfg.volume_window
        feats[f"volume_ratio_{v}"] = vol / vol.rolling(v, min_periods=v).mean() - 1
        feats["volume_change_1d"] = np.log1p(vol).diff()

    features = pd.DataFrame(feats, index=prices.index)
    return features.replace([np.inf, -np.inf], np.nan)


def max_lookback(cfg: FeatureConfig) -> int:
    """Number of leading rows that cannot have complete features."""
    return max([cfg.return_lags, *(w for w in cfg.momentum_windows),
                *(w for w in cfg.volatility_windows), *cfg.ma_distance_windows,
                *(p[1] for p in cfg.ma_ratio_pairs), cfg.range_window, cfg.volume_window])


@dataclass
class LookaheadAudit:
    passed: bool
    n_checked: int
    max_abs_difference: float
    offending_dates: list[str] = field(default_factory=list)


def audit_lookahead(prices: pd.DataFrame, cfg: FeatureConfig, n_checks: int = 25,
                    tolerance: float = 1e-9) -> LookaheadAudit:
    """Recompute features on data truncated at sampled dates and compare with the full run.

    If any feature at ``t`` changes when observations after ``t`` are removed, that feature uses
    future information and the audit fails.
    """
    use_volume = volume_usable(prices, cfg)
    full = build_features(prices, cfg, use_volume=use_volume)
    complete = full.dropna().index
    if len(complete) == 0:
        return LookaheadAudit(False, 0, float("nan"), ["no complete feature rows"])
    positions = np.unique(np.linspace(0, len(complete) - 1, min(n_checks, len(complete))).astype(int))
    max_diff, offending = 0.0, []
    for pos in positions:
        t = complete[pos]
        truncated = build_features(prices.loc[:t], cfg, use_volume=use_volume).iloc[-1]
        diff = float(np.nanmax(np.abs(truncated.to_numpy() - full.loc[t].to_numpy())))
        mismatched_nan = bool((truncated.isna() != full.loc[t].isna()).any())
        max_diff = max(max_diff, diff)
        if diff > tolerance or mismatched_nan:
            offending.append(t.strftime("%Y-%m-%d"))
    return LookaheadAudit(not offending, len(positions), max_diff, offending)

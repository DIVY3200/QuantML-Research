"""Model selection on validation data and robustness diagnostics.

Nothing in this module looks at the final test period except :func:`cost_sensitivity` and
:func:`block_bootstrap`, which only *describe* already-fixed strategies on that period.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from quantml.backtester import CostModel, run_backtest
from quantml.metrics import periodic_risk_free, sharpe_ratio
from quantml.models import fit_model, predict_up_probability
from quantml.strategies import ProbabilityThresholdStrategy


def tune_hyperparameters(name: str, candidates: list[dict[str, Any]], X_train: pd.DataFrame,
                         y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                         seed: int = 42) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit each candidate on training data and rank by validation log loss (lower is better)."""
    rows = []
    for params in candidates:
        model = fit_model(name, params, X_train, y_train, seed)
        p = predict_up_probability(model, X_val).clip(1e-6, 1 - 1e-6)
        y = y_val.astype(int)
        rows.append({"params": params, "validation_log_loss": log_loss(y, p, labels=[0, 1]),
                     "validation_accuracy": float(((p >= 0.5).astype(int) == y).mean())})
    table = pd.DataFrame(rows)
    best = table.loc[table["validation_log_loss"].idxmin(), "params"]
    table["params"] = table["params"].astype(str)
    return best, table


def select_threshold(probabilities: pd.Series, prices: pd.DataFrame, start: object, end: object,
                     grid: list[float], costs: CostModel, execution: str,
                     below_threshold_position: float = 0.0, periods_per_year: int = 252,
                     risk_free_rate: float = 0.0, default: float = 0.5) -> tuple[float, pd.DataFrame]:
    """Choose the threshold that maximises *validation* net Sharpe. Never call with test dates.

    Thresholds that leave the strategy permanently in cash have undefined Sharpe and are skipped;
    ties are broken in favour of the threshold closest to 0.5.
    """
    rows = []
    for thr in grid:
        strat = ProbabilityThresholdStrategy(probabilities, thr, below_threshold_position)
        bt = run_backtest(prices, strat.generate_signals(prices), costs, execution, start, end)
        rows.append({"threshold": thr,
                     "validation_sharpe_net": sharpe_ratio(bt.net_returns, periods_per_year,
                                                           risk_free_rate),
                     "validation_return_net": float((1 + bt.net_returns).prod() - 1),
                     "average_exposure": float(bt.positions.mean())})
    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["validation_sharpe_net"])
    if valid.empty:
        return default, table
    best_sharpe = valid["validation_sharpe_net"].max()
    best = valid[np.isclose(valid["validation_sharpe_net"], best_sharpe)]
    chosen = float(best.loc[(best["threshold"] - 0.5).abs().idxmin(), "threshold"])
    return chosen, table


def cost_sensitivity(prices: pd.DataFrame, signals: dict[str, pd.Series], start: object, end: object,
                     cost_grid_bps: list[float], execution: str, periods_per_year: int = 252,
                     risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Net performance of fixed signals as the total cost per unit turnover varies."""
    rows = []
    for total_bps in cost_grid_bps:
        costs = CostModel(fee_bps=float(total_bps), slippage_bps=0.0)
        for name, sig in signals.items():
            bt = run_backtest(prices, sig, costs, execution, start, end)
            rows.append({"cost_bps": float(total_bps), "strategy": name,
                         "cumulative_return_net": float((1 + bt.net_returns).prod() - 1),
                         "sharpe_net": sharpe_ratio(bt.net_returns, periods_per_year,
                                                    risk_free_rate)})
    return pd.DataFrame(rows)


def _block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block_size)[None, :]).ravel()[:n]


def _sharpe_matrix(samples: np.ndarray, periods_per_year: int, rf_periodic: float) -> np.ndarray:
    excess = samples - rf_periodic
    sd = excess.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = excess.mean(axis=0) / sd * np.sqrt(periods_per_year)
    return np.where(sd > 1e-12, out, np.nan)


def block_bootstrap(returns: pd.DataFrame, benchmark: str, n_resamples: int = 1000,
                    block_size: int = 20, confidence: float = 0.95, periods_per_year: int = 252,
                    risk_free_rate: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Moving-block bootstrap of annualised Sharpe ratios and Sharpe differences vs ``benchmark``.

    Days are resampled in contiguous blocks (same blocks for every strategy, preserving
    cross-sectional pairing and short-range autocorrelation). Assumes approximate stationarity
    within the evaluation window; intervals are approximate, not exact significance tests, and
    do not correct for the number of strategies or configurations tried.
    """
    data = returns.dropna()
    n = len(data)
    if n < 2 * block_size or benchmark not in data.columns:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rf = periodic_risk_free(risk_free_rate, periods_per_year)
    values = data.to_numpy()
    boot = np.empty((n_resamples, values.shape[1]))
    for b in range(n_resamples):
        boot[b] = _sharpe_matrix(values[_block_indices(n, block_size, rng)], periods_per_year, rf)

    lo_q, hi_q = (1 - confidence) / 2, 1 - (1 - confidence) / 2
    point = _sharpe_matrix(values, periods_per_year, rf)
    bench_col = list(data.columns).index(benchmark)
    rows = []
    for j, name in enumerate(data.columns):
        diff = boot[:, j] - boot[:, bench_col]
        rows.append({
            "strategy": name,
            "sharpe_net": point[j],
            "sharpe_ci_low": np.nanquantile(boot[:, j], lo_q),
            "sharpe_ci_high": np.nanquantile(boot[:, j], hi_q),
            "sharpe_diff_vs_benchmark": point[j] - point[bench_col],
            "diff_ci_low": np.nanquantile(diff, lo_q) if name != benchmark else np.nan,
            "diff_ci_high": np.nanquantile(diff, hi_q) if name != benchmark else np.nan,
            "share_resamples_beating_benchmark": float(np.nanmean(diff > 0))
            if name != benchmark else np.nan,
        })
    return pd.DataFrame(rows)

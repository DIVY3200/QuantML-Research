"""Performance and classification metrics.

Conventions
-----------
* Inputs are *decimal* periodic returns (0.01 = 1%). Outputs are decimals unless the key ends
  in ``_count``; the dashboard formats them as percentages.
* Annualisation assumes ``periods_per_year`` observations per year (252 for daily equity data;
  a convention - the actual number of trading days varies by year and exchange).
* The annual risk-free rate is converted geometrically: ``(1 + rf) ** (1 / periods) - 1``.
* Undefined values (zero volatility, too few observations, no drawdown) are returned as NaN,
  never as infinity.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)

NAN = float("nan")
_EPS = 1e-12


def periodic_risk_free(annual_rate: float, periods_per_year: int = 252) -> float:
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def cumulative_return(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    return float((1 + r).prod() - 1) if len(r) else NAN


def cagr(returns: pd.Series, periods_per_year: int = 252, min_obs: int = 2) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < max(min_obs, 1):
        return NAN
    growth = float((1 + r).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(r)) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252, min_obs: int = 2) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < max(min_obs, 2):
        return NAN
    return float(r.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0,
                 min_obs: int = 2) -> float:
    """Annualised Sharpe ratio of excess returns; NaN when volatility is (near) zero."""
    r = pd.Series(returns).dropna()
    if len(r) < max(min_obs, 2):
        return NAN
    excess = r - periodic_risk_free(risk_free_rate, periods_per_year)
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < _EPS:
        return NAN
    return float(excess.mean() / sd * math.sqrt(periods_per_year))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown from the running peak of the compounded equity curve (starting at 1.0)."""
    equity = (1 + pd.Series(returns).fillna(0.0)).cumprod()
    peak = np.maximum(equity.cummax(), 1.0)
    return equity / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    return float(drawdown_series(r).min()) if len(r) else NAN


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252, min_obs: int = 2) -> float:
    growth = cagr(returns, periods_per_year, min_obs)
    mdd = max_drawdown(returns)
    if not np.isfinite(growth) or not np.isfinite(mdd) or abs(mdd) < _EPS:
        return NAN
    return growth / abs(mdd)


def daily_win_rate(net_returns: pd.Series, exposure: pd.Series) -> float:
    """Share of days with market exposure on which the net return was positive."""
    in_market = pd.Series(exposure).reindex(net_returns.index).fillna(0) > 0
    if not in_market.any():
        return NAN
    return float((net_returns[in_market] > 0).mean())


def round_trip_returns(daily: pd.DataFrame) -> pd.Series:
    """Compounded net return of each holding period (entry day through exit day inclusive).

    Works for both execution modes: the entry day carries the entry cost and the exit day carries
    the final return of the old position plus the exit cost. A position still open at the end of
    the window is included, marked to market.
    """
    pos = daily["position"].to_numpy()
    held_into = np.concatenate([[0.0], pos[:-1]])
    active = (pos > 0) | (held_into > 0)
    starts = np.flatnonzero(active & (held_into == 0))
    ends = np.flatnonzero(active & (pos == 0))
    net = daily["net_return"].to_numpy()
    out = []
    for s in starts:
        later_ends = ends[ends >= s]
        e = later_ends[0] if len(later_ends) else len(pos) - 1
        out.append(float(np.prod(1 + net[s:e + 1]) - 1))
    return pd.Series(out, dtype=float)


def summarize_backtest(daily: pd.DataFrame, periods_per_year: int = 252, risk_free_rate: float = 0.0,
                       min_obs_annualize: int = 60) -> dict[str, Any]:
    """All headline metrics for one backtest (see :data:`METRIC_INFO` for units)."""
    net, gross = daily["net_return"], daily["gross_return"]
    n = len(daily)
    years = n / periods_per_year
    trips = round_trip_returns(daily)
    kw = dict(periods_per_year=periods_per_year, min_obs=min_obs_annualize)
    total_turnover = float(daily["turnover"].sum())
    return {
        "start": daily.index[0].strftime("%Y-%m-%d"),
        "end": daily.index[-1].strftime("%Y-%m-%d"),
        "n_days_count": n,
        "cumulative_return_net": cumulative_return(net),
        "cumulative_return_gross": cumulative_return(gross),
        "cagr_net": cagr(net, **kw),
        "annualized_volatility": annualized_volatility(net, **kw),
        "sharpe_net": sharpe_ratio(net, risk_free_rate=risk_free_rate, **kw),
        "sharpe_gross": sharpe_ratio(gross, risk_free_rate=risk_free_rate, **kw),
        "max_drawdown": max_drawdown(net),
        "calmar_ratio": calmar_ratio(net, **kw),
        "daily_win_rate": daily_win_rate(net, daily["position"]),
        "trade_win_rate": float((trips > 0).mean()) if len(trips) else NAN,
        "position_changes_count": int((daily["turnover"] > 0).sum()),
        "round_trips_count": len(trips),
        "total_turnover": total_turnover,
        "annualized_turnover": total_turnover / years if n >= min_obs_annualize else NAN,
        "average_exposure": float(daily["position"].mean()),
        "total_costs": float(daily["total_cost"].sum()),
        "cost_drag": cumulative_return(gross) - cumulative_return(net),
    }


METRIC_INFO: dict[str, tuple[str, str]] = {
    "start": ("Start date", "text"),
    "end": ("End date", "text"),
    "n_days_count": ("Trading days", "count"),
    "cumulative_return_net": ("Cumulative return (net)", "pct"),
    "cumulative_return_gross": ("Cumulative return (gross)", "pct"),
    "cagr_net": ("CAGR (net)", "pct"),
    "annualized_volatility": ("Annualised volatility", "pct"),
    "sharpe_net": ("Sharpe ratio (net)", "ratio"),
    "sharpe_gross": ("Sharpe ratio (gross)", "ratio"),
    "max_drawdown": ("Maximum drawdown", "pct"),
    "calmar_ratio": ("Calmar ratio", "ratio"),
    "daily_win_rate": ("Win rate (days in market)", "pct"),
    "trade_win_rate": ("Win rate (round trips)", "pct"),
    "position_changes_count": ("Position changes", "count"),
    "round_trips_count": ("Round trips", "count"),
    "total_turnover": ("Total turnover (x equity)", "ratio"),
    "annualized_turnover": ("Annualised turnover (x equity)", "ratio"),
    "average_exposure": ("Average exposure", "pct"),
    "total_costs": ("Total costs (sum, % of equity)", "pct"),
    "cost_drag": ("Cost drag (gross - net cumulative)", "pct"),
}


def metrics_table(summaries: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Rows = metric keys, columns = strategy names (raw decimal values)."""
    return pd.DataFrame(summaries)


def format_metric(value: Any, unit: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if unit == "pct":
        return f"{value:.2%}"
    if unit == "ratio":
        return f"{value:.2f}"
    if unit == "count":
        return f"{int(value):,}"
    return str(value)


def format_metrics_table(table: pd.DataFrame) -> pd.DataFrame:
    """Human-readable copy of :func:`metrics_table` output with units in the labels."""
    rows = {}
    for key in table.index:
        label, unit = METRIC_INFO.get(key, (key, "ratio"))
        rows[label] = [format_metric(v, unit) for v in table.loc[key]]
    return pd.DataFrame(rows, index=table.columns).T


def classification_metrics(y_true: pd.Series, probabilities: pd.Series,
                           threshold: float = 0.5) -> dict[str, Any]:
    """Classification diagnostics; ROC-AUC and log loss only when both classes are present."""
    mask = pd.Series(y_true).notna() & pd.Series(probabilities).notna()
    y = pd.Series(y_true)[mask].astype(int).to_numpy()
    p = pd.Series(probabilities)[mask].to_numpy(dtype=float)
    if len(y) == 0:
        return {"n_count": 0}
    pred = (p >= threshold).astype(int)
    both_classes = len(np.unique(y)) == 2
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n_count": int(len(y)),
        "positive_rate": float(y.mean()),
        "predicted_positive_rate": float(pred.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
        "precision": float(precision_score(y, pred, zero_division=0)) if pred.any() else NAN,
        "recall": float(recall_score(y, pred, zero_division=0)) if y.any() else NAN,
        "f1": float(f1_score(y, pred, zero_division=0)) if (pred.any() or y.any()) else NAN,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else NAN,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "confusion_matrix": cm.tolist(),
    }

"""Performance metrics, undefined-value handling, classification metrics and the bootstrap."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantml.backtester import CostModel, run_backtest
from quantml.evaluation import block_bootstrap
from quantml.metrics import (annualized_volatility, cagr, calmar_ratio, classification_metrics,
                             cumulative_return, daily_win_rate, format_metric, max_drawdown,
                             periodic_risk_free, round_trip_returns, sharpe_ratio, summarize_backtest)

from conftest import make_prices


def test_cumulative_return_and_cagr():
    r = pd.Series([0.1, -0.05, 0.02])
    assert cumulative_return(r) == pytest.approx(1.1 * 0.95 * 1.02 - 1)
    assert cagr(r, periods_per_year=3) == pytest.approx(1.1 * 0.95 * 1.02 - 1)
    assert cagr(pd.Series([0.01] * 126), 252) == pytest.approx(1.01 ** 252 - 1)


def test_volatility_and_sharpe_hand_calculation():
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01])
    sd = np.std(r, ddof=1)
    assert annualized_volatility(r, 252) == pytest.approx(sd * math.sqrt(252))
    assert sharpe_ratio(r, 252) == pytest.approx(r.mean() / sd * math.sqrt(252))
    rf = 0.05
    rf_d = (1.05) ** (1 / 252) - 1
    assert periodic_risk_free(rf, 252) == pytest.approx(rf_d)
    assert sharpe_ratio(r, 252, rf) == pytest.approx((r.mean() - rf_d) / sd * math.sqrt(252))


def test_max_drawdown_and_calmar():
    r = pd.Series([0.10, -0.20, 0.05, -0.10, 0.30])
    # equity: 1.1, 0.88, 0.924, 0.8316, 1.08108 -> worst 0.8316 / 1.1 - 1
    assert max_drawdown(r) == pytest.approx(0.8316 / 1.1 - 1)
    assert calmar_ratio(r, 5) == pytest.approx(cagr(r, 5) / abs(0.8316 / 1.1 - 1))
    assert max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1), "peak includes start"


def test_undefined_metrics_are_nan_not_infinite():
    flat = pd.Series([0.0] * 100)
    assert math.isnan(sharpe_ratio(flat))
    assert math.isnan(calmar_ratio(pd.Series([0.01] * 100))), "no drawdown -> undefined Calmar"
    assert math.isnan(sharpe_ratio(pd.Series([0.01])))
    assert math.isnan(cagr(pd.Series([0.01] * 10), min_obs=60))
    assert math.isnan(cumulative_return(pd.Series([], dtype=float)))
    assert math.isnan(daily_win_rate(pd.Series([0.01, 0.02]), pd.Series([0, 0])))
    assert cagr(pd.Series([-1.0, 0.1]), 2) == -1.0
    assert format_metric(float("nan"), "pct") == "n/a"
    assert format_metric(0.1234, "pct") == "12.34%"


def test_round_trips_and_win_rates():
    prices = make_prices([100, 100, 110, 110, 100, 90, 95, 100])
    sig = pd.Series([1, 1, 0, 0, 1, 0, 1, 1], index=prices.index, dtype=float)
    bt = run_backtest(prices, sig, CostModel(0, 0), "next_close")
    trips = round_trip_returns(bt.daily)
    # positions = [0,1,1,0,0,1,0,1]. Trip 1 (days 1-3) earns day 2 (+10%); trip 2 (days 5-6) is
    # filled at the close of day 5, so it misses day 5's -10% and earns day 6 (95/90); trip 3 is
    # still open at the end (filled at the final close, earns nothing yet).
    np.testing.assert_allclose(trips.to_numpy(), [0.10, 95 / 90 - 1, 0.0])
    summary = summarize_backtest(bt.daily, min_obs_annualize=1)
    assert summary["round_trips_count"] == 3
    assert summary["trade_win_rate"] == pytest.approx(2 / 3)
    assert summary["position_changes_count"] == 5
    assert summary["average_exposure"] == pytest.approx(bt.positions.mean())


def test_summary_gross_vs_net_and_costs():
    prices = make_prices(list(np.linspace(100, 120, 80)))
    sig = pd.Series(np.tile([1.0, 0.0], 40), index=prices.index)
    bt = run_backtest(prices, sig, CostModel(10, 5), "next_close")
    s = summarize_backtest(bt.daily, min_obs_annualize=60)
    assert s["total_costs"] == pytest.approx(s["total_turnover"] * 15e-4)
    assert s["cumulative_return_gross"] > s["cumulative_return_net"]
    assert s["cost_drag"] == pytest.approx(s["cumulative_return_gross"] - s["cumulative_return_net"])
    assert not math.isnan(s["annualized_turnover"])


def test_classification_metrics_and_single_class_auc():
    y = pd.Series([1, 0, 1, 1, 0])
    p = pd.Series([0.9, 0.2, 0.6, 0.4, 0.55])
    m = classification_metrics(y, p, 0.5)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    assert m["majority_baseline_accuracy"] == pytest.approx(0.6)
    one_class = classification_metrics(pd.Series([1, 1, 1]), pd.Series([0.6, 0.7, 0.4]))
    assert math.isnan(one_class["roc_auc"])


def test_block_bootstrap_is_reproducible_and_paired():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"bench": rng.normal(0.0005, 0.01, 500)})
    data["copy"] = data["bench"]
    a = block_bootstrap(data, "bench", n_resamples=200, block_size=10, seed=1)
    b = block_bootstrap(data, "bench", n_resamples=200, block_size=10, seed=1)
    pd.testing.assert_frame_equal(a, b)
    row = a.set_index("strategy").loc["copy"]
    assert row["diff_ci_low"] == pytest.approx(0) and row["diff_ci_high"] == pytest.approx(0)
    assert row["sharpe_ci_low"] < row["sharpe_net"] < row["sharpe_ci_high"]
    assert block_bootstrap(data.head(10), "bench", block_size=10).empty

"""Execution timing, costs, returns and equity curves of the backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantml.backtester import CostModel, run_backtest

from conftest import make_prices

ZERO = CostModel(0.0, 0.0)


def signals_for(prices, values):
    return pd.Series(values, index=prices.index, dtype=float)


def test_position_is_previous_signal_and_window_starts_in_cash():
    prices = make_prices([100, 101, 102, 103, 104])
    bt = run_backtest(prices, signals_for(prices, [1, 1, 0, 1, 1]), ZERO, "next_close")
    np.testing.assert_array_equal(bt.positions.to_numpy(), [0, 1, 1, 0, 1])


def test_next_close_returns_hand_calculation():
    prices = make_prices([100, 110, 99, 99, 108.9])
    bt = run_backtest(prices, signals_for(prices, [1, 1, 0, 0, 0]), ZERO, "next_close")
    # Filled at close of day 1 (signal day 0) -> earns day 2 (-10%) only; exited at close of day 2.
    np.testing.assert_allclose(bt.gross_returns.to_numpy(), [0, 0, -0.1, 0, 0])


def test_next_open_splits_overnight_and_intraday_returns():
    prices = make_prices(close=[100, 100, 110, 121, 121], open_=[100, 105, 100, 110, 133.1])
    bt = run_backtest(prices, signals_for(prices, [1, 1, 0, 0, 0]), ZERO, "next_open")
    # day1: buy at open 105 -> intraday 100/105-1 ; day2: overnight 100/100, intraday 110/100
    # day3: exit at open: overnight 110/110-1 = 0, then cash.
    expected = [0.0, 100 / 105 - 1, (1 + 0.0) * (1 + 0.1) - 1, 0.0, 0.0]
    np.testing.assert_allclose(bt.gross_returns.to_numpy(), expected)


def test_trade_cannot_earn_move_before_execution():
    """A signal at close t must not earn the gap or the move that precede its execution."""
    # Huge overnight gap between close of day 2 and open of day 3, then flat.
    prices = make_prices(close=[100, 100, 100, 150, 150, 150], open_=[100, 100, 100, 150, 150, 150])
    signal = signals_for(prices, [0, 0, 1, 1, 1, 1])                 # decided at close of day 2
    for execution in ("next_open", "next_close"):
        bt = run_backtest(prices, signal, ZERO, execution)
        assert bt.daily["equity_net"].iloc[-1] == pytest.approx(1.0), execution
        assert bt.daily.loc[prices.index[3], "gross_return"] == 0.0

    # An 'oracle' that goes long on the day the jump closes still cannot capture it.
    jump = make_prices([100, 100, 100, 150, 150, 150])
    oracle = signals_for(jump, [0, 0, 0, 1, 0, 0])
    bt = run_backtest(jump, oracle, ZERO, "next_close")
    assert bt.gross_returns.abs().sum() == 0.0


def test_signal_on_final_day_is_never_executed():
    prices = make_prices([100, 101, 102, 103])
    bt = run_backtest(prices, signals_for(prices, [0, 0, 0, 1]), CostModel(5, 5), "next_open")
    assert bt.positions.sum() == 0 and bt.daily["total_cost"].sum() == 0
    assert len(bt.daily) == 4, "no extra period appended"


def test_entry_and_exit_costs_on_turnover():
    prices = make_prices([100, 100, 100, 100, 100, 100])
    costs = CostModel(fee_bps=3, slippage_bps=2)
    bt = run_backtest(prices, signals_for(prices, [1, 1, 0, 0, 1, 1]), costs, "next_open")
    np.testing.assert_array_equal(bt.daily["turnover"].to_numpy(), [0, 1, 0, 1, 0, 1])
    np.testing.assert_allclose(bt.daily["fee_cost"].to_numpy(), np.array([0, 1, 0, 1, 0, 1]) * 3e-4)
    np.testing.assert_allclose(bt.daily["slippage_cost"].to_numpy(), np.array([0, 1, 0, 1, 0, 1]) * 2e-4)
    np.testing.assert_allclose(bt.net_returns.to_numpy(), -np.array([0, 1, 0, 1, 0, 1]) * 5e-4)
    assert list(bt.trades["side"]) == ["BUY", "SELL", "BUY"]
    assert bt.trades["signal_date"].iloc[0] == prices.index[0]
    assert bt.trades["execution_date"].iloc[0] == prices.index[1]


def test_partial_position_changes_cost_proportionally():
    prices = make_prices([100, 100, 100, 100])
    bt = run_backtest(prices, signals_for(prices, [0.5, 1.0, 1.0, 1.0]), CostModel(10, 0), "next_close")
    np.testing.assert_allclose(bt.daily["total_cost"].to_numpy(), [0, 5e-4, 5e-4, 0])


def test_zero_cost_net_equals_gross_and_equity_compounds():
    prices = make_prices([100, 102, 99, 105, 107])
    bt = run_backtest(prices, signals_for(prices, [1] * 5), ZERO, "next_close", initial_capital=1000)
    pd.testing.assert_series_equal(bt.net_returns, bt.gross_returns, check_names=False)
    expected = 1000 * np.cumprod(1 + bt.net_returns.to_numpy())
    np.testing.assert_allclose(bt.equity.to_numpy(), expected)
    assert bt.equity.iloc[-1] == pytest.approx(1000 * 107 / 102)


def test_buy_and_hold_total_return_matches_execution_convention():
    prices = make_prices(close=[100, 104, 108, 112], open_=[99, 102, 107, 111])
    ones = signals_for(prices, [1, 1, 1, 1])
    nxt_open = run_backtest(prices, ones, ZERO, "next_open")
    nxt_close = run_backtest(prices, ones, ZERO, "next_close")
    assert nxt_open.equity.iloc[-1] - 1 == pytest.approx(112 / 102 - 1)    # bought at open of day 1
    assert nxt_close.equity.iloc[-1] - 1 == pytest.approx(112 / 104 - 1)   # bought at close of day 1


def test_window_uses_prior_history_and_identical_dates():
    prices = make_prices(list(np.linspace(100, 130, 30)))
    sig = signals_for(prices, [1] * 30)
    bt = run_backtest(prices, sig, ZERO, "next_close", start=prices.index[10], end=prices.index[19])
    assert bt.daily.index[0] == prices.index[10] and len(bt.daily) == 10
    assert bt.positions.iloc[0] == 0, "each window starts in cash"


def test_drawdown_is_measured_from_initial_capital():
    prices = make_prices([100, 100, 90, 95, 100])
    bt = run_backtest(prices, signals_for(prices, [1] * 5), ZERO, "next_close")
    assert bt.daily["drawdown"].min() == pytest.approx(-0.1)
    assert bt.daily["drawdown"].iloc[-1] == pytest.approx(0.0)


def test_invalid_signals_and_inputs_are_rejected():
    prices = make_prices([100, 101, 102])
    with pytest.raises(ValueError, match="long/cash"):
        run_backtest(prices, signals_for(prices, [1, -1, 0]))
    with pytest.raises(ValueError, match="long/cash"):
        run_backtest(prices, signals_for(prices, [2, 1, 0]))
    with pytest.raises(ValueError):
        run_backtest(prices, signals_for(prices, [1, 1, 1]), execution="same_close")
    with pytest.raises(ValueError):
        CostModel(fee_bps=-1)


def test_missing_signals_mean_cash():
    prices = make_prices([100, 110, 121])
    bt = run_backtest(prices, signals_for(prices, [np.nan, np.nan, 1]), ZERO, "next_close")
    assert bt.positions.sum() == 0

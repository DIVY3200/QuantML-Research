"""Transparent long/cash backtester with one execution convention for every strategy.

Execution model
---------------
* A strategy emits a target position ``s[t]`` in ``[0, 1]`` after the close of day ``t``.
* It is executed at the *next* trading observation:

  - ``next_open``: filled at ``open[t+1]``. On day ``t+1`` the new position earns only the
    intraday move ``open -> close``; the overnight gap ``close[t] -> open[t+1]`` belongs to the
    position that was held before the trade.
  - ``next_close``: filled at ``close[t+1]``; the new position first earns day ``t+2``.

* Therefore ``position[t] = s[t-1]`` (the position held after execution on day ``t``). Each
  window starts in cash, so the first day never earns a return from a signal, and the signal
  generated on the final day is never executed: no period is gained or lost.

Daily returns (decimals)
------------------------
* ``next_open``:  ``gross[t] = (1 + position[t-1] * overnight[t]) * (1 + position[t] * intraday[t]) - 1``
* ``next_close``: ``gross[t] = position[t-1] * (close[t] / close[t-1] - 1)``

Turnover and costs
------------------
``turnover[t] = |position[t] - position[t-1]|``; a 0 -> 1 entry and a 1 -> 0 exit each count
1.0. ``cost[t] = turnover[t] * (fee_bps + slippage_bps) / 10_000`` is charged as a fraction of
equity on the execution day: ``net[t] = gross[t] - cost[t]``. Positions are marked to market at
the final close and are not force-liquidated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantml.config import EXECUTION_MODES

BPS = 10_000.0


@dataclass(frozen=True)
class CostModel:
    """Per-unit-turnover trading costs in basis points (a generic assumption, not a broker quote)."""

    fee_bps: float = 1.0
    slippage_bps: float = 2.0

    def __post_init__(self) -> None:
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Costs must be non-negative.")

    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.slippage_bps


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame
    trades: pd.DataFrame
    execution: str
    costs: CostModel
    initial_capital: float

    @property
    def net_returns(self) -> pd.Series:
        return self.daily["net_return"]

    @property
    def gross_returns(self) -> pd.Series:
        return self.daily["gross_return"]

    @property
    def positions(self) -> pd.Series:
        return self.daily["position"]

    @property
    def equity(self) -> pd.Series:
        return self.daily["equity_net"]


def run_backtest(prices: pd.DataFrame, signals: pd.Series, costs: CostModel | None = None,
                 execution: str = "next_open", start: object = None, end: object = None,
                 initial_capital: float = 1.0, name: str = "strategy") -> BacktestResult:
    """Backtest target-position ``signals`` over ``prices.loc[start:end]``.

    ``prices`` must contain adjusted ``open`` and ``close``. Return calculations use the full
    price history, so the first day of a window that starts mid-sample has a proper prior close.
    """
    costs = costs or CostModel()
    if execution not in EXECUTION_MODES:
        raise ValueError(f"execution must be one of {EXECUTION_MODES}")
    missing = {"open", "close"} - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing columns {sorted(missing)}")

    close, open_ = prices["close"], prices["open"]
    window = prices.loc[start:end].index
    if len(window) < 2:
        raise ValueError("The backtest window needs at least two observations.")

    signal = signals.reindex(window).astype(float)
    known = signal.dropna()
    if ((known < 0) | (known > 1)).any():
        raise ValueError("Signals must lie in [0, 1]: this engine supports long/cash only "
                         "(no shorting or leverage).")
    target = signal.fillna(0.0)

    position = target.shift(1).fillna(0.0)          # s[t-1], executed on day t
    prev_position = position.shift(1).fillna(0.0)    # held into day t

    overnight = (open_ / close.shift(1) - 1).reindex(window).fillna(0.0)
    intraday = (close / open_ - 1).reindex(window)
    close_to_close = close.pct_change(fill_method=None).reindex(window).fillna(0.0)

    if execution == "next_open":
        gross = (1 + prev_position * overnight) * (1 + position * intraday) - 1
        exec_price = open_.reindex(window)
    else:
        gross = prev_position * close_to_close
        exec_price = close.reindex(window)

    trade = position - prev_position
    turnover = trade.abs()
    fee = turnover * costs.fee_bps / BPS
    slippage = turnover * costs.slippage_bps / BPS
    total_cost = fee + slippage
    net = gross - total_cost

    equity_net = initial_capital * (1 + net).cumprod()
    equity_gross = initial_capital * (1 + gross).cumprod()
    peak = np.maximum(equity_net.cummax(), initial_capital)
    signal_dates = pd.Series(window, index=window).shift(1)

    daily = pd.DataFrame({
        "open": open_.reindex(window),
        "close": close.reindex(window),
        "signal": signal,
        "position": position,
        "trade": trade,
        "turnover": turnover,
        "signal_date": signal_dates.where(turnover > 0),
        "execution_price": exec_price.where(turnover > 0),
        "gross_return": gross,
        "fee_cost": fee,
        "slippage_cost": slippage,
        "total_cost": total_cost,
        "net_return": net,
        "cumulative_cost": total_cost.cumsum(),
        "equity_gross": equity_gross,
        "equity_net": equity_net,
        "drawdown": equity_net / peak - 1,
    }, index=window)
    daily.index.name = "date"

    executed = daily[turnover > 0]
    trades = pd.DataFrame({
        "signal_date": executed["signal_date"],
        "execution_date": executed.index,
        "side": np.where(executed["trade"] > 0, "BUY", "SELL"),
        "from_position": prev_position[turnover > 0],
        "to_position": executed["position"],
        "execution_price": executed["execution_price"],
        "cost": executed["total_cost"],
    }).reset_index(drop=True)
    return BacktestResult(name, daily, trades, execution, costs, initial_capital)


def run_backtests(prices: pd.DataFrame, signals: dict[str, pd.Series], costs: CostModel | None = None,
                  execution: str = "next_open", start: object = None, end: object = None,
                  initial_capital: float = 1.0) -> dict[str, BacktestResult]:
    """Backtest several strategies on identical dates with identical assumptions."""
    return {name: run_backtest(prices, sig, costs, execution, start, end, initial_capital, name)
            for name, sig in signals.items()}

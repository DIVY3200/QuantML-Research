"""Trading strategies behind a common interface.

A strategy maps price history to a *target position* series ``s[t]`` in ``[0, 1]`` (0 = cash,
1 = fully long) that is decided after the close of day ``t`` using information up to ``t``.
Strategies never decide when a trade is filled - the backtester applies one execution
convention to every strategy. NaN means "no signal available yet" and is treated as cash.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Base class: ``generate_signals`` returns target positions indexed like ``prices``."""

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Target position decided at each date's close (NaN = no signal yet)."""

    @abstractmethod
    def describe(self) -> str:
        """One-sentence description of the trading rule."""


class BuyAndHold(Strategy):
    """Passive benchmark: target a full long position on every date."""

    name = "Buy & Hold"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index, name=self.name)

    def describe(self) -> str:
        return ("Enter a fully long position at the first executable point of the evaluation "
                "window and hold it to the end (marked to market, not liquidated).")


class MovingAverageCrossover(Strategy):
    """Long when the short simple moving average is above the long one, otherwise cash."""

    def __init__(self, short_window: int = 20, long_window: int = 50, price_col: str = "close"):
        if not 1 <= short_window < long_window:
            raise ValueError("short_window must be >= 1 and smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.price_col = price_col
        self.name = f"MA Crossover ({short_window}/{long_window})"

    def moving_averages(self, prices: pd.DataFrame) -> pd.DataFrame:
        px = prices[self.price_col]
        return pd.DataFrame({
            "sma_short": px.rolling(self.short_window, min_periods=self.short_window).mean(),
            "sma_long": px.rolling(self.long_window, min_periods=self.long_window).mean(),
        }, index=prices.index)

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        ma = self.moving_averages(prices)
        signal = (ma["sma_short"] > ma["sma_long"]).astype(float)
        return signal.where(ma["sma_long"].notna()).rename(self.name)

    def describe(self) -> str:
        return (f"Long when SMA{self.short_window} > SMA{self.long_window} of the adjusted close "
                "at the signal date's close; otherwise cash.")


class ProbabilityThresholdStrategy(Strategy):
    """Convert model probabilities P(up) into long/cash target positions.

    Probabilities must be out-of-sample for the dates being evaluated; they are produced by the
    research pipeline, which fits the model on earlier data only.
    """

    def __init__(self, probabilities: pd.Series, threshold: float = 0.5,
                 below_threshold_position: float = 0.0, name: str = "ML Strategy"):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        if not 0.0 <= below_threshold_position < 1.0:
            raise ValueError("below_threshold_position must be in [0, 1)")
        self.probabilities = probabilities
        self.threshold = threshold
        self.below_threshold_position = below_threshold_position
        self.name = name

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        prob = self.probabilities.reindex(prices.index)
        signal = pd.Series(np.where(prob >= self.threshold, 1.0, self.below_threshold_position),
                           index=prices.index, name=self.name)
        return signal.where(prob.notna())

    def describe(self) -> str:
        below = ("cash" if self.below_threshold_position == 0
                 else f"{self.below_threshold_position:.0%} exposure")
        return (f"Long when the model's predicted probability of a positive next-period return "
                f"is >= {self.threshold:.2f}; otherwise {below}.")

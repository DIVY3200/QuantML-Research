"""Signal generation for the three strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantml.strategies import BuyAndHold, MovingAverageCrossover, ProbabilityThresholdStrategy

from conftest import make_prices


def test_buy_and_hold_targets_full_long():
    prices = make_prices([1, 2, 3, 4])
    assert (BuyAndHold().generate_signals(prices) == 1.0).all()


def test_ma_crossover_hand_example():
    prices = make_prices([10, 10, 10, 13, 16, 7, 4])
    sig = MovingAverageCrossover(2, 3).generate_signals(prices)
    # SMA2: nan,10,10,11.5,14.5,11.5,5.5 | SMA3: nan,nan,10,11,13,12,9
    expected = [np.nan, np.nan, 0.0, 1.0, 1.0, 0.0, 0.0]
    np.testing.assert_array_equal(sig.to_numpy(), expected)


def test_ma_crossover_uses_only_past_prices():
    prices = make_prices(list(np.linspace(100, 120, 60)))
    strat = MovingAverageCrossover(5, 20)
    base = strat.generate_signals(prices)
    shocked = prices.copy()
    shocked.iloc[40:, :] *= 0.5
    after = strat.generate_signals(shocked)
    pd.testing.assert_series_equal(base.iloc[:40], after.iloc[:40])


def test_ma_crossover_validates_windows():
    with pytest.raises(ValueError):
        MovingAverageCrossover(50, 20)
    with pytest.raises(ValueError):
        MovingAverageCrossover(20, 20)


def test_probability_threshold_strategy():
    prices = make_prices([1, 2, 3, 4, 5])
    prob = pd.Series([np.nan, 0.49, 0.5, 0.7, 0.2], index=prices.index)
    sig = ProbabilityThresholdStrategy(prob, threshold=0.5).generate_signals(prices)
    np.testing.assert_array_equal(sig.to_numpy(), [np.nan, 0.0, 1.0, 1.0, 0.0])
    neutral = ProbabilityThresholdStrategy(prob, 0.5, below_threshold_position=0.5)
    np.testing.assert_array_equal(neutral.generate_signals(prices).to_numpy(),
                                  [np.nan, 0.5, 1.0, 1.0, 0.5])


def test_probability_threshold_validates_inputs():
    prob = pd.Series([0.5])
    with pytest.raises(ValueError):
        ProbabilityThresholdStrategy(prob, threshold=1.0)
    with pytest.raises(ValueError):
        ProbabilityThresholdStrategy(prob, threshold=0.5, below_threshold_position=1.0)

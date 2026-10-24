"""Feature values against hand calculations, look-ahead safety and target alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantml import features as features_module
from quantml.config import FeatureConfig, TargetConfig
from quantml.features import audit_lookahead, build_features, max_lookback
from quantml.labels import forward_returns, label_lookahead, make_labels

from conftest import make_prices

SMALL = FeatureConfig(return_lags=2, momentum_windows=[2], volatility_windows=[3],
                      ma_distance_windows=[3], ma_ratio_pairs=[[2, 3]], range_window=3,
                      volume_window=3, use_volume="true")


def test_features_match_hand_calculations():
    close = [100.0, 102.0, 101.0, 104.0, 103.0]
    prices = make_prices(close)
    prices["volume"] = [10.0, 20.0, 30.0, 60.0, 30.0]
    f = build_features(prices, SMALL)
    t = prices.index[3]                                   # close = 104

    assert f.loc[t, "ret_lag_0"] == pytest.approx(104 / 101 - 1)
    assert f.loc[t, "ret_lag_1"] == pytest.approx(101 / 102 - 1)
    assert f.loc[t, "mom_2"] == pytest.approx(104 / 102 - 1)
    rets = np.array([102 / 100 - 1, 101 / 102 - 1, 104 / 101 - 1])
    assert f.loc[t, "vol_3"] == pytest.approx(rets.std(ddof=1))
    assert f.loc[t, "dist_ma_3"] == pytest.approx(104 / np.mean([102, 101, 104]) - 1)
    assert f.loc[t, "ma_ratio_2_3"] == pytest.approx(np.mean([101, 104]) / np.mean([102, 101, 104]) - 1)
    assert f.loc[t, "hl_range"] == pytest.approx((104 * 1.01 - 104 * 0.99) / 104)
    hh = max(102 * 1.01, 101 * 1.01, 104 * 1.01)          # highs of rows 1..3 (open == close)
    ll = min(102 * 0.99, 101 * 0.99, 104 * 0.99)
    assert f.loc[t, "range_pos_3"] == pytest.approx((104 - ll) / (hh - ll))
    assert f.loc[t, "volume_ratio_3"] == pytest.approx(60 / np.mean([20, 30, 60]) - 1)
    assert f.loc[t, "volume_change_1d"] == pytest.approx(np.log(61) - np.log(31))


def test_warmup_rows_are_nan_not_invented(synthetic_prices):
    cfg = FeatureConfig()
    f = build_features(synthetic_prices, cfg)
    lookback = max_lookback(cfg)
    assert f.iloc[: lookback - 1].isna().any(axis=1).all()
    assert f.iloc[lookback:].notna().all().all()


def test_future_prices_do_not_change_past_features(synthetic_prices):
    cfg = FeatureConfig()
    base = build_features(synthetic_prices, cfg)
    shocked = synthetic_prices.copy()
    cut = 600
    shocked.iloc[cut:, :4] *= 3.0                           # change everything after the cut
    shocked.iloc[cut:, 4] *= 5.0
    after = build_features(shocked, cfg)
    pd.testing.assert_frame_equal(base.iloc[:cut], after.iloc[:cut])


def test_lookahead_audit_passes_for_real_features(synthetic_prices):
    audit = audit_lookahead(synthetic_prices, FeatureConfig(), n_checks=15)
    assert audit.passed and audit.n_checked == 15 and audit.max_abs_difference == 0


def test_lookahead_audit_detects_a_leaky_feature(synthetic_prices, monkeypatch):
    original = features_module.build_features

    def leaky(prices, cfg, use_volume=None):
        f = original(prices, cfg, use_volume)
        f["tomorrow_return"] = prices["close"].pct_change().shift(-1)   # uses the future
        return f

    monkeypatch.setattr(features_module, "build_features", leaky)
    audit = audit_lookahead(synthetic_prices, FeatureConfig(), n_checks=10)
    assert not audit.passed and audit.offending_dates


def test_close_label_is_next_period_return():
    prices = make_prices([100.0, 101.0, 100.5, 100.5, 102.0])
    labels = make_labels(prices, TargetConfig(horizon=1, basis="close"))
    expected_fwd = [0.01, 100.5 / 101 - 1, 0.0, 102 / 100.5 - 1, np.nan]
    np.testing.assert_allclose(labels["fwd_return"].to_numpy(), expected_fwd)
    np.testing.assert_array_equal(labels["label"].to_numpy(), [1.0, 0.0, 0.0, 1.0, np.nan])
    assert labels["label_end_date"].iloc[0] == prices.index[1]
    assert pd.isna(labels["label_end_date"].iloc[-1])


def test_label_horizon_and_execution_basis():
    prices = make_prices([100, 101, 102, 103, 104, 105], open_=[99, 100, 110, 100, 120, 100])
    fwd2 = forward_returns(prices, horizon=2, basis="close")
    assert fwd2.iloc[0] == pytest.approx(102 / 100 - 1)
    assert fwd2.iloc[-2:].isna().all()
    exec_open = forward_returns(prices, horizon=1, basis="execution", execution="next_open")
    assert exec_open.iloc[0] == pytest.approx(110 / 100 - 1)    # open[t+2] / open[t+1]
    assert label_lookahead(TargetConfig(horizon=1, basis="execution")) == 2
    assert label_lookahead(TargetConfig(horizon=3, basis="close")) == 3


def test_target_is_not_a_feature(synthetic_prices):
    f = build_features(synthetic_prices, FeatureConfig())
    labels = make_labels(synthetic_prices, TargetConfig())
    joined = f.join(labels).dropna()
    for col in f.columns:
        assert not np.allclose(joined[col], joined["fwd_return"]), f"{col} equals the target"

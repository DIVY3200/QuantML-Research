"""Data loading, standardisation, validation and caching (offline)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from quantml import data_loader
from quantml.config import DataConfig, ValidationConfig
from quantml.data_loader import (DataLoadError, load_csv, load_market_data, load_yfinance,
                                 prepare_ohlcv, standardize_columns)
from quantml.data_validation import DataValidationError, validate_ohlcv

CFG = ValidationConfig(min_observations=50)


def checks(report, severity):
    return {i.check for i in report.issues if i.severity == severity}


def test_valid_data_passes(valid_raw):
    report = validate_ohlcv(valid_raw, CFG)
    assert report.ok
    assert report.n_valid_rows == 120
    assert "date_gaps" not in checks(report, "warning"), "weekends must not be flagged as gaps"


def test_duplicate_dates_are_an_error(valid_raw):
    raw = pd.concat([valid_raw, valid_raw.iloc[[10]]], ignore_index=True)
    report = validate_ohlcv(raw, CFG)
    assert not report.ok and "duplicate_dates" in checks(report, "error")
    with pytest.raises(DataValidationError):
        raise DataValidationError(report)


def test_partially_missing_prices_are_an_error(valid_raw):
    raw = valid_raw.copy()
    raw.loc[5, "high"] = np.nan
    report = validate_ohlcv(raw, CFG)
    assert "missing_prices" in checks(report, "error")


def test_empty_placeholder_rows_warn_and_are_dropped_not_filled(valid_raw):
    raw = valid_raw.copy()
    raw.loc[5, ["open", "high", "low", "close", "adj_close"]] = np.nan
    report = validate_ohlcv(raw, CFG)
    assert report.ok and "empty_rows" in checks(report, "warning")
    prices, _ = prepare_ohlcv(raw)
    assert len(prices) == len(raw) - 1
    assert raw.loc[5, "date"] not in prices.index


def test_non_numeric_prices_are_an_error(valid_raw):
    raw = valid_raw.astype({"close": object})
    raw.loc[3, "close"] = "n/a"
    assert "non_numeric" in checks(validate_ohlcv(raw, CFG), "error")


def test_non_positive_prices_are_an_error(valid_raw):
    raw = valid_raw.copy()
    raw.loc[7, "low"] = 0.0
    assert "non_positive_prices" in checks(validate_ohlcv(raw, CFG), "error")


def test_invalid_ohlc_relationship_is_an_error(valid_raw):
    raw = valid_raw.copy()
    raw.loc[9, "high"] = raw.loc[9, "low"] * 0.5
    assert "ohlc_consistency" in checks(validate_ohlcv(raw, CFG), "error")


def test_missing_columns_and_volume(valid_raw):
    no_close = valid_raw.drop(columns=["close"])
    assert "required_columns" in checks(validate_ohlcv(no_close, CFG), "error")
    no_volume = valid_raw.drop(columns=["volume"])
    report = validate_ohlcv(no_volume, CFG)
    assert report.ok and "volume" in checks(report, "warning")


def test_insufficient_observations(valid_raw):
    report = validate_ohlcv(valid_raw.head(30), CFG)
    assert "insufficient_observations" in checks(report, "error")


def test_long_gaps_and_extreme_returns_warn(valid_raw):
    raw = valid_raw.drop(index=range(40, 50)).reset_index(drop=True)
    raw.loc[60, ["open", "high", "low", "close", "adj_close"]] *= 2.0
    report = validate_ohlcv(raw, CFG)
    assert {"date_gaps", "extreme_returns"} <= checks(report, "warning")


def test_unsorted_rows_are_sorted(valid_raw):
    shuffled = valid_raw.sample(frac=1.0, random_state=0).reset_index(drop=True)
    report = validate_ohlcv(shuffled, CFG)
    assert "sorting" in checks(report, "info")
    prices, _ = prepare_ohlcv(shuffled)
    assert prices.index.is_monotonic_increasing


def test_adjustment_is_applied_exactly_once(valid_raw):
    prices, description = prepare_ohlcv(valid_raw)
    np.testing.assert_allclose(prices["close"].to_numpy(), valid_raw["adj_close"].to_numpy())
    np.testing.assert_allclose(prices["open"].to_numpy(), valid_raw["open"].to_numpy() * 0.9)
    np.testing.assert_allclose(prices["raw_close"].to_numpy(), valid_raw["close"].to_numpy())
    assert "Adj Close / Close" in description


def test_standardize_yfinance_multiindex_and_tz():
    idx = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=3, freq="B", tz="America/New_York"),
                           name="Date")
    cols = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"], ["SPY"]],
                                      names=["Price", "Ticker"])
    df = pd.DataFrame(np.ones((3, 6)), index=idx, columns=cols)
    out = standardize_columns(df)
    assert list(out.columns) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert out["date"].dt.tz is None
    assert out["date"].iloc[0] == pd.Timestamp("2024-01-02")


def test_load_csv_handles_aliases_and_thousands_separators():
    text = ("Trade Date,OPEN,High,Low,Close Price,Adj_Close,Shares Traded\n"
            "2024-01-03,\"1,000.5\",1010,990,1005,1005,\"12,000\"\n"
            "2024-01-02,995,1001,994,1000,1000,\"11,000\"\n")
    market = load_csv(io.StringIO(text), ticker="TEST")
    prices, _ = prepare_ohlcv(market.raw)
    assert list(prices.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert prices.loc["2024-01-03", "open"] == pytest.approx(1000.5)
    assert prices.loc["2024-01-03", "volume"] == 12000


def test_load_csv_missing_file_is_actionable():
    with pytest.raises(DataLoadError, match="not found"):
        load_csv("does/not/exist.csv")


def test_synthetic_source_is_labelled():
    market = load_market_data(DataConfig(source="synthetic"))
    assert market.is_synthetic and market.ticker == "SYNTHETIC"
    assert "SYNTHETIC" in market.source_label


def test_csv_source_respects_date_range(tmp_path, valid_raw):
    path = tmp_path / "prices.csv"
    valid_raw.to_csv(path, index=False)
    start, end = valid_raw["date"].iloc[10], valid_raw["date"].iloc[19]
    market = load_market_data(DataConfig(source="csv", csv_path=str(path),
                                         start=str(start.date()), end=str(end.date())))
    assert len(market.raw) == 10


def _fake_download(calls):
    def download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        idx = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=5), name="Date")
        cols = pd.MultiIndex.from_product([["Adj Close", "Close", "High", "Low", "Open", "Volume"],
                                           [ticker]])
        data = np.column_stack([np.full(5, 99.0), np.full(5, 100.0), np.full(5, 101.0),
                                np.full(5, 98.0), np.full(5, 100.0), np.full(5, 1e6)])
        return pd.DataFrame(data, index=idx, columns=cols)
    return download


def test_yfinance_download_is_cached(tmp_path, monkeypatch):
    yf = pytest.importorskip("yfinance")
    calls = []
    monkeypatch.setattr(yf, "download", _fake_download(calls))
    first = load_yfinance("spy", "2020-01-01", "2020-01-10", cache_dir=tmp_path)
    second = load_yfinance("SPY", "2020-01-01", "2020-01-10", cache_dir=tmp_path)
    assert len(calls) == 1, "second call must be served from the cache"
    assert calls[0][1]["auto_adjust"] is False
    assert calls[0][1]["end"] == "2020-01-11", "inclusive end date must be passed as end + 1 day"
    assert not first.from_cache and second.from_cache
    pd.testing.assert_frame_equal(first.raw, second.raw)


def test_yfinance_empty_result_gives_actionable_error(tmp_path, monkeypatch):
    yf = pytest.importorskip("yfinance")
    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(DataLoadError, match=r"\.NS"):
        load_yfinance("NOTAREALTICKER", "2020-01-01", "2020-02-01", cache_dir=tmp_path,
                      use_cache=False)


def test_synthetic_generator_is_deterministic():
    a = data_loader.generate_synthetic_ohlcv(n_days=50, seed=1)
    b = data_loader.generate_synthetic_ohlcv(n_days=50, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()

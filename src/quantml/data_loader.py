"""Load daily OHLCV data from Yahoo Finance (cached), CSV files, or a labelled synthetic generator.

Loading is split into two steps so that data problems are reported rather than silently fixed:

1. ``load_*`` functions return a :class:`MarketData` whose ``raw`` frame has canonical column
   names (``date, open, high, low, close, adj_close, volume``) but untouched values.
2. :func:`prepare_ohlcv` (called after :func:`quantml.data_validation.validate_ohlcv`) coerces
   types, sorts by date, drops provider placeholder rows that contain no prices at all, and applies
   the corporate-action adjustment exactly once.

Price adjustment convention
---------------------------
Yahoo Finance's ``Close`` is split-adjusted but not dividend-adjusted; ``Adj Close`` is adjusted
for splits and dividends. We download with ``auto_adjust=False`` and derive adjusted
Open/High/Low/Close by multiplying the raw columns by ``Adj Close / Close``. We never request
``auto_adjust=True`` *and* apply the factor, which would adjust twice.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from quantml.config import DataConfig, resolve_path

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close")
CANONICAL_COLUMNS = ("date", "open", "high", "low", "close", "adj_close", "volume")

_COLUMN_ALIASES = {
    "date": "date", "datetime": "date", "timestamp": "date", "time": "date",
    "trade date": "date", "trading date": "date", "day": "date",
    "open": "open", "open price": "open",
    "high": "high", "high price": "high",
    "low": "low", "low price": "low",
    "close": "close", "close price": "close", "last": "close", "last price": "close",
    "adj close": "adj_close", "adjclose": "adj_close", "adjusted close": "adj_close",
    "adj close price": "adj_close", "adjusted close price": "adj_close", "close adjusted": "adj_close",
    "volume": "volume", "vol": "volume", "shares traded": "volume",
}


class DataLoadError(RuntimeError):
    """Raised when market data cannot be obtained; the message says what to try next."""


@dataclass
class MarketData:
    """Standardised (but not yet validated) market data plus provenance metadata."""

    raw: pd.DataFrame
    ticker: str
    source: str
    is_synthetic: bool = False
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    from_cache: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def source_label(self) -> str:
        if self.is_synthetic:
            return "SYNTHETIC data (generated, not real market prices)"
        return {"yfinance": "Yahoo Finance via yfinance", "csv": "User-supplied CSV"}.get(
            self.source, self.source)


def _normalise_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[_\-.]+", " ", text)
    return re.sub(r"\s+", " ", text)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema and parse dates (values are not coerced).

    Handles yfinance MultiIndex columns, a date stored in the index, case/spacing variants
    (``Adj Close``, ``adj_close``, ``AdjClose``) and timezone-aware timestamps, which are
    converted to timezone-naive exchange-local trading dates.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    frame = df.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance returns (Price, Ticker); keep the level that carries the price field names.
        for level in range(frame.columns.nlevels):
            names = {_normalise_name(v) for v in frame.columns.get_level_values(level)}
            if "close" in names:
                frame.columns = frame.columns.get_level_values(level)
                break
        else:
            frame.columns = ["_".join(map(str, c)) for c in frame.columns]

    renamed = {c: _COLUMN_ALIASES.get(_normalise_name(c), _normalise_name(c)) for c in frame.columns}
    frame = frame.rename(columns=renamed)
    frame = frame.loc[:, ~frame.columns.duplicated()]

    if "date" not in frame.columns:
        index_name = _normalise_name(frame.index.name) if frame.index.name else ""
        if isinstance(frame.index, pd.DatetimeIndex) or _COLUMN_ALIASES.get(index_name) == "date":
            frame = frame.reset_index().rename(columns={frame.index.name or "index": "date"})
            if "date" not in frame.columns:
                frame = frame.rename(columns={frame.columns[0]: "date"})

    if "date" in frame.columns:
        frame["date"] = _parse_dates(frame["date"])

    keep = [c for c in CANONICAL_COLUMNS if c in frame.columns]
    return frame[keep].reset_index(drop=True)


def _parse_dates(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = values
    else:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if getattr(parsed.dt, "tz", None) is not None:
        # Keep the exchange-local calendar date; dropping tz avoids shifting dates across midnight.
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def coerce_numeric(values: pd.Series) -> pd.Series:
    """Convert a column to floats, accepting thousands separators; invalid entries become NaN."""
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    cleaned = values.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def prepare_ohlcv(raw: pd.DataFrame, is_synthetic: bool = False) -> tuple[pd.DataFrame, str]:
    """Turn validated raw data into the analysis frame used everywhere else.

    Returns ``(prices, adjustment_description)``. ``prices`` is indexed by a tz-naive
    ``DatetimeIndex`` named ``date`` and has adjusted ``open, high, low, close`` plus ``volume``
    (NaN when unavailable), ``raw_close`` and ``adj_factor``.
    """
    frame = raw.copy()
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        if col in frame.columns:
            frame[col] = coerce_numeric(frame[col])

    empty_rows = frame[list(PRICE_COLUMNS)].isna().all(axis=1)
    if empty_rows.any():
        logger.info("Dropping %d placeholder rows with no price data", int(empty_rows.sum()))
        frame = frame.loc[~empty_rows]

    frame = frame.sort_values("date").set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date")

    prices = pd.DataFrame(index=frame.index)
    has_adj = "adj_close" in frame.columns and frame["adj_close"].notna().all() \
        and (frame["adj_close"] > 0).all()
    if has_adj:
        factor = frame["adj_close"] / frame["close"]
        description = ("Adjusted for splits and dividends: raw Open/High/Low/Close multiplied by "
                       "Adj Close / Close (applied once).")
    else:
        factor = pd.Series(1.0, index=frame.index)
        description = ("Synthetic data: no corporate actions exist, prices used as generated."
                       if is_synthetic else
                       "No adjusted close supplied: prices used as given. Returns exclude dividends "
                       "and assume splits are already adjusted in the source.")
    for col in PRICE_COLUMNS:
        prices[col] = frame[col] * factor
    prices["volume"] = frame["volume"] if "volume" in frame.columns else np.nan
    prices["raw_close"] = frame["close"]
    prices["adj_factor"] = factor
    return prices, description


def generate_synthetic_ohlcv(n_days: int = 2500, start: str = "2014-01-02", seed: int = 42,
                             annual_drift: float = 0.06, annual_vol: float = 0.18) -> pd.DataFrame:
    """Generate clearly synthetic daily OHLCV data for tests and offline demonstrations.

    Returns follow a GARCH(1,1)-style volatility process with a small drift and *no* built-in
    predictability, so any apparent model edge on this data is noise. The output must never be
    presented as real market performance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    daily_var = (annual_vol ** 2) / 252
    omega, alpha, beta = daily_var * 0.05, 0.08, 0.87
    var = daily_var
    shocks = rng.standard_normal(n_days)
    log_ret = np.empty(n_days)
    for i in range(n_days):
        eps = np.sqrt(var) * shocks[i]
        log_ret[i] = annual_drift / 252 - 0.5 * var + eps
        var = omega + alpha * eps ** 2 + beta * var

    close = 100.0 * np.exp(np.cumsum(log_ret))
    prev_close = np.concatenate([[100.0], close[:-1]])
    sigma = np.sqrt(daily_var)
    # Split each day's log return into an overnight gap and an intraday move.
    overnight = rng.normal(0.0, 0.3 * sigma, n_days)
    open_ = prev_close * np.exp(overnight)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.4 * sigma, n_days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.4 * sigma, n_days)))
    volume = np.round(rng.lognormal(mean=16.0, sigma=0.3, size=n_days)
                      * (1 + 20 * np.abs(log_ret))).astype(np.int64)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume})


def load_synthetic(n_days: int = 2500, seed: int = 42, start: str = "2014-01-02") -> MarketData:
    raw = generate_synthetic_ohlcv(n_days=n_days, seed=seed, start=start)
    return MarketData(raw=standardize_columns(raw), ticker="SYNTHETIC", source="synthetic",
                      is_synthetic=True,
                      notes=["SYNTHETIC data generated by quantml.data_loader - not real prices."])


def load_csv(source: str | Path | IO, ticker: str = "CSV") -> MarketData:
    """Load OHLCV data from a CSV path or file-like object (e.g. a Streamlit upload)."""
    try:
        if isinstance(source, (str, Path)):
            path = resolve_path(source)
            if not path.exists():
                raise DataLoadError(f"CSV file not found: {path}")
            raw = pd.read_csv(path)
        else:
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8-sig")
            raw = pd.read_csv(io.StringIO(content))
    except DataLoadError:
        raise
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise DataLoadError(f"Could not parse the CSV file: {exc}. Expected a header row with "
                            "Date, Open, High, Low, Close and optionally Adj Close and Volume.") from exc
    frame = standardize_columns(raw)
    return MarketData(raw=frame, ticker=ticker, source="csv",
                      notes=[f"Columns found in CSV: {list(raw.columns)}"])


def _cache_paths(cache_dir: Path, ticker: str, start: str, end: str) -> tuple[Path, Path]:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker.upper())
    stem = f"{safe}_{start}_{end}_1d"
    return cache_dir / f"{stem}.csv", cache_dir / f"{stem}.meta.json"


def load_yfinance(ticker: str, start: str, end: str, cache_dir: str | Path = "data/raw",
                  use_cache: bool = True) -> MarketData:
    """Download daily data with yfinance, caching completed date ranges locally.

    ``end`` is inclusive (yfinance's own ``end`` is exclusive, so one day is added). Ranges that
    reach today or later are not cached because the latest bar may still change.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise DataLoadError("Please enter a ticker symbol, e.g. SPY, AAPL or RELIANCE.NS.")
    start_d, end_d = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if start_d >= end_d:
        raise DataLoadError(f"Start date {start_d} must be before end date {end_d}.")

    cache_root = resolve_path(cache_dir)
    csv_path, meta_path = _cache_paths(cache_root, ticker, str(start_d), str(end_d))
    if use_cache and csv_path.exists():
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        logger.info("Loading %s from cache %s", ticker, csv_path)
        return MarketData(raw=standardize_columns(pd.read_csv(csv_path)), ticker=ticker,
                          source="yfinance", from_cache=True,
                          retrieved_at=meta.get("retrieved_at", "unknown"),
                          notes=[f"Loaded from local cache {csv_path.name}; delete it to re-download."])

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataLoadError("yfinance is not installed. Run `pip install yfinance`, or upload a "
                            "CSV file / use the synthetic dataset instead.") from exc

    logger.info("Downloading %s %s..%s from Yahoo Finance", ticker, start_d, end_d)
    try:
        downloaded = yf.download(ticker, start=str(start_d), end=str(end_d + timedelta(days=1)),
                                 auto_adjust=False, actions=False, progress=False, threads=False)
    except Exception as exc:  # yfinance raises many exception types for network/provider issues
        raise DataLoadError(f"Download of '{ticker}' failed: {exc}. Check your internet "
                            "connection, or upload a CSV file instead.") from exc

    if downloaded is None or downloaded.empty:
        raise DataLoadError(
            f"No data returned for '{ticker}' between {start_d} and {end_d}. Check the symbol "
            "(Indian NSE listings use a '.NS' suffix, e.g. RELIANCE.NS; BSE uses '.BO'), the date "
            "range and your connection. Yahoo coverage varies by market; you can upload a CSV "
            "instead.")

    raw = standardize_columns(downloaded)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    notes = []
    if end_d < date.today():
        cache_root.mkdir(parents=True, exist_ok=True)
        out = raw.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(csv_path, index=False)
        meta_path.write_text(json.dumps({
            "ticker": ticker, "start": str(start_d), "end": str(end_d), "rows": len(raw),
            "retrieved_at": retrieved_at, "provider": "Yahoo Finance via yfinance",
            "yfinance_version": getattr(yf, "__version__", "unknown"),
        }, indent=2))
        notes.append(f"Downloaded and cached to {csv_path.name}.")
    else:
        notes.append("Range reaches today or later; not cached because recent bars may change.")
    return MarketData(raw=raw, ticker=ticker, source="yfinance", retrieved_at=retrieved_at,
                      notes=notes)


def filter_date_range(data: MarketData, start: str | None, end: str | None) -> MarketData:
    """Keep rows with ``start <= date <= end`` (rows with unparseable dates are kept for validation)."""
    if "date" not in data.raw.columns or (not start and not end):
        return data
    dates = data.raw["date"]
    keep = dates.isna()
    in_range = pd.Series(True, index=dates.index)
    if start:
        in_range &= dates >= pd.Timestamp(start)
    if end:
        in_range &= dates <= pd.Timestamp(end)
    filtered = data.raw.loc[keep | in_range].reset_index(drop=True)
    if len(filtered) < len(data.raw):
        data.notes.append(f"Restricted to {start} .. {end}: kept {len(filtered)} of "
                          f"{len(data.raw)} rows.")
    data.raw = filtered
    return data


def load_market_data(cfg: DataConfig, uploaded_csv: IO | None = None, seed: int = 42) -> MarketData:
    """Dispatch to the configured data source."""
    if cfg.source == "synthetic":
        return load_synthetic(seed=seed)
    if cfg.source == "csv":
        if uploaded_csv is not None:
            data = load_csv(uploaded_csv, ticker=cfg.ticker or "CSV")
        elif cfg.csv_path:
            data = load_csv(cfg.csv_path, ticker=cfg.ticker or Path(cfg.csv_path).stem)
        else:
            raise DataLoadError("data.source is 'csv' but no csv_path was configured or uploaded.")
        return filter_date_range(data, cfg.start, cfg.end)
    if cfg.source == "yfinance":
        return load_yfinance(cfg.ticker, cfg.start, cfg.end, cfg.cache_dir, cfg.use_cache)
    raise DataLoadError(f"Unknown data source '{cfg.source}'.")

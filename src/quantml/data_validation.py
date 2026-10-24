"""Data-quality checks for daily OHLCV data.

Validation never repairs data. It reports *errors* (the experiment must not run), *warnings*
(results may be affected; shown to the user) and *info* messages. Weekends and exchange holidays
are expected, so only unusually long gaps between observations are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantml.config import ValidationConfig
from quantml.data_loader import PRICE_COLUMNS, coerce_numeric

ERROR, WARNING, INFO = "error", "warning", "info"
_MAX_EXAMPLES = 5


@dataclass
class ValidationIssue:
    severity: str
    check: str
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    issues: list[ValidationIssue]
    n_rows: int
    n_valid_rows: int
    start: str | None
    end: str | None

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def status(self) -> str:
        if self.errors:
            return "FAILED"
        return "PASSED WITH WARNINGS" if self.warnings else "PASSED"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "severity": i.severity, "check": i.check, "message": i.message,
            "count": i.count, "examples": ", ".join(i.examples),
        } for i in self.issues], columns=["severity", "check", "message", "count", "examples"])

    def summary(self) -> str:
        lines = [f"Data validation {self.status}: {self.n_valid_rows} valid rows "
                 f"({self.start} to {self.end})."]
        lines += [f"[{i.severity.upper()}] {i.message}" for i in self.issues]
        return "\n".join(lines)


class DataValidationError(ValueError):
    """Raised when data fails validation; ``report`` holds every issue found."""

    def __init__(self, report: ValidationReport):
        self.report = report
        details = "; ".join(i.message for i in report.errors)
        super().__init__(f"Data validation failed: {details}")


def _examples(dates: pd.Series) -> list[str]:
    return [d.strftime("%Y-%m-%d") if isinstance(d, pd.Timestamp) else str(d)
            for d in dates.head(_MAX_EXAMPLES)]


def validate_ohlcv(raw: pd.DataFrame, cfg: ValidationConfig | None = None) -> ValidationReport:
    """Validate a frame produced by :func:`quantml.data_loader.standardize_columns`."""
    cfg = cfg or ValidationConfig()
    issues: list[ValidationIssue] = []

    def add(severity: str, check: str, message: str, count: int = 0,
            examples: list[str] | None = None) -> None:
        issues.append(ValidationIssue(severity, check, message, count, examples or []))

    if raw is None or raw.empty:
        add(ERROR, "empty", "The dataset is empty. Check the ticker, date range or file contents.")
        return ValidationReport(issues, 0, 0, None, None)

    n_rows = len(raw)
    if "date" not in raw.columns:
        add(ERROR, "date_column", "No date column found. Expected a column named Date, Datetime "
                                  "or Timestamp.")
        return ValidationReport(issues, n_rows, 0, None, None)

    missing_cols = [c for c in PRICE_COLUMNS if c not in raw.columns]
    if missing_cols:
        add(ERROR, "required_columns",
            f"Missing required price column(s): {missing_cols}. Found: {list(raw.columns)}.")
        return ValidationReport(issues, n_rows, 0, None, None)

    dates = raw["date"]
    bad_dates = dates.isna()
    if bad_dates.any():
        add(ERROR, "unparseable_dates", f"{int(bad_dates.sum())} row(s) have missing or "
            "unparseable dates. Use ISO dates such as 2024-01-31.", int(bad_dates.sum()))

    duplicated = dates.duplicated(keep=False) & ~bad_dates
    if duplicated.any():
        dup_dates = dates[duplicated].drop_duplicates()
        add(ERROR, "duplicate_dates", f"{len(dup_dates)} date(s) appear more than once. Each "
            "trading day must have exactly one row.", len(dup_dates), _examples(dup_dates))

    valid_dates = dates[~bad_dates]
    if not valid_dates.is_monotonic_increasing:
        add(INFO, "sorting", "Rows are not in chronological order; they will be sorted by date.")

    prices = {c: coerce_numeric(raw[c]) for c in PRICE_COLUMNS}
    for col in PRICE_COLUMNS:
        non_numeric = prices[col].isna() & raw[col].notna() & (raw[col].astype(str).str.strip() != "")
        if non_numeric.any():
            add(ERROR, "non_numeric", f"Column '{col}' has {int(non_numeric.sum())} non-numeric "
                "value(s).", int(non_numeric.sum()), _examples(dates[non_numeric]))

    price_frame = pd.DataFrame(prices)
    all_missing = price_frame.isna().all(axis=1)
    partial_missing = price_frame.isna().any(axis=1) & ~all_missing
    if all_missing.any():
        add(WARNING, "empty_rows", f"{int(all_missing.sum())} row(s) contain no prices at all "
            "(provider placeholders); they will be dropped, not filled.",
            int(all_missing.sum()), _examples(dates[all_missing]))
    if partial_missing.any():
        add(ERROR, "missing_prices", f"{int(partial_missing.sum())} row(s) have some but not all "
            "OHLC prices missing. Missing prices are never filled in automatically.",
            int(partial_missing.sum()), _examples(dates[partial_missing]))

    non_positive = (price_frame <= 0).any(axis=1)
    if non_positive.any():
        add(ERROR, "non_positive_prices", f"{int(non_positive.sum())} row(s) have zero or negative "
            "prices.", int(non_positive.sum()), _examples(dates[non_positive]))

    o, h, lo, c = (price_frame[k] for k in PRICE_COLUMNS)
    tol = cfg.ohlc_tolerance
    bad_ohlc = ((h < pd.concat([o, c, lo], axis=1).max(axis=1) * (1 - tol))
                | (lo > pd.concat([o, c, h], axis=1).min(axis=1) * (1 + tol)))
    bad_ohlc &= price_frame.notna().all(axis=1)
    if bad_ohlc.any():
        add(ERROR, "ohlc_consistency", f"{int(bad_ohlc.sum())} row(s) violate High >= max(Open, "
            f"Close, Low) or Low <= min(Open, Close, High) beyond a {tol:.2%} tolerance.",
            int(bad_ohlc.sum()), _examples(dates[bad_ohlc]))

    if "adj_close" in raw.columns:
        adj = coerce_numeric(raw["adj_close"])
        bad_adj = (adj.isna() | (adj <= 0)) & ~all_missing
        if bad_adj.any():
            add(WARNING, "adj_close", f"Adjusted close is missing or non-positive on "
                f"{int(bad_adj.sum())} row(s); unadjusted prices will be used for the whole series.",
                int(bad_adj.sum()), _examples(dates[bad_adj]))
    else:
        add(WARNING, "adj_close", "No adjusted close column: returns will exclude dividends and "
            "rely on the source having adjusted for splits.")

    if "volume" not in raw.columns:
        add(WARNING, "volume", "No volume column: volume features and charts are disabled.")
    else:
        volume = coerce_numeric(raw["volume"])
        missing_vol = volume.isna() & ~all_missing
        if missing_vol.any():
            add(WARNING, "volume", f"Volume is missing on {int(missing_vol.sum())} row(s).",
                int(missing_vol.sum()), _examples(dates[missing_vol]))
        if (volume < 0).any():
            add(ERROR, "volume", f"Volume is negative on {int((volume < 0).sum())} row(s).",
                int((volume < 0).sum()), _examples(dates[volume < 0]))
        zero_vol = (volume == 0) & ~all_missing
        if zero_vol.any():
            add(WARNING, "volume", f"Volume is zero on {int(zero_vol.sum())} row(s) (common for "
                "indices; suspicious for single stocks).", int(zero_vol.sum()),
                _examples(dates[zero_vol]))

    usable = ~bad_dates & ~all_missing
    n_valid = int(dates[usable].nunique())
    if n_valid < cfg.min_observations:
        add(ERROR, "insufficient_observations", f"Only {n_valid} usable daily observations; at "
            f"least {cfg.min_observations} are required. Choose a longer date range.", n_valid)

    ordered = pd.DataFrame({"date": dates, "close": c})[usable].drop_duplicates("date") \
        .sort_values("date")
    gaps = ordered["date"].diff().dt.days
    long_gaps = gaps > cfg.max_gap_calendar_days
    if long_gaps.any():
        gap_text = [f"{prev:%Y-%m-%d}->{cur:%Y-%m-%d}" for prev, cur in
                    zip(ordered["date"].shift(1)[long_gaps], ordered["date"][long_gaps])]
        add(WARNING, "date_gaps", f"{int(long_gaps.sum())} gap(s) longer than "
            f"{cfg.max_gap_calendar_days} calendar days between observations (weekends and "
            "holidays are normal; longer gaps may indicate missing data). Gaps are not filled.",
            int(long_gaps.sum()), gap_text[:_MAX_EXAMPLES])

    returns = ordered["close"].pct_change(fill_method=None)
    extreme = returns.abs() > cfg.extreme_return_threshold
    if extreme.any():
        add(WARNING, "extreme_returns", f"{int(extreme.sum())} daily close-to-close move(s) "
            f"exceed {cfg.extreme_return_threshold:.0%}; check for unadjusted splits or bad ticks.",
            int(extreme.sum()), _examples(ordered["date"][extreme]))

    start = ordered["date"].min() if len(ordered) else None
    end = ordered["date"].max() if len(ordered) else None
    return ValidationReport(issues, n_rows, n_valid,
                            start.strftime("%Y-%m-%d") if start is not None else None,
                            end.strftime("%Y-%m-%d") if end is not None else None)

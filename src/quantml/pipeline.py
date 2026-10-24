"""End-to-end research pipeline.

Order of operations (each step only sees information it is allowed to see):

1. Load data, validate it, and prepare adjusted prices.
2. Build backward-looking features and forward-looking labels; keep rows with complete features.
3. Chronological train / validation / test split with label purging at each boundary.
4. Tune hyper-parameters and (optionally) the probability threshold on the *validation* period
   using a model fitted on the training period only.
5. Fit the final model, generate probabilities, and backtest all strategies on identical
   windows with identical execution and cost assumptions.
6. Walk-forward evaluation: refit on past data only, predict the next block, roll forward.
7. Diagnostics on fixed strategies: cost sensitivity and block-bootstrap uncertainty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import IO, Any

import numpy as np
import pandas as pd

from quantml.backtester import BacktestResult, CostModel, run_backtests
from quantml.config import ResearchConfig, environment_info
from quantml.data_loader import MarketData, load_market_data, prepare_ohlcv
from quantml.data_validation import DataValidationError, ValidationReport, validate_ohlcv
from quantml.evaluation import block_bootstrap, cost_sensitivity, select_threshold, tune_hyperparameters
from quantml.features import (LookaheadAudit, audit_lookahead, build_features, feature_descriptions,
                              volume_usable)
from quantml.labels import label_lookahead, make_labels
from quantml.metrics import classification_metrics, metrics_table, summarize_backtest
from quantml.models import (MODEL_LABELS, base_params, candidate_params, coefficient_table, fit_model,
                            predict_up_probability)
from quantml.splits import assert_no_overlap, chronological_split, walk_forward_splits
from quantml.strategies import BuyAndHold, MovingAverageCrossover, ProbabilityThresholdStrategy

logger = logging.getLogger(__name__)

SEGMENT_LABELS = {
    "train": "In-sample (training period)",
    "validation": "Validation (used for model choices)",
    "test": "Out-of-sample (untouched test period)",
}


@dataclass
class PreparedData:
    market: MarketData
    report: ValidationReport
    prices: pd.DataFrame
    adjustment: str

    def provenance(self) -> dict[str, Any]:
        return {
            "ticker": self.market.ticker,
            "source": self.market.source_label,
            "is_synthetic": self.market.is_synthetic,
            "retrieved_at": self.market.retrieved_at,
            "from_cache": self.market.from_cache,
            "price_adjustment": self.adjustment,
            "validation_status": self.report.status,
            "rows": len(self.prices),
            "start": self.prices.index[0].strftime("%Y-%m-%d"),
            "end": self.prices.index[-1].strftime("%Y-%m-%d"),
            "notes": list(self.market.notes),
        }


@dataclass
class SegmentResult:
    key: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    backtests: dict[str, BacktestResult]
    metrics: pd.DataFrame
    classification: dict[str, Any]


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    probabilities: pd.Series
    backtests: dict[str, BacktestResult]
    metrics: pd.DataFrame
    classification: dict[str, Any]
    threshold: float
    bootstrap: pd.DataFrame


@dataclass
class ResearchResult:
    config: ResearchConfig
    prices: pd.DataFrame
    features: pd.DataFrame
    feature_descriptions: dict[str, str]
    dataset: pd.DataFrame
    split_table: pd.DataFrame
    model_name: str
    model_params: dict[str, Any]
    tuning_table: pd.DataFrame
    coefficients: pd.DataFrame | None
    probabilities: pd.Series
    threshold: float
    threshold_table: pd.DataFrame
    signals: dict[str, pd.Series]
    moving_averages: pd.DataFrame
    strategy_descriptions: dict[str, str]
    segments: dict[str, SegmentResult]
    walk_forward: WalkForwardResult | None
    cost_sensitivity: pd.DataFrame
    bootstrap: pd.DataFrame
    lookahead_audit: LookaheadAudit
    checks: list[str]
    warnings: list[str]
    run_info: dict[str, Any] = field(default_factory=dict)

    @property
    def benchmark_name(self) -> str:
        return BuyAndHold.name

    @property
    def ml_name(self) -> str:
        return list(self.signals)[-1]


def load_and_prepare(cfg: ResearchConfig, uploaded_csv: IO | None = None) -> PreparedData:
    """Load, validate and prepare prices; raises DataLoadError or DataValidationError."""
    market = load_market_data(cfg.data, uploaded_csv, seed=cfg.seed)
    report = validate_ohlcv(market.raw, cfg.validation)
    if not report.ok:
        raise DataValidationError(report)
    prices, adjustment = prepare_ohlcv(market.raw, market.is_synthetic)
    return PreparedData(market, report, prices, adjustment)


def _previous_date(index: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    return index[index.get_loc(date) - 1]


def _segment_metrics(backtests: dict[str, BacktestResult], cfg: ResearchConfig) -> pd.DataFrame:
    ev = cfg.evaluation
    return metrics_table({name: summarize_backtest(bt.daily, ev.periods_per_year, ev.risk_free_rate,
                                                   ev.min_obs_annualize)
                          for name, bt in backtests.items()})


def run_research(prices: pd.DataFrame, cfg: ResearchConfig,
                 provenance: dict[str, Any] | None = None) -> ResearchResult:
    """Run the complete experiment on prepared, validated prices."""
    ev, st, bt_cfg = cfg.evaluation, cfg.strategy, cfg.backtest
    ppy, rf = ev.periods_per_year, ev.risk_free_rate
    costs = CostModel(bt_cfg.fee_bps, bt_cfg.slippage_bps)
    execution = bt_cfg.execution
    checks: list[str] = []
    warnings: list[str] = []

    # --- Features and labels -------------------------------------------------------------
    use_volume = volume_usable(prices, cfg.features)
    features = build_features(prices, cfg.features, use_volume)
    feature_cols = list(features.columns)
    labels = make_labels(prices, cfg.target, execution)
    dataset = features.dropna().join(labels)
    labeled = dataset[dataset["label"].notna()]
    dates = labeled.index
    X, y = labeled[feature_cols], labeled["label"].astype(int)
    lookahead = label_lookahead(cfg.target)
    logger.info("Dataset: %d eligible rows, %d labelled, %d features", len(dataset), len(labeled),
                len(feature_cols))

    # --- Chronological split with purging ------------------------------------------------
    sp = cfg.split
    split = chronological_split(len(labeled), sp.train, sp.validation, sp.test,
                                purge=lookahead, embargo=sp.embargo)
    assert_no_overlap(dates, labeled["label_end_date"], split.train, split.test)
    if len(split.validation):
        assert_no_overlap(dates, labeled["label_end_date"], split.train, split.validation)
        assert_no_overlap(dates, labeled["label_end_date"], split.validation, split.test)
    checks.append(f"Label purge: {lookahead + sp.embargo} row(s) removed before each split "
                  "boundary; no training label uses prices from a later period.")

    audit = audit_lookahead(prices, cfg.features)
    checks.append(f"Look-ahead audit {'PASSED' if audit.passed else 'FAILED'}: features recomputed "
                  f"on truncated history at {audit.n_checked} dates (max abs diff "
                  f"{audit.max_abs_difference:.2e}).")
    if not audit.passed:
        warnings.append(f"Feature look-ahead audit failed at {audit.offending_dates}.")

    # --- Model selection on validation (model fitted on training only) -------------------
    name = cfg.model.name
    X_tr, y_tr = X.iloc[split.train], y.iloc[split.train]
    X_val, y_val = X.iloc[split.validation], y.iloc[split.validation]
    if cfg.model.tune_on_validation and len(split.validation):
        params, tuning_table = tune_hyperparameters(name, candidate_params(cfg.model), X_tr, y_tr,
                                                    X_val, y_val, cfg.seed)
    else:
        params, tuning_table = base_params(cfg.model), pd.DataFrame()
    model = fit_model(name, params, X_tr, y_tr, cfg.seed)
    probabilities = predict_up_probability(model, dataset[feature_cols])
    checks.append("Preprocessing (standardisation) is inside the model pipeline and fitted on "
                  "training rows only.")

    val_start = dates[split.validation[0]] if len(split.validation) else None
    test_start = dates[split.test[0]]
    first_eval = val_start if val_start is not None else test_start
    windows = {"train": (dates[split.train[0]], _previous_date(prices.index, first_eval))}
    if val_start is not None:
        windows["validation"] = (val_start, _previous_date(prices.index, test_start))
    windows["test"] = (test_start, prices.index[-1])

    threshold, threshold_table = st.ml_threshold, pd.DataFrame()
    if st.threshold_selection == "validation" and "validation" in windows:
        threshold, threshold_table = select_threshold(
            probabilities, prices, *windows["validation"], st.threshold_grid, costs, execution,
            st.below_threshold_position, ppy, rf, default=st.ml_threshold)
        checks.append(f"Threshold {threshold:.2f} selected on the validation period only.")

    if cfg.model.refit_on_train_and_validation and len(split.validation):
        refit_idx = np.concatenate([split.train, split.validation])
        model = fit_model(name, params, X.iloc[refit_idx], y.iloc[refit_idx], cfg.seed)
        probabilities = predict_up_probability(model, dataset[feature_cols])
        warnings.append("Final model was refitted on training + validation data, so validation "
                        "results for the ML strategy are in-sample.")

    # --- Strategies -----------------------------------------------------------------------
    ml_name = f"ML ({MODEL_LABELS[name].replace(' (experimental)', '')})"
    bh = BuyAndHold()
    ma = MovingAverageCrossover(st.ma_short, st.ma_long)
    ml = ProbabilityThresholdStrategy(probabilities, threshold, st.below_threshold_position, ml_name)
    signals = {s.name: s.generate_signals(prices) for s in (bh, ma, ml)}
    descriptions = {s.name: s.describe() for s in (bh, ma, ml)}

    # --- Backtests on identical windows ---------------------------------------------------
    index_sets = {"train": split.train, "validation": split.validation, "test": split.test}
    segments: dict[str, SegmentResult] = {}
    for key, (start, end) in windows.items():
        backtests = run_backtests(prices, signals, costs, execution, start, end,
                                  bt_cfg.initial_capital)
        idx = index_sets[key]
        clf = classification_metrics(y.iloc[idx], probabilities.reindex(dates[idx]), threshold)
        segments[key] = SegmentResult(key, SEGMENT_LABELS[key], start, end, backtests,
                                      _segment_metrics(backtests, cfg), clf)
    if cfg.model.refit_on_train_and_validation and "validation" in segments:
        segments["validation"].label = "Validation (in-sample after refit)"

    split_table = pd.DataFrame([
        {"segment": SEGMENT_LABELS[k], "start": s.strftime("%Y-%m-%d"), "end": e.strftime("%Y-%m-%d"),
         "model_rows": len(index_sets[k]), "backtest_days": len(prices.loc[s:e])}
        for k, (s, e) in windows.items()])

    test_window = windows["test"]
    sensitivity = cost_sensitivity(prices, signals, *test_window, ev.cost_grid_bps, execution, ppy, rf)
    test_returns = pd.DataFrame({k: v.net_returns for k, v in segments["test"].backtests.items()})
    boot_cfg = ev.bootstrap
    bootstrap = (block_bootstrap(test_returns, bh.name, boot_cfg.n_resamples, boot_cfg.block_size,
                                 boot_cfg.confidence, ppy, rf, cfg.seed)
                 if boot_cfg.enabled else pd.DataFrame())

    walk_forward = (_run_walk_forward(prices, cfg, X, y, labeled["label_end_date"], lookahead,
                                      signals, ml_name, costs)
                    if cfg.walk_forward.enabled else None)
    if cfg.walk_forward.enabled and walk_forward is None:
        warnings.append(f"Walk-forward evaluation skipped: initial_train_size "
                        f"({cfg.walk_forward.initial_train_size}) is not smaller than the "
                        f"{len(labeled)} eligible observations.")

    if not np.isclose(threshold, 0.5) and st.threshold_selection == "fixed":
        warnings.append(f"A fixed threshold of {threshold:.2f} is used; make sure it was not chosen "
                        "by looking at test-period results.")
    if (segments["test"].classification.get("n_count", 0) or 0) < 250:
        warnings.append("The test period has fewer than ~1 year of observations; annualised "
                        "metrics are noisy.")

    run_info = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_info(),
        "data": provenance or {},
        "n_features": len(feature_cols),
        "volume_features": use_volume,
        "label_lookahead_rows": lookahead,
        "model": {"name": name, "params": params},
        "threshold": threshold,
    }
    return ResearchResult(
        config=cfg, prices=prices, features=features,
        feature_descriptions=feature_descriptions(cfg.features, use_volume), dataset=dataset,
        split_table=split_table, model_name=name, model_params=params, tuning_table=tuning_table,
        coefficients=coefficient_table(model, feature_cols), probabilities=probabilities,
        threshold=threshold, threshold_table=threshold_table, signals=signals,
        moving_averages=ma.moving_averages(prices), strategy_descriptions=descriptions,
        segments=segments, walk_forward=walk_forward, cost_sensitivity=sensitivity,
        bootstrap=bootstrap, lookahead_audit=audit, checks=checks, warnings=warnings,
        run_info=run_info)


def _run_walk_forward(prices: pd.DataFrame, cfg: ResearchConfig, X: pd.DataFrame, y: pd.Series,
                      label_end: pd.Series, lookahead: int, signals: dict[str, pd.Series],
                      ml_name: str, costs: CostModel) -> WalkForwardResult | None:
    """Refit on past data only for each fold and stitch the out-of-sample predictions together.

    Hyper-parameters and the threshold are the configured fixed values. Settings tuned on the
    holdout validation period are deliberately *not* reused, because that period lies after
    the early walk-forward test windows and would leak information into them.
    """
    wf, ev, st = cfg.walk_forward, cfg.evaluation, cfg.strategy
    params = base_params(cfg.model)
    dates = X.index
    try:
        folds = walk_forward_splits(len(X), wf.initial_train_size, wf.test_size, lookahead,
                                    cfg.split.embargo, wf.window, wf.max_train_size)
    except ValueError as exc:
        logger.warning("Walk-forward skipped: %s", exc)
        return None

    parts, rows = [], []
    for fold in folds:
        assert_no_overlap(dates, label_end, fold.train, fold.test)
        model = fit_model(cfg.model.name, params, X.iloc[fold.train], y.iloc[fold.train], cfg.seed)
        prob = predict_up_probability(model, X.iloc[fold.test])
        parts.append(prob)
        clf = classification_metrics(y.iloc[fold.test], prob, st.ml_threshold)
        rows.append({"fold": fold.fold + 1,
                     "train_start": dates[fold.train[0]], "train_end": dates[fold.train[-1]],
                     "test_start": dates[fold.test[0]], "test_end": dates[fold.test[-1]],
                     "n_train": len(fold.train), "n_test": len(fold.test),
                     "accuracy": clf["accuracy"], "positive_rate": clf["positive_rate"],
                     "roc_auc": clf["roc_auc"]})
    oos = pd.concat(parts)
    ml = ProbabilityThresholdStrategy(oos, st.ml_threshold, st.below_threshold_position, ml_name)
    wf_signals = {**{k: v for k, v in signals.items() if k != ml_name},
                  ml_name: ml.generate_signals(prices)}
    start, end = oos.index[0], oos.index[-1]
    backtests = run_backtests(prices, wf_signals, costs, cfg.backtest.execution, start, end,
                              cfg.backtest.initial_capital)

    fold_table = pd.DataFrame(rows)
    for name, bt in backtests.items():
        fold_table[f"return_net::{name}"] = [
            float((1 + bt.net_returns.loc[r.test_start:r.test_end]).prod() - 1)
            for r in fold_table.itertuples()]
    y_oos = y.reindex(oos.index)
    returns = pd.DataFrame({k: v.net_returns for k, v in backtests.items()})
    boot = ev.bootstrap
    bootstrap = (block_bootstrap(returns, BuyAndHold.name, boot.n_resamples, boot.block_size,
                                 boot.confidence, ev.periods_per_year, ev.risk_free_rate, cfg.seed)
                 if boot.enabled else pd.DataFrame())
    return WalkForwardResult(fold_table, oos, backtests, _segment_metrics(backtests, cfg),
                             classification_metrics(y_oos, oos, st.ml_threshold), st.ml_threshold,
                             bootstrap)

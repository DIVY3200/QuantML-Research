"""Export research results as tidy tables and save reproducible run folders."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quantml.backtester import BacktestResult
from quantml.config import resolve_path
from quantml.metrics import format_metrics_table
from quantml.pipeline import ResearchResult

_DAILY_COLUMNS = ["signal", "position", "trade", "turnover", "gross_return", "fee_cost",
                  "slippage_cost", "total_cost", "net_return", "cumulative_cost", "equity_gross",
                  "equity_net", "drawdown"]


def daily_results_frame(backtests: dict[str, BacktestResult], segment: str = "") -> pd.DataFrame:
    """Long-format daily results: one row per (date, strategy). Returns are decimals."""
    frames = []
    for name, bt in backtests.items():
        frame = bt.daily[_DAILY_COLUMNS].copy()
        frame.insert(0, "strategy", name)
        if segment:
            frame.insert(0, "segment", segment)
        frames.append(frame)
    return pd.concat(frames).reset_index()


def trades_frame(backtests: dict[str, BacktestResult]) -> pd.DataFrame:
    frames = [bt.trades.assign(strategy=name) for name, bt in backtests.items()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def predictions_frame(result: ResearchResult) -> pd.DataFrame:
    """Holdout-model probabilities with segment labels, targets and resulting ML signals."""
    data = result.dataset[["fwd_return", "label", "label_end_date"]].copy()
    data["prob_up"] = result.probabilities
    data["ml_signal"] = result.signals[result.ml_name].reindex(data.index)
    data["segment"] = ""
    for seg in result.segments.values():
        data.loc[seg.start:seg.end, "segment"] = seg.label
    return data.reset_index()


def all_metrics_frame(result: ResearchResult) -> pd.DataFrame:
    """Raw (decimal) metrics for every segment and strategy, including walk-forward."""
    frames = []
    for seg in result.segments.values():
        frames.append(seg.metrics.T.assign(segment=seg.label))
    if result.walk_forward is not None:
        frames.append(result.walk_forward.metrics.T.assign(segment="Walk-forward out-of-sample"))
    out = pd.concat(frames)
    out.index.name = "strategy"
    return out.reset_index().set_index(["segment", "strategy"]).reset_index()


def classification_frame(result: ResearchResult) -> pd.DataFrame:
    rows = {seg.label: seg.classification for seg in result.segments.values()}
    if result.walk_forward is not None:
        rows["Walk-forward out-of-sample"] = result.walk_forward.classification
    frame = pd.DataFrame(rows).T
    return frame.drop(columns=["confusion_matrix"], errors="ignore")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def save_run(result: ResearchResult, out_dir: str | Path | None = None,
             compact: bool = False) -> Path:
    """Write configuration, environment, data provenance and all result tables to a folder.

    ``compact=True`` skips the large day-by-day files (daily results and predictions), keeping
    only summary tables - suitable for committing to version control.
    """
    stamp = result.run_info.get("run_timestamp_utc", "")[:19].replace(":", "").replace("-", "")
    ticker = re.sub(r"[^A-Za-z0-9._-]", "_", str(result.run_info.get("data", {}).get("ticker", "run")))
    folder = Path(out_dir) if out_dir else resolve_path(result.config.output.reports_dir) / \
        f"run_{stamp}_{ticker}"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "config.yaml").write_text(yaml.safe_dump(result.config.to_dict(), sort_keys=False))
    info = {**result.run_info, "checks": result.checks, "warnings": result.warnings,
            "lookahead_audit": result.lookahead_audit.__dict__}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2, default=_json_default))

    all_metrics_frame(result).to_csv(folder / "metrics.csv", index=False)
    classification_frame(result).to_csv(folder / "classification_metrics.csv")
    result.split_table.to_csv(folder / "splits.csv", index=False)
    if not compact:
        predictions_frame(result).to_csv(folder / "predictions.csv", index=False)
        daily = pd.concat([daily_results_frame(seg.backtests, seg.key)
                           for seg in result.segments.values()])
        daily.to_csv(folder / "daily_results.csv", index=False)
    trades_frame(result.segments["test"].backtests).to_csv(folder / "trades_test.csv", index=False)
    result.cost_sensitivity.to_csv(folder / "cost_sensitivity.csv", index=False)
    if not result.bootstrap.empty:
        result.bootstrap.to_csv(folder / "bootstrap_test.csv", index=False)
    if not result.tuning_table.empty:
        result.tuning_table.to_csv(folder / "hyperparameter_tuning.csv", index=False)
    if not result.threshold_table.empty:
        result.threshold_table.to_csv(folder / "threshold_selection.csv", index=False)
    if result.walk_forward is not None:
        wf = result.walk_forward
        wf.folds.to_csv(folder / "walk_forward_folds.csv", index=False)
        if not compact:
            daily_results_frame(wf.backtests, "walk_forward").to_csv(
                folder / "walk_forward_daily.csv", index=False)
        if not wf.bootstrap.empty:
            wf.bootstrap.to_csv(folder / "bootstrap_walk_forward.csv", index=False)
    (folder / "summary.md").write_text(render_summary(result), encoding="utf-8")
    return folder


def render_summary(result: ResearchResult) -> str:
    """Markdown summary generated from computed outputs only."""
    data = result.run_info.get("data", {})
    lines = [
        f"# Research run: {data.get('ticker', '?')}",
        "",
        f"- Data: {data.get('source')} ({data.get('start')} to {data.get('end')}, "
        f"{data.get('rows')} rows); retrieved {data.get('retrieved_at')}",
        f"- Price adjustment: {data.get('price_adjustment')}",
        f"- Execution: {result.config.backtest.execution}; costs: "
        f"{result.config.backtest.fee_bps} bps fees + {result.config.backtest.slippage_bps} bps "
        "slippage per unit turnover",
        f"- Model: {result.model_name} {result.model_params}; threshold {result.threshold:.2f}",
        "",
    ]
    if data.get("is_synthetic"):
        lines += ["> **SYNTHETIC DATA** - these results do not describe any real market.", ""]
    for seg in result.segments.values():
        lines += [f"## {seg.label} ({seg.start:%Y-%m-%d} to {seg.end:%Y-%m-%d})", "", "```",
                  format_metrics_table(seg.metrics).to_string(), "```", ""]
    if result.walk_forward is not None:
        wf = result.walk_forward
        lines += [f"## Walk-forward out-of-sample ({len(wf.folds)} folds)", "", "```",
                  format_metrics_table(wf.metrics).to_string(), "```", ""]
    lines += ["## Checks", *[f"- {c}" for c in result.checks], "",
              "## Warnings", *([f"- {w}" for w in result.warnings] or ["- none"]), ""]
    return "\n".join(lines)

"""Run the identical default experiment on several instruments and compare the results.

    python scripts/run_multi_ticker.py
    python scripts/run_multi_ticker.py --tickers SPY AAPL RELIANCE.NS --out results

For every ticker the same configuration (dates, features, model, costs, splits, walk-forward)
is used - nothing is tuned per instrument. Compact per-ticker reports and a summary table are
written to ``results/``. Tickers whose data fail validation are reported, not silently dropped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantml.config import load_config  # noqa: E402
from quantml.data_loader import DataLoadError  # noqa: E402
from quantml.data_validation import DataValidationError  # noqa: E402
from quantml.pipeline import ResearchResult, load_and_prepare, run_research  # noqa: E402
from quantml.reporting import save_run  # noqa: E402

DEFAULT_TICKERS = ["SPY", "AAPL", "MSFT", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
BH, MA = "Buy & Hold", "MA Crossover (20/50)"


def summarize(ticker: str, result: ResearchResult, n_warnings: int) -> dict:
    """One row of headline numbers, all read from the computed result."""
    data = result.run_info["data"]
    ml = result.ml_name
    test = result.segments["test"].metrics
    row = {
        "ticker": ticker,
        "market": "India (NSE)" if ticker.endswith(".NS") else "US",
        "rows": data["rows"], "start": data["start"], "end": data["end"],
        "validation_status": data["validation_status"], "validation_warnings": n_warnings,
        "test_start": test.loc["start", BH], "test_end": test.loc["end", BH],
        "test_sharpe_bh": test.loc["sharpe_net", BH],
        "test_sharpe_ma": test.loc["sharpe_net", MA],
        "test_sharpe_ml": test.loc["sharpe_net", ml],
        "test_ml_exposure": test.loc["average_exposure", ml],
    }
    wf = result.walk_forward
    if wf is not None:
        m = wf.metrics
        excess = wf.folds[f"return_net::{ml}"] - wf.folds[f"return_net::{BH}"]
        boot = wf.bootstrap.set_index("strategy") if not wf.bootstrap.empty else None
        row.update({
            "wf_folds": len(wf.folds), "wf_start": m.loc["start", BH], "wf_end": m.loc["end", BH],
            "wf_return_bh": m.loc["cumulative_return_net", BH],
            "wf_return_ma": m.loc["cumulative_return_net", MA],
            "wf_return_ml": m.loc["cumulative_return_net", ml],
            "wf_sharpe_bh": m.loc["sharpe_net", BH],
            "wf_sharpe_ma": m.loc["sharpe_net", MA],
            "wf_sharpe_ml": m.loc["sharpe_net", ml],
            "wf_sharpe_ml_gross": m.loc["sharpe_gross", ml],
            "wf_ml_position_changes": m.loc["position_changes_count", ml],
            "wf_ml_total_costs": m.loc["total_costs", ml],
            "wf_accuracy": wf.classification["accuracy"],
            "wf_majority_baseline": wf.classification["majority_baseline_accuracy"],
            "wf_roc_auc": wf.classification["roc_auc"],
            "wf_folds_ml_beats_bh": int((excess > 0).sum()),
            "wf_sharpe_diff_ci_low": boot.loc[ml, "diff_ci_low"] if boot is not None else float("nan"),
            "wf_sharpe_diff_ci_high": boot.loc[ml, "diff_ci_high"] if boot is not None else float("nan"),
        })
    return row


def markdown_table(summary: pd.DataFrame) -> str:
    header = ("| Ticker | Walk-forward period | Folds | Net Sharpe: Buy & Hold | MA 20/50 | ML | "
              "ML gross Sharpe | ML position changes | ML accuracy (baseline) | "
              "ML - B&H Sharpe, 95% CI | Folds ML > B&H |")
    lines = [header, "|" + "---|" * 11]
    for r in summary.itertuples():
        lines.append(
            f"| {r.ticker} | {r.wf_start} to {r.wf_end} | {r.wf_folds} | {r.wf_sharpe_bh:.2f} | "
            f"{r.wf_sharpe_ma:.2f} | {r.wf_sharpe_ml:.2f} | {r.wf_sharpe_ml_gross:.2f} | "
            f"{int(r.wf_ml_position_changes)} | {r.wf_accuracy:.1%} ({r.wf_majority_baseline:.1%}) | "
            f"[{r.wf_sharpe_diff_ci_low:+.2f}, {r.wf_sharpe_diff_ci_high:+.2f}] | "
            f"{r.wf_folds_ml_beats_bh} / {r.wf_folds} |")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--config", help="YAML config (default: config/default_config.yaml)")
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()
    out = Path(args.out)

    rows, failures = [], []
    for ticker in args.tickers:
        print(f"Running {ticker} ...", flush=True)
        try:
            cfg = load_config(args.config, {"data": {"ticker": ticker}})
            prepared = load_and_prepare(cfg)
            result = run_research(prepared.prices, cfg, prepared.provenance())
        except (DataLoadError, DataValidationError, ValueError) as exc:
            failures.append((ticker, str(exc)))
            print(f"  skipped: {exc}")
            continue
        save_run(result, out / ticker.replace(".", "_"), compact=True)
        rows.append(summarize(ticker, result, len(prepared.report.warnings)))

    if not rows:
        print("No ticker completed.")
        return 1
    summary = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    failure_text = "\n".join(f"- {t}: {msg}" for t, msg in failures) or "- none"
    (out / "summary_table.md").write_text(
        f"Generated {stamp} by `scripts/run_multi_ticker.py` with the default configuration.\n\n"
        f"{markdown_table(summary)}\n\nTickers that could not be run:\n{failure_text}\n",
        encoding="utf-8")
    print(summary[["ticker", "wf_sharpe_bh", "wf_sharpe_ma", "wf_sharpe_ml", "wf_sharpe_ml_gross",
                   "wf_accuracy", "wf_majority_baseline"]].round(3).to_string(index=False))
    print(f"\nWrote {out / 'summary.csv'} and {out / 'summary_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

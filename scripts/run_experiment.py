"""Run a complete, reproducible research experiment from the command line.

Examples
--------
    python scripts/run_experiment.py                         # default config (SPY via yfinance)
    python scripts/run_experiment.py --source synthetic      # offline demo on SYNTHETIC data
    python scripts/run_experiment.py --ticker RELIANCE.NS --start 2012-01-01 --end 2024-12-31
    python scripts/run_experiment.py --source csv --csv data/sample/SYNTHETIC_sample_ohlcv.csv

Each run writes the resolved config, package versions, data provenance and all result tables
to ``reports/run_<timestamp>_<ticker>/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantml.config import ConfigError, load_config  # noqa: E402
from quantml.data_loader import DataLoadError  # noqa: E402
from quantml.data_validation import DataValidationError  # noqa: E402
from quantml.metrics import format_metrics_table  # noqa: E402
from quantml.pipeline import load_and_prepare, run_research  # noqa: E402
from quantml.reporting import save_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML config (default: config/default_config.yaml)")
    p.add_argument("--source", choices=["yfinance", "csv", "synthetic"])
    p.add_argument("--ticker")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--csv", help="CSV path (implies --source csv)")
    p.add_argument("--model", choices=["logistic_regression", "random_forest"])
    p.add_argument("--execution", choices=["next_open", "next_close"])
    p.add_argument("--fee-bps", type=float)
    p.add_argument("--slippage-bps", type=float)
    p.add_argument("--threshold-selection", choices=["fixed", "validation"])
    p.add_argument("--out", help="Output folder (default: reports/run_<timestamp>_<ticker>)")
    return p.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    data = {k: v for k, v in {"source": "csv" if args.csv else args.source, "ticker": args.ticker,
                              "start": args.start, "end": args.end, "csv_path": args.csv}.items()
            if v is not None}
    overrides: dict = {"data": data} if data else {}
    if args.model:
        overrides["model"] = {"name": args.model}
    backtest = {k: v for k, v in {"execution": args.execution, "fee_bps": args.fee_bps,
                                  "slippage_bps": args.slippage_bps}.items() if v is not None}
    if backtest:
        overrides["backtest"] = backtest
    if args.threshold_selection:
        overrides["strategy"] = {"threshold_selection": args.threshold_selection}
    return overrides


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        cfg = load_config(args.config, build_overrides(args))
        prepared = load_and_prepare(cfg)
    except (ConfigError, DataLoadError) as exc:
        logging.error("%s", exc)
        return 2
    except DataValidationError as exc:
        logging.error("%s", exc.report.summary())
        return 2

    print(prepared.report.summary())
    if prepared.market.is_synthetic:
        print("\n*** SYNTHETIC DATA: results below do not describe any real market. ***\n")
    result = run_research(prepared.prices, cfg, prepared.provenance())

    for seg in result.segments.values():
        print(f"\n=== {seg.label}: {seg.start:%Y-%m-%d} to {seg.end:%Y-%m-%d} ===")
        print(format_metrics_table(seg.metrics).to_string())
    if result.walk_forward is not None:
        wf = result.walk_forward
        print(f"\n=== Walk-forward out-of-sample ({len(wf.folds)} folds) ===")
        print(format_metrics_table(wf.metrics).to_string())
    print("\nChecks:\n  " + "\n  ".join(result.checks))
    if result.warnings:
        print("Warnings:\n  " + "\n  ".join(result.warnings))
    folder = save_run(result, args.out)
    print(f"\nResults saved to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Multi-instrument results

Compact outputs of `python scripts/run_multi_ticker.py`: the **same default configuration**
(`config/default_config.yaml`: 2010-01-01 to 2024-12-31, next-open execution, 1 bp fees +
2 bp slippage, logistic regression, 60/20/20 split, 63-day walk-forward folds) applied to six
instruments. Nothing is tuned per instrument.

- `summary_table.md` / `summary.csv`: one row per instrument (walk-forward and holdout test).
- `<TICKER>/`: per-instrument `config.yaml`, `run_info.json` (package versions, data source,
  retrieval time, integrity checks), metric tables, walk-forward folds, cost sensitivity,
  bootstrap intervals and the test-period trade log. Day-by-day files and raw prices are not
  committed (size and data-provider terms); re-run the script to regenerate them locally.

## Data notes

- Yahoo Finance data retrieved on 2026-10-20 (SPY) and 2026-10-24 (others); see each
  `run_info.json`. Re-downloading may change the numbers.
- NSE tickers pass validation **with warnings**: two zero-volume placeholder rows on market
  holidays (2014-04-24, 2014-10-15) and one holiday gap (2014-10-01 to 2014-10-07). These rows
  produce flat days and are reported, not removed.
- **NIFTYBEES.NS was excluded**: validation flagged two prices at one-tenth scale on
  2019-12-19 and 2019-12-20 (about ₹13 between closes of about ₹129 and ₹131), consistent with a
  badly applied unit split in the provider data. Backtesting through those ticks would create
  artificial ±90% moves.

## Reading the table

Six instruments from two markets are still a small sample, the period is a single
(mostly rising) regime, and the default settings were fixed before these runs but chosen with
general knowledge of markets. Treat the results as evidence about this configuration, not as a
general statement about machine learning in trading.

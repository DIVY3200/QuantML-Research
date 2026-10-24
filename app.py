"""Streamlit dashboard for the QuantML Research Platform.

Run with:  streamlit run app.py

Every number and chart shown here is computed by the ``quantml`` package from the selected
data and settings at run time; nothing is hard-coded.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from quantml import eda, visualization as viz  # noqa: E402
from quantml.backtester import BacktestResult  # noqa: E402
from quantml.config import ConfigError, load_config  # noqa: E402
from quantml.data_loader import DataLoadError  # noqa: E402
from quantml.data_validation import DataValidationError  # noqa: E402
from quantml.metrics import format_metrics_table  # noqa: E402
from quantml.models import MODEL_LABELS  # noqa: E402
from quantml.pipeline import PreparedData, ResearchResult, load_and_prepare, run_research  # noqa: E402
from quantml.reporting import (all_metrics_frame, classification_frame, daily_results_frame,  # noqa: E402
                               predictions_frame, save_run, trades_frame)

st.set_page_config(page_title="QuantML Research Platform", page_icon="📈", layout="wide")

SOURCES = {"Yahoo Finance (yfinance)": "yfinance", "Upload CSV": "csv",
           "Synthetic demo data": "synthetic"}
EXECUTION_LABELS = {"next_open": "Next open (signal at close t, fill at open t+1)",
                    "next_close": "Next close (signal at close t, fill at close t+1)"}
PERIODS = {"Out-of-sample test": "test", "Validation": "validation",
           "In-sample (training)": "train", "Walk-forward out-of-sample": "walk_forward"}

LIMITATIONS = [
    "Historical performance does not guarantee future results.",
    "Free data (Yahoo Finance) can be incomplete, revised, or adjusted differently over time; "
    "re-downloading may change results. Check the provider's licensing terms before reuse.",
    "Single-instrument backtests ignore survivorship bias in instrument choice and cannot "
    "establish general robustness.",
    "Every additional configuration you try is another test: repeated experimentation creates "
    "multiple-testing and selection bias, and the 'best' run is optimistically biased.",
    "Execution is simplified: fills at the official open/close with a fixed basis-point cost; "
    "real spreads, market impact, auctions, taxes and borrowing constraints differ by broker, "
    "exchange and instrument.",
    "Cash earns 0% in the backtest; with a non-zero risk-free rate the Sharpe ratio penalises "
    "time spent in cash.",
    "This project is for research and education only: it is not financial advice and not an "
    "automated trading system.",
]


# --- Helpers ---------------------------------------------------------------------------------

def theme_mode() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:
        return "light"


def show(fig) -> None:
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


def csv_bytes(frame: pd.DataFrame, index: bool = False) -> bytes:
    return frame.to_csv(index=index).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=16)
def cached_prepare(data_json: str, validation_json: str, seed: int,
                   upload: bytes | None) -> PreparedData:
    """Load + validate + prepare data; cached on the data-relevant settings only."""
    cfg = load_config(overrides={"data": json.loads(data_json),
                                 "validation": json.loads(validation_json), "seed": seed})
    return load_and_prepare(cfg, io.BytesIO(upload) if upload else None)


def sidebar() -> tuple[dict, bytes | None, list[str], bool]:
    base = load_config()
    st.sidebar.title("QuantML Research")
    source_label = st.sidebar.radio("Data source", list(SOURCES), index=0,
                                    help="If a download fails, upload a CSV or use synthetic data.")
    source = SOURCES[source_label]
    upload = None
    if source == "csv":
        file = st.sidebar.file_uploader("OHLCV CSV", type=["csv"],
                                        help="Columns: Date, Open, High, Low, Close, optional "
                                             "Adj Close and Volume (names are matched flexibly).")
        upload = file.getvalue() if file is not None else None
    elif source == "synthetic":
        st.sidebar.warning("Synthetic data is generated noise, not real market prices.")

    with st.sidebar.form("settings"):
        ticker = st.text_input("Ticker / label", base.data.ticker if source != "synthetic" else
                               "SYNTHETIC", disabled=source == "synthetic",
                               help="e.g. SPY, AAPL, MSFT, RELIANCE.NS, HDFCBANK.NS. Provider "
                                    "coverage and symbol conventions vary by market.")
        c1, c2 = st.columns(2)
        start = c1.date_input("Start", date.fromisoformat(base.data.start),
                              min_value=date(1990, 1, 1), disabled=source == "synthetic")
        end = c2.date_input("End", date.fromisoformat(base.data.end),
                            min_value=date(1990, 1, 2), disabled=source == "synthetic")

        with st.expander("Strategies", expanded=True):
            all_names = ["Buy & Hold", "MA Crossover", "ML"]
            shown = st.multiselect("Show strategies", all_names, default=all_names)
            c1, c2 = st.columns(2)
            ma_short = c1.number_input("MA short", 2, 250, base.strategy.ma_short)
            ma_long = c2.number_input("MA long", 3, 400, base.strategy.ma_long)

        with st.expander("Execution & costs", expanded=True):
            execution = st.selectbox("Execution convention", list(EXECUTION_LABELS),
                                     format_func=EXECUTION_LABELS.get)
            c1, c2 = st.columns(2)
            fee = c1.number_input("Fees (bps)", 0.0, 200.0, base.backtest.fee_bps, 0.5,
                                  help="Per unit of turnover; 1 bp = 0.01%.")
            slip = c2.number_input("Slippage (bps)", 0.0, 200.0, base.backtest.slippage_bps, 0.5)
            capital = st.number_input("Initial capital", 100.0, 1e9, base.backtest.initial_capital,
                                      1000.0)

        with st.expander("Model & splits"):
            model = st.selectbox("Model", list(MODEL_LABELS), format_func=MODEL_LABELS.get)
            tune = st.checkbox("Tune regularisation on validation", base.model.tune_on_validation)
            train_pct = st.slider("Training %", 30, 80, int(base.split.train * 100), 5)
            val_pct = st.slider("Validation %", 5, 40, int(base.split.validation * 100), 5)
            st.caption(f"Test %: {100 - train_pct - val_pct}")
            threshold = st.slider("Probability threshold", 0.40, 0.70, base.strategy.ml_threshold, 0.01)
            thr_mode = st.radio("Threshold", ["fixed", "validation"], horizontal=True,
                                format_func={"fixed": "Fixed", "validation": "Select on validation"}.get)

        with st.expander("Walk-forward & evaluation"):
            wf_on = st.checkbox("Walk-forward evaluation", base.walk_forward.enabled)
            c1, c2 = st.columns(2)
            wf_train = c1.number_input("Initial train rows", 100, 5000, base.walk_forward.initial_train_size, 21)
            wf_test = c2.number_input("Test rows / fold", 5, 504, base.walk_forward.test_size, 21)
            wf_window = st.radio("Window", ["expanding", "rolling"], horizontal=True)
            rf = st.number_input("Risk-free rate (annual %)", -5.0, 20.0,
                                 base.evaluation.risk_free_rate * 100, 0.25)
            boot = st.checkbox("Block-bootstrap Sharpe intervals", base.evaluation.bootstrap.enabled)

        submitted = st.form_submit_button("Run research", type="primary", width="stretch")

    overrides = {
        "data": {"source": source, "ticker": ticker.strip() or "CSV",
                 "start": start.isoformat(), "end": end.isoformat()},
        "strategy": {"ma_short": int(ma_short), "ma_long": int(ma_long), "ml_threshold": float(threshold),
                     "threshold_selection": thr_mode},
        "backtest": {"execution": execution, "fee_bps": float(fee), "slippage_bps": float(slip),
                     "initial_capital": float(capital)},
        "split": {"train": train_pct / 100, "validation": val_pct / 100,
                  "test": (100 - train_pct - val_pct) / 100},
        "model": {"name": model, "tune_on_validation": tune},
        "walk_forward": {"enabled": wf_on, "initial_train_size": int(wf_train),
                         "test_size": int(wf_test), "window": wf_window},
        "evaluation": {"risk_free_rate": rf / 100, "bootstrap": {"enabled": boot}},
    }
    return overrides, upload, shown, submitted


def run(overrides: dict, upload: bytes | None) -> None:
    try:
        cfg = load_config(overrides=overrides)
    except ConfigError as exc:
        st.error(f"Invalid settings: {exc}")
        return
    if cfg.data.source == "csv" and upload is None:
        st.error("Upload a CSV file in the sidebar first.")
        return
    try:
        with st.spinner("Loading and validating data..."):
            prepared = cached_prepare(json.dumps(cfg.to_dict()["data"]),
                                      json.dumps(cfg.to_dict()["validation"]), cfg.seed, upload)
    except DataLoadError as exc:
        st.error(str(exc))
        st.info("You can switch the data source to **Upload CSV** (any daily OHLCV export) or "
                "**Synthetic demo data** to keep working offline.")
        return
    except DataValidationError as exc:
        st.error("The data failed validation, so no experiment was run.")
        st.dataframe(exc.report.to_frame(), hide_index=True)
        return
    try:
        with st.spinner("Running features, model, backtests and walk-forward evaluation..."):
            result = run_research(prepared.prices, cfg, prepared.provenance())
    except (ValueError, AssertionError) as exc:
        st.error(f"The experiment could not run with these settings: {exc}")
        return
    st.session_state["prepared"] = prepared
    st.session_state["result"] = result


def note_identical_positions(backtests: dict[str, BacktestResult], names: list[str]) -> None:
    """Explain coinciding curves (e.g. a model that stayed long every day) instead of hiding them."""
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if backtests[a].positions.equals(backtests[b].positions):
                st.info(f"**{b}** held exactly the same position as **{a}** on every day of this "
                        "period, so their curves coincide (one line is drawn on top of the other).")


def filter_names(names: list[str], shown: list[str]) -> list[str]:
    keep = []
    for name in names:
        if (name == "Buy & Hold" and "Buy & Hold" in shown) or \
           (name.startswith("MA Crossover") and "MA Crossover" in shown) or \
           (name.startswith("ML") and "ML" in shown):
            keep.append(name)
    return keep


def period_data(result: ResearchResult, key: str) -> tuple[dict[str, BacktestResult], pd.DataFrame,
                                                          dict, pd.Series, str]:
    if key == "walk_forward":
        wf = result.walk_forward
        return wf.backtests, wf.metrics, wf.classification, wf.probabilities, "Walk-forward out-of-sample"
    seg = result.segments[key]
    probs = result.probabilities.loc[seg.start:seg.end]
    return seg.backtests, seg.metrics, seg.classification, probs, seg.label


# --- Sections --------------------------------------------------------------------------------

def overview(result: ResearchResult, prepared: PreparedData) -> None:
    data = result.run_info["data"]
    c = st.columns(4)
    c[0].metric("Instrument", data["ticker"])
    c[1].metric("Observations", f"{data['rows']:,}")
    c[2].metric("Date range", f"{data['start'][:4]} – {data['end'][:4]}",
                help=f"{data['start']} to {data['end']}")
    c[3].metric("Data quality", prepared.report.status)
    st.caption(f"Source: {data['source']} · retrieved {data['retrieved_at']}"
               f"{' (local cache)' if data['from_cache'] else ''}")
    issues = prepared.report.to_frame()
    if not issues.empty:
        with st.expander(f"Data-quality messages ({len(issues)})", expanded=bool(prepared.report.warnings)):
            st.dataframe(issues, hide_index=True)

    st.subheader("Key assumptions")
    cfg = result.config
    st.markdown(
        f"- **Execution:** {EXECUTION_LABELS[cfg.backtest.execution]}. Positions earn returns only "
        "after execution; every window starts in cash and the final day's signal is never executed.\n"
        f"- **Costs:** {cfg.backtest.fee_bps:g} bps fees + {cfg.backtest.slippage_bps:g} bps slippage "
        "per unit of turnover (entry and exit each = 1.0 turnover), identical for every strategy.\n"
        f"- **Prices:** {data['price_adjustment']}\n"
        f"- **Model:** {MODEL_LABELS[result.model_name]} `{result.model_params}`; long when "
        f"P(up) ≥ {result.threshold:.2f} ({cfg.strategy.threshold_selection} threshold).\n"
        f"- **Target:** sign of the {cfg.target.horizon}-period forward return ({cfg.target.basis} basis); "
        "features use information up to the close of the signal date only.")
    for w in result.warnings:
        st.warning(w)

    st.subheader("Out-of-sample headline (test period)")
    test = result.segments["test"]
    st.caption(f"{test.start:%Y-%m-%d} to {test.end:%Y-%m-%d}. All strategies use identical dates, "
               "execution and costs. Compare against the walk-forward results before drawing conclusions.")
    note_identical_positions(test.backtests, list(test.backtests))
    cols = st.columns(len(test.metrics.columns))
    for col, name in zip(cols, test.metrics.columns):
        m = test.metrics[name]
        col.metric(name, f"Sharpe {m['sharpe_net']:.2f}" if pd.notna(m["sharpe_net"]) else "Sharpe n/a",
                   help="Annualised net Sharpe ratio")
        changes = int(m["position_changes_count"])
        col.caption(f"Net return {m['cumulative_return_net']:.1%} · max DD {m['max_drawdown']:.1%} · "
                    f"{changes} position change{'' if changes == 1 else 's'}")


def market_data(result: ResearchResult, theme: viz.Theme) -> None:
    prices = result.prices
    ppy = result.config.evaluation.periods_per_year
    st.markdown("Price **levels** trend and are not comparable across time; **returns** (relative "
                "changes) are what a strategy earns and are the basis for all statistics below.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Return summary** (decimal returns)")
        st.dataframe(eda.return_summary(prices, ppy).style.format("{:.6g}"))
    with c2:
        st.markdown("**Statistical tests** (full sample, descriptive)")
        st.dataframe(eda.statistical_tests(prices).style.format({"statistic": "{:.2f}",
                                                                 "p_value": "{:.4f}"}),
                     hide_index=True)
        st.caption("A small Ljung-Box p-value indicates serial correlation; it does not imply a "
                   "tradable edge after costs.")
    show(viz.price_chart(prices, theme))
    if prices["volume"].notna().any():
        show(viz.volume_chart(prices, theme))
    returns = eda.daily_returns(prices)
    show(viz.returns_chart(returns, theme))
    rolling = eda.rolling_statistics(prices, 21, ppy)
    c1, c2 = st.columns(2)
    with c1:
        show(viz.return_histogram(returns, theme))
        show(viz.line_chart(rolling["rolling_mean_return"], theme, "21-day rolling mean daily return",
                            ".2%"))
    with c2:
        show(viz.line_chart(rolling["rolling_volatility_ann"], theme,
                            "21-day rolling volatility (annualised)", ".0%"))
        show(viz.area_drawdown_chart(eda.price_drawdown(prices), theme,
                                     "Instrument drawdown from running peak"))
    corr_data = result.dataset[list(result.features.columns) + ["fwd_return"]].dropna()
    show(viz.correlation_heatmap(corr_data, theme,
                                 "Correlation of features (and next-period return), full sample"))
    st.caption("Correlation is descriptive and computed on the full sample for exploration only; it "
               "is not used for modelling and does not imply causation.")


def signals_section(result: ResearchResult, backtests: dict[str, BacktestResult], probs: pd.Series,
                    period_label: str, names: list[str], colors: dict[str, str], theme: viz.Theme) -> None:
    if not names:
        st.info("Select at least one strategy in the sidebar.")
        return
    name = st.selectbox("Strategy", names, index=len(names) - 1)
    bt = backtests[name]
    ma = result.moving_averages.rename(columns={
        "sma_short": f"SMA{result.config.strategy.ma_short}",
        "sma_long": f"SMA{result.config.strategy.ma_long}"}) if name.startswith("MA") else None
    show(viz.signals_chart(result.prices, bt, colors[name], theme, ma))
    st.caption("Hollow circles mark the close at which a signal was generated; triangles mark the "
               "later execution (fill date and price). The position never earns the move between "
               "the two.")
    if name == result.ml_name:
        if period_label.startswith("Walk-forward"):
            show(viz.probability_chart(probs, result.walk_forward.threshold, theme, colors[name], []))
        else:
            bands = [(s.label.split(" (")[0], s.start, s.end) for s in result.segments.values()]
            show(viz.probability_chart(result.probabilities, result.threshold, theme, colors[name], bands))
    with st.expander(f"Trades ({len(bt.trades)})"):
        st.dataframe(bt.trades, hide_index=True)


def backtest_section(backtests: dict[str, BacktestResult], names: list[str], colors: dict[str, str],
                     theme: viz.Theme, period_label: str) -> None:
    sel = {n: backtests[n] for n in names}
    if not sel:
        st.info("Select at least one strategy in the sidebar.")
        return
    st.caption(f"{period_label}. Equity starts at the initial capital on the first day of the window.")
    note_identical_positions(backtests, list(sel))
    gross = st.toggle("Overlay gross (pre-cost) equity", value=False)
    show(viz.equity_chart(sel, colors, theme, show_gross=gross,
                          title="Equity curves: net (solid) vs gross (dotted)" if gross else
                          "Equity curves (net of costs)"))
    show(viz.drawdown_chart(sel, colors, theme))
    c1, c2 = st.columns(2)
    with c1:
        show(viz.cumulative_cost_chart(sel, colors, theme))
    with c2:
        table = pd.DataFrame({n: {"Gross cumulative return": f"{bt.daily['equity_gross'].iloc[-1] / bt.initial_capital - 1:.2%}",
                                  "Net cumulative return": f"{bt.equity.iloc[-1] / bt.initial_capital - 1:.2%}",
                                  "Total costs (% of equity)": f"{bt.daily['total_cost'].sum():.2%}",
                                  "Position changes": f"{int((bt.daily['turnover'] > 0).sum()):,}"}
                              for n, bt in sel.items()})
        st.markdown("**Gross vs net**")
        st.dataframe(table)
    show(viz.exposure_chart(sel, colors, theme))


def performance_section(result: ResearchResult, metrics: pd.DataFrame, clf: dict, probs: pd.Series,
                        period_label: str, names: list[str], colors: dict[str, str],
                        theme: viz.Theme) -> None:
    st.subheader(f"Metrics: {period_label}")
    st.dataframe(format_metrics_table(metrics[names]) if names else pd.DataFrame())
    st.caption("Percentages are decimal returns × 100. Annualised values use "
               f"{result.config.evaluation.periods_per_year} periods/year and are shown as n/a for "
               f"windows shorter than {result.config.evaluation.min_obs_annualize} days.")

    st.subheader("In-sample vs out-of-sample")
    rows = []
    for seg in result.segments.values():
        for n in names:
            m = seg.metrics[n]
            rows.append({"Period": seg.label, "Strategy": n, "Net return": f"{m['cumulative_return_net']:.2%}",
                         "CAGR": f"{m['cagr_net']:.2%}" if pd.notna(m["cagr_net"]) else "n/a",
                         "Sharpe (net)": f"{m['sharpe_net']:.2f}" if pd.notna(m["sharpe_net"]) else "n/a",
                         "Max DD": f"{m['max_drawdown']:.2%}"})
    if result.walk_forward is not None:
        for n in names:
            m = result.walk_forward.metrics[n]
            rows.append({"Period": "Walk-forward out-of-sample", "Strategy": n,
                         "Net return": f"{m['cumulative_return_net']:.2%}",
                         "CAGR": f"{m['cagr_net']:.2%}" if pd.notna(m["cagr_net"]) else "n/a",
                         "Sharpe (net)": f"{m['sharpe_net']:.2f}" if pd.notna(m["sharpe_net"]) else "n/a",
                         "Max DD": f"{m['max_drawdown']:.2%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Training-period results are in-sample for the ML model and are not evidence of "
               "investment performance.")

    if result.walk_forward is not None:
        wf = result.walk_forward
        st.subheader(f"Walk-forward evaluation ({len(wf.folds)} folds)")
        st.caption(f"{result.config.walk_forward.window.title()} window; each fold refits on past "
                   "data only, using the configured hyper-parameters and a fixed threshold of "
                   f"{wf.threshold:.2f}.")
        show(viz.walk_forward_excess_chart(wf.folds, result.ml_name, result.benchmark_name, theme))
        folds = wf.folds.copy()
        for col in ("train_start", "train_end", "test_start", "test_end"):
            folds[col] = folds[col].dt.strftime("%Y-%m-%d")
        with st.expander("Fold details"):
            st.dataframe(folds, hide_index=True)

    st.subheader("Classification diagnostics")
    st.markdown("Accuracy alone does not establish profitability: a model can be right on many "
                "small moves and wrong on the few large ones, and trading costs are ignored. "
                "Compare accuracy with the **majority-class baseline** (always predicting the more "
                "common outcome).")
    clf_table = classification_frame(result).apply(pd.to_numeric)
    st.dataframe(clf_table.style.format(precision=3, na_rep="n/a").format("{:.0f}", subset=["n_count"]))
    c1, c2 = st.columns(2)
    if "confusion_matrix" in clf:
        with c1:
            show(viz.confusion_matrix_chart(clf["confusion_matrix"], theme,
                                            f"Confusion matrix: {period_label}"))
    with c2:
        labels = result.dataset["label"].reindex(probs.index)
        show(viz.probability_histogram(probs, labels, theme,
                                       result.walk_forward.threshold if period_label.startswith("Walk")
                                       else result.threshold))

    st.subheader("Sensitivity to transaction costs (test period)")
    sens = result.cost_sensitivity[result.cost_sensitivity["strategy"].isin(names)]
    c1, c2 = st.columns(2)
    with c1:
        show(viz.cost_sensitivity_chart(sens, colors, theme, "sharpe_net"))
    with c2:
        show(viz.cost_sensitivity_chart(sens, colors, theme, "cumulative_return_net"))

    boot_cfg = result.config.evaluation.bootstrap
    boot_tables = [("Test period", result.bootstrap)]
    if result.walk_forward is not None:
        boot_tables.append(("Walk-forward", result.walk_forward.bootstrap))
    for label, table in boot_tables:
        if table is None or table.empty:
            continue
        st.subheader(f"Uncertainty: block bootstrap ({label})")
        table = table[table["strategy"].isin(names)]
        show(viz.bootstrap_chart(table, colors, theme, boot_cfg.confidence))
        st.dataframe(table.style.format(precision=3), hide_index=True)
    if boot_cfg.enabled:
        st.caption(f"Moving-block bootstrap ({boot_cfg.n_resamples} resamples, {boot_cfg.block_size}-day "
                   "blocks, same blocks for all strategies). Assumes rough stationarity within the "
                   "window; intervals are approximate, do not adjust for the number of configurations "
                   "tried, and are not a formal significance test.")

    st.subheader("Model details")
    c1, c2 = st.columns(2)
    with c1:
        if not result.tuning_table.empty:
            st.markdown("**Hyper-parameter selection (validation log loss)**")
            st.dataframe(result.tuning_table, hide_index=True)
        if not result.threshold_table.empty:
            st.markdown("**Threshold selection (validation net Sharpe)**")
            st.dataframe(result.threshold_table, hide_index=True)
    with c2:
        if result.coefficients is not None:
            show(viz.coefficient_chart(result.coefficients, theme, colors.get(result.ml_name, theme.series[2])))


def research_notes(result: ResearchResult) -> None:
    cfg = result.config
    st.subheader("Feature definitions")
    st.caption("All features for a row dated t use prices and volume up to the close of t only.")
    st.dataframe(pd.DataFrame(result.feature_descriptions.items(), columns=["feature", "definition"]),
                 hide_index=True)
    st.subheader("Model, target and splits")
    st.dataframe(result.split_table, hide_index=True)
    st.markdown(
        f"- Target: 1 if the forward return over {cfg.target.horizon} period(s) "
        f"({cfg.target.basis} basis) is > {cfg.target.positive_threshold:g}, else 0.\n"
        f"- {result.run_info['label_lookahead_rows']} row(s) purged before each split boundary "
        f"(+{cfg.split.embargo} embargo) so training labels never use later prices.\n"
        f"- Model: {MODEL_LABELS[result.model_name]} with parameters `{result.model_params}`.")
    st.subheader("Strategy rules")
    for name, text in result.strategy_descriptions.items():
        st.markdown(f"- **{name}:** {text}")
    st.subheader("Transaction-cost assumptions")
    st.markdown(
        f"Costs are {cfg.backtest.fee_bps:g} bps (fees) + {cfg.backtest.slippage_bps:g} bps "
        "(slippage) per unit of turnover, charged as a fraction of equity on the execution day. "
        "Turnover = |change in position|, so a full entry or exit costs "
        f"{cfg.backtest.fee_bps + cfg.backtest.slippage_bps:g} bps. This is a generic assumption, "
        "not a quote for any specific broker, exchange or instrument.")
    st.subheader("Integrity checks run")
    for check in result.checks:
        st.markdown(f"- {check}")
    st.subheader("Data provenance")
    st.json(result.run_info["data"])
    st.subheader("Limitations")
    for item in LIMITATIONS:
        st.markdown(f"- {item}")
    with st.expander("Full configuration and environment"):
        st.code(yaml.safe_dump(cfg.to_dict(), sort_keys=False), language="yaml")
        st.json(result.run_info["environment"])


def downloads(result: ResearchResult, backtests: dict[str, BacktestResult], period_key: str) -> None:
    ticker = result.run_info["data"]["ticker"]
    st.markdown("All files contain decimal returns (0.01 = 1%) computed in this session.")
    items = [
        ("Daily results: selected period (equity, returns, positions, costs)",
         daily_results_frame(backtests, period_key), f"{ticker}_{period_key}_daily.csv"),
        ("Daily results: all holdout periods",
         pd.concat([daily_results_frame(s.backtests, s.key) for s in result.segments.values()]),
         f"{ticker}_holdout_daily.csv"),
        ("Model predictions (holdout model)", predictions_frame(result), f"{ticker}_predictions.csv"),
        ("Metrics (all periods and strategies)", all_metrics_frame(result), f"{ticker}_metrics.csv"),
        ("Trades: selected period", trades_frame(backtests), f"{ticker}_{period_key}_trades.csv"),
        ("Cost sensitivity", result.cost_sensitivity, f"{ticker}_cost_sensitivity.csv"),
    ]
    if result.walk_forward is not None:
        wf = result.walk_forward
        items += [("Walk-forward folds", wf.folds, f"{ticker}_walk_forward_folds.csv"),
                  ("Walk-forward predictions", wf.probabilities.rename("prob_up").to_frame().reset_index(),
                   f"{ticker}_walk_forward_predictions.csv")]
    for i, (label, frame, name) in enumerate(items):
        st.download_button(label, csv_bytes(frame), name, "text/csv", key=f"dl{i}")
    st.download_button("Configuration (YAML)", yaml.safe_dump(result.config.to_dict(), sort_keys=False),
                       f"{ticker}_config.yaml", "text/yaml", key="dlcfg")
    if st.button("Save complete run to reports/"):
        folder = save_run(result)
        st.success(f"Saved to {folder}")


# --- Page ------------------------------------------------------------------------------------

def main() -> None:
    overrides, upload, shown, submitted = sidebar()
    if submitted:
        run(overrides, upload)

    result: ResearchResult | None = st.session_state.get("result")
    prepared: PreparedData | None = st.session_state.get("prepared")
    st.title("QuantML Research Platform")
    if result is None or prepared is None:
        st.markdown(
            "Does a machine-learning model improve on simple benchmarks **after realistic trading "
            "costs, on unseen data**? Choose data and settings in the sidebar and press "
            "**Run research**.\n\n"
            "- **Buy & Hold**: passive benchmark\n- **Moving-average crossover**: technical rule\n"
            "- **ML strategy**: logistic regression on lagged price/volume features\n\n"
            "All strategies share one execution convention and cost model and are compared on "
            "identical out-of-sample dates.")
        st.info("No internet? Choose **Synthetic demo data** or **Upload CSV**.")
        return

    data = result.run_info["data"]
    st.caption(f"{data['ticker']} · {data['start']} to {data['end']} · {data['source']}")
    if data["is_synthetic"]:
        st.error("SYNTHETIC DATA: these results describe generated noise, not any real market.")

    theme = viz.get_theme(theme_mode())
    colors = viz.strategy_colors(list(result.signals), theme)
    names = filter_names(list(result.signals), shown)
    periods = [p for p, k in PERIODS.items() if k != "walk_forward" or result.walk_forward is not None]
    period = st.segmented_control("Evaluation period", periods, default=periods[0],
                                  help="Applies to the Signals, Backtest and Performance tabs.") or periods[0]
    period_key = PERIODS[period]
    backtests, metrics, clf, probs, period_label = period_data(result, period_key)

    tabs = st.tabs(["Overview", "Market data", "Strategy signals", "Backtest results",
                    "Performance comparison", "Research notes", "Downloads"])
    with tabs[0]:
        overview(result, prepared)
    with tabs[1]:
        market_data(result, theme)
    with tabs[2]:
        signals_section(result, backtests, probs, period_label, names, colors, theme)
    with tabs[3]:
        backtest_section(backtests, names, colors, theme, period_label)
    with tabs[4]:
        performance_section(result, metrics, clf, probs, period_label, names, colors, theme)
    with tabs[5]:
        research_notes(result)
    with tabs[6]:
        downloads(result, backtests, period_key)


main()

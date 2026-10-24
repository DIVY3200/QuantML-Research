"""Plotly figure builders used by the dashboard and research scripts.

Design rules: each strategy keeps one fixed colour everywhere (colour follows the entity, never
its rank); one y-axis per panel (related measures go in stacked panels, never dual axes);
hairline grids; a legend whenever there are two or more series, plus direct end-of-line labels.
Light and dark palettes are separate validated steps of the same hues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantml.backtester import BacktestResult


@dataclass(frozen=True)
class Theme:
    series: tuple[str, ...]
    text_primary: str
    text_secondary: str
    muted: str
    grid: str
    axis: str
    neutral: str
    positive: str
    negative: str
    band: str
    sequential: tuple[str, ...]


LIGHT = Theme(
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
    text_primary="#0b0b0b", text_secondary="#52514e", muted="#898781", grid="#e1e0d9",
    axis="#c3c2b7", neutral="#f0efec", positive="#2a78d6", negative="#e34948",
    band="rgba(137,135,129,0.10)",
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
)
DARK = Theme(
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    text_primary="#f0efec", text_secondary="#c3c2b7", muted="#898781", grid="#2c2c2a",
    axis="#383835", neutral="#383835", positive="#3987e5", negative="#e66767",
    band="rgba(195,194,183,0.08)",
    sequential=("#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"),
)

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def get_theme(mode: str | None) -> Theme:
    return DARK if mode == "dark" else LIGHT


def strategy_colors(names: list[str], theme: Theme) -> dict[str, str]:
    """Fixed slot per strategy by its position in the canonical strategy list."""
    return {name: theme.series[i % len(theme.series)] for i, name in enumerate(names)}


def _layout(fig: go.Figure, theme: Theme, title: str = "", height: int = 380,
            y_format: str | None = None, legend: bool = True) -> go.Figure:
    # Reserve top margin for the title plus however many rows the horizontal legend needs,
    # so the two never overlap.
    # Estimated legend width (~7px per character + ~40px per swatch) against a conservative
    # 420px, so legends in half-width dashboard columns get enough rows too.
    names = [t.name or "" for t in fig.data if t.showlegend is not False] if legend else []
    legend_px = sum(7 * len(n) + 40 for n in names)
    legend_rows = int(np.ceil(legend_px / 420)) if names else 0
    top = (36 if title else 8) + 24 * legend_rows
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=theme.text_primary), x=0, xanchor="left",
                   yref="container", y=1, yanchor="top", pad=dict(t=8)),
        # Explicit side/bottom room for tick labels and end-of-line labels: some Plotly.js
        # builds (including Streamlit's) do not auto-expand small margins, which clips them.
        height=height + 24 * max(legend_rows - 1, 0), margin=dict(l=64, r=64, t=top, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=theme.text_secondary),
        hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(color=theme.text_secondary), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=theme.axis, tickcolor=theme.axis,
                     tickfont=dict(color=theme.muted), zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=theme.grid, gridwidth=1, linecolor=theme.axis,
                     tickfont=dict(color=theme.muted), zeroline=False, automargin=True)
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    return fig


def _end_label(fig: go.Figure, x: object, y: float, text: str, color: str, theme: Theme,
               row: int | None = None, col: int | None = None) -> None:
    """Direct label at a line's final point; the text uses ink, the dot carries identity."""
    kwargs = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[text],
                             textposition="middle right", showlegend=False, hoverinfo="skip",
                             cliponaxis=False,
                             marker=dict(size=8, color=color, line=dict(width=2, color="rgba(0,0,0,0)")),
                             textfont=dict(color=theme.text_secondary, size=11)), **kwargs)


# --- Market data ----------------------------------------------------------------------------

def price_chart(prices: pd.DataFrame, theme: Theme, moving_averages: pd.DataFrame | None = None,
                title: str = "Adjusted close") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices["close"], name="Adjusted close",
                             line=dict(width=2, color=theme.text_secondary)))
    if moving_averages is not None:
        for i, col in enumerate(moving_averages.columns):
            fig.add_trace(go.Scatter(x=moving_averages.index, y=moving_averages[col], name=col,
                                     line=dict(width=1.5, color=theme.series[i + 3])))
    return _layout(fig, theme, title, legend=moving_averages is not None)


def volume_chart(prices: pd.DataFrame, theme: Theme) -> go.Figure:
    fig = go.Figure(go.Bar(x=prices.index, y=prices["volume"], name="Volume",
                           marker=dict(color=theme.series[0], line=dict(width=0))))
    fig.update_layout(bargap=0.1)
    return _layout(fig, theme, "Daily volume (shares)", height=280, legend=False)


def returns_chart(returns: pd.Series, theme: Theme) -> go.Figure:
    colors = np.where(returns >= 0, theme.positive, theme.negative)
    fig = go.Figure(go.Bar(x=returns.index, y=returns, name="Daily return",
                           marker=dict(color=colors, line=dict(width=0)),
                           hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra></extra>"))
    return _layout(fig, theme, "Daily close-to-close return", height=300, y_format=".1%",
                   legend=False)


def return_histogram(returns: pd.Series, theme: Theme, bins: int = 80) -> go.Figure:
    r = returns.dropna()
    width = (r.max() - r.min()) / bins if len(r) > 1 else 0.01
    fig = go.Figure(go.Histogram(x=r, xbins=dict(start=r.min(), end=r.max() + width, size=width),
                                 name="Daily returns",
                                 marker=dict(color=theme.series[0], line=dict(width=1, color="rgba(0,0,0,0)")),
                                 hovertemplate="%{x}: %{y} days<extra></extra>"))
    if len(r) > 2:
        # Normal density with the same mean/std, scaled to counts with the same explicit bin
        # width, for visual comparison only.
        grid = np.linspace(r.min(), r.max(), 300)
        density = np.exp(-0.5 * ((grid - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(x=grid, y=density * len(r) * width, name="Normal (same mean/std)",
                                 line=dict(width=2, color=theme.text_secondary)))
    fig.update_xaxes(tickformat=".1%")
    return _layout(fig, theme, "Distribution of daily returns", height=320)


def line_chart(series: pd.Series, theme: Theme, title: str, y_format: str | None = None,
               color: str | None = None, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Scatter(x=series.index, y=series, name=series.name or title,
                               line=dict(width=2, color=color or theme.series[0])))
    return _layout(fig, theme, title, height=height, y_format=y_format, legend=False)


def area_drawdown_chart(drawdown: pd.Series, theme: Theme, title: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="Drawdown",
                               line=dict(width=1.5, color=theme.negative)))
    return _layout(fig, theme, title, height=280, y_format=".0%", legend=False)


def correlation_heatmap(frame: pd.DataFrame, theme: Theme, title: str) -> go.Figure:
    corr = frame.corr()
    scale = [[0.0, theme.negative], [0.5, theme.neutral], [1.0, theme.positive]]
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                               zmid=0, colorscale=scale, xgap=2, ygap=2,
                               colorbar=dict(title="corr", tickfont=dict(color=theme.muted)),
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, theme, title, height=max(380, 26 * len(corr)), legend=False)


# --- Strategy signals -----------------------------------------------------------------------

def signals_chart(prices: pd.DataFrame, backtest: BacktestResult, color: str, theme: Theme,
                  moving_averages: pd.DataFrame | None = None) -> go.Figure:
    """Price with executed trades (filled triangles, at fill price/date) and the signal dates."""
    daily = backtest.daily
    px = prices.loc[daily.index]
    fig = price_chart(px, theme, moving_averages.loc[daily.index] if moving_averages is not None
                      else None, title=f"Signals and executions: {backtest.name}")
    trades = backtest.trades
    for side, symbol in (("BUY", "triangle-up"), ("SELL", "triangle-down")):
        t = trades[trades["side"] == side]
        if t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=t["execution_date"], y=t["execution_price"], mode="markers",
            name=f"{side.title()} executed",
            marker=dict(symbol=symbol, size=10, color=color if side == "BUY" else "rgba(0,0,0,0)",
                        line=dict(width=2, color=color)),
            customdata=np.stack([t["signal_date"].dt.strftime("%Y-%m-%d"),
                                 t["execution_date"].dt.strftime("%Y-%m-%d")], axis=1),
            hovertemplate=(f"{side}<br>signal at close of %{{customdata[0]}}<br>"
                           "filled %{customdata[1]} at %{y:.2f}<extra></extra>")))
        sig_close = prices["close"].reindex(t["signal_date"])
        fig.add_trace(go.Scatter(
            x=t["signal_date"], y=sig_close.values, mode="markers",
            name=f"{side.title()} signal generated",
            marker=dict(symbol="circle-open", size=8, color=theme.muted, line=dict(width=1.5)),
            hovertemplate=f"{side} signal generated at close %{{x|%Y-%m-%d}}<extra></extra>"))
    return _layout(fig, theme, f"Signals and executions: {backtest.name}", height=440)


def probability_chart(probabilities: pd.Series, threshold: float, theme: Theme, color: str,
                      segments: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> go.Figure:
    fig = go.Figure()
    for i, (label, start, end) in enumerate(segments):
        if i % 2 == 0:
            fig.add_vrect(x0=start, x1=end, fillcolor=theme.band, line_width=0, layer="below")
        fig.add_annotation(x=start, y=1.0, yref="paper", text=label, showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=theme.muted))
    fig.add_trace(go.Scatter(x=probabilities.index, y=probabilities, name="P(next return > 0)",
                             line=dict(width=1.5, color=color)))
    fig.add_hline(y=threshold, line=dict(width=1, color=theme.text_secondary),
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="bottom right",
                  annotation_font=dict(color=theme.text_secondary))
    return _layout(fig, theme, "Model probability of a positive next-period return", height=340,
                   y_format=".0%", legend=False)


def probability_histogram(probabilities: pd.Series, labels: pd.Series, theme: Theme,
                          threshold: float) -> go.Figure:
    data = pd.DataFrame({"p": probabilities, "y": labels}).dropna()
    fig = go.Figure()
    for value, name, color in ((1, "Realised up", theme.positive), (0, "Realised down", theme.negative)):
        fig.add_trace(go.Histogram(x=data.loc[data["y"] == value, "p"], name=name, opacity=0.65,
                                   nbinsx=40, marker=dict(color=color)))
    fig.add_vline(x=threshold, line=dict(width=1, color=theme.text_secondary))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(tickformat=".0%", title="Predicted probability")
    return _layout(fig, theme, "Predicted probabilities by realised outcome", height=320)


def confusion_matrix_chart(cm: list[list[int]], theme: Theme, title: str) -> go.Figure:
    z = np.asarray(cm)
    labels = ["Down / flat (0)", "Up (1)"]
    scale = [[i / (len(theme.sequential) - 1), c] for i, c in enumerate(theme.sequential)]
    fig = go.Figure(go.Heatmap(z=z, x=[f"Predicted {lab}" for lab in labels],
                               y=[f"Actual {lab}" for lab in labels], colorscale=scale,
                               showscale=False, xgap=2, ygap=2, text=z, texttemplate="%{text}",
                               textfont=dict(size=14),
                               hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _layout(fig, theme, title, height=300, legend=False)


# --- Backtest results -----------------------------------------------------------------------

def equity_chart(backtests: dict[str, BacktestResult], colors: dict[str, str], theme: Theme,
                 title: str = "Equity curves (net of costs)", show_gross: bool = False) -> go.Figure:
    fig = go.Figure()
    for name, bt in backtests.items():
        eq = bt.daily["equity_net"]
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=f"{name} (net)" if show_gross else name,
                                 line=dict(width=2, color=colors[name]),
                                 hovertemplate="%{y:,.0f}"))
        if show_gross:
            g = bt.daily["equity_gross"]
            fig.add_trace(go.Scatter(x=g.index, y=g, name=f"{name} (gross)",
                                     line=dict(width=1.5, color=colors[name], dash="dot"),
                                     hovertemplate="%{y:,.0f}"))
        total = eq.iloc[-1] / bt.initial_capital - 1
        _end_label(fig, eq.index[-1], eq.iloc[-1], f"{total:+.1%}", colors[name], theme)
    return _layout(fig, theme, title, height=420, y_format=",.0f")


def drawdown_chart(backtests: dict[str, BacktestResult], colors: dict[str, str],
                   theme: Theme) -> go.Figure:
    fig = go.Figure()
    for name, bt in backtests.items():
        fig.add_trace(go.Scatter(x=bt.daily.index, y=bt.daily["drawdown"], name=name,
                                 line=dict(width=1.5, color=colors[name]),
                                 hovertemplate="%{y:.1%}"))
    return _layout(fig, theme, "Drawdown from running peak (net)", height=340, y_format=".0%")


def cumulative_cost_chart(backtests: dict[str, BacktestResult], colors: dict[str, str],
                          theme: Theme) -> go.Figure:
    fig = go.Figure()
    for name, bt in backtests.items():
        c = bt.daily["cumulative_cost"]
        fig.add_trace(go.Scatter(x=c.index, y=c, name=name, line=dict(width=2, color=colors[name]),
                                 hovertemplate="%{y:.2%}"))
        _end_label(fig, c.index[-1], c.iloc[-1], f"{c.iloc[-1]:.2%}", colors[name], theme)
    return _layout(fig, theme, "Cumulative transaction costs (sum of daily costs, % of equity)",
                   height=320, y_format=".1%")


def exposure_chart(backtests: dict[str, BacktestResult], colors: dict[str, str],
                   theme: Theme) -> go.Figure:
    """Small multiples (one panel per strategy) so 0/1 step lines never overlap."""
    names = list(backtests)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=names)
    for i, name in enumerate(names, start=1):
        pos = backtests[name].daily["position"]
        fig.add_trace(go.Scatter(x=pos.index, y=pos, name=name, line=dict(width=1.5, color=colors[name],
                                                                          shape="hv"),
                                 fill="tozeroy", hovertemplate="%{y:.0%}"), row=i, col=1)
        fig.update_yaxes(range=[-0.05, 1.05], tickformat=".0%", tickvals=[0, 1], row=i, col=1)
    fig.update_annotations(font=dict(size=12, color=theme.text_secondary), x=0, xanchor="left")
    return _layout(fig, theme, "Position held after execution (0% = cash, 100% = long)",
                   height=110 + 110 * len(names), legend=False)


def walk_forward_excess_chart(folds: pd.DataFrame, ml_name: str, benchmark: str,
                              theme: Theme) -> go.Figure:
    excess = folds[f"return_net::{ml_name}"] - folds[f"return_net::{benchmark}"]
    colors = np.where(excess >= 0, theme.positive, theme.negative)
    labels = [f"{r.test_start:%Y-%m-%d} to {r.test_end:%Y-%m-%d}" for r in folds.itertuples()]
    fig = go.Figure(go.Bar(x=folds["fold"], y=excess, marker=dict(color=colors, line=dict(width=0)),
                           customdata=labels,
                           hovertemplate="Fold %{x} (%{customdata})<br>ML minus benchmark: "
                                         "%{y:.2%}<extra></extra>"))
    fig.update_layout(bargap=0.15)
    fig.update_xaxes(title="Walk-forward fold (chronological)")
    return _layout(fig, theme, f"Per-fold net return: {ml_name} minus {benchmark}", height=320,
                   y_format=".1%", legend=False)


def cost_sensitivity_chart(table: pd.DataFrame, colors: dict[str, str], theme: Theme,
                           metric: str = "sharpe_net") -> go.Figure:
    fig = go.Figure()
    for name, grp in table.groupby("strategy", sort=False):
        fig.add_trace(go.Scatter(x=grp["cost_bps"], y=grp[metric], name=name, mode="lines+markers",
                                 line=dict(width=2, color=colors[name]), marker=dict(size=8)))
    fig.update_xaxes(title="Total cost per unit turnover (bps)")
    title = ("Net Sharpe ratio vs. trading cost" if metric == "sharpe_net"
             else "Net cumulative return vs. trading cost")
    return _layout(fig, theme, title, height=340,
                   y_format=None if metric == "sharpe_net" else ".0%")


def bootstrap_chart(table: pd.DataFrame, colors: dict[str, str], theme: Theme,
                    confidence: float) -> go.Figure:
    fig = go.Figure()
    for row in table.itertuples():
        fig.add_trace(go.Scatter(
            x=[row.sharpe_net], y=[row.strategy], mode="markers", name=row.strategy,
            marker=dict(size=10, color=colors[row.strategy]),
            error_x=dict(type="data", symmetric=False, array=[row.sharpe_ci_high - row.sharpe_net],
                         arrayminus=[row.sharpe_net - row.sharpe_ci_low], color=colors[row.strategy],
                         thickness=2, width=6),
            hovertemplate=(f"Sharpe %{{x:.2f}}<br>{confidence:.0%} interval "
                           f"[{row.sharpe_ci_low:.2f}, {row.sharpe_ci_high:.2f}]<extra></extra>")))
    fig.add_vline(x=0, line=dict(width=1, color=theme.axis))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=theme.grid, title="Annualised net Sharpe ratio")
    return _layout(fig, theme, f"Block-bootstrap {confidence:.0%} intervals for net Sharpe",
                   height=260, legend=False)


def coefficient_chart(table: pd.DataFrame, theme: Theme, color: str) -> go.Figure:
    data = table.iloc[::-1]
    fig = go.Figure(go.Bar(x=data["value"], y=data["feature"], orientation="h",
                           marker=dict(color=color, line=dict(width=0)),
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(showgrid=True, gridcolor=theme.grid)
    fig.update_yaxes(showgrid=False)
    kind = table["kind"].iloc[0] if len(table) else ""
    return _layout(fig, theme, f"Model {kind}s (descriptive, not causal)",
                   height=max(320, 22 * len(table)), legend=False)

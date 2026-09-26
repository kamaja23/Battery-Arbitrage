"""Plotly figures for the backtest ledger.

The price + SoC + dispatch chart is the primary demo visual: it shows the
strategy buying low and selling high in the same frame.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from arb.sim.engine import BacktestResult

_PRICE_COLOR = "#1f3b57"
_SOC_COLOR = "#0e7c86"
_CHARGE_COLOR = "#2e7d32"
_DISCHARGE_COLOR = "#c62828"

_MARKET_NAMES = {"RTM": "real-time prices", "DAM": "day-ahead prices"}


def _market(code: str) -> str:
    return _MARKET_NAMES.get(code, code)


def dispatch_figure(
    result: BacktestResult,
    title: str | None = None,
) -> go.Figure:
    """Price, SoC, and charge/discharge power over the backtest window."""
    ledger = result.ledger
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        subplot_titles=("Settlement point price", "Battery state of charge"),
    )

    fig.add_trace(
        go.Scatter(
            x=ledger["interval_start"],
            y=ledger["price_usd_per_mwh"],
            name="price $/MWh",
            line=dict(color=_PRICE_COLOR, width=1),
            hovertemplate="%{x|%b %d %H:%M}<br>$%{y:,.2f}/MWh<extra></extra>",
        ),
        row=1,
        col=1,
    )

    charge = ledger.loc[ledger["charge_kw"] > 0]
    discharge = ledger.loc[ledger["discharge_kw"] > 0]

    fig.add_trace(
        go.Bar(
            x=charge["interval_start"],
            y=charge["charge_kw"],
            name="charging kW",
            marker_color=_CHARGE_COLOR,
            hovertemplate="%{x|%b %d %H:%M}<br>charge %{y:,.1f} kW<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=discharge["interval_start"],
            y=-discharge["discharge_kw"],
            name="discharging kW",
            marker_color=_DISCHARGE_COLOR,
            hovertemplate="%{x|%b %d %H:%M}<br>discharge %{y:,.1f} kW<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ledger["interval_start"],
            y=100 * ledger["soc_fraction"],
            name="SoC %",
            line=dict(color=_SOC_COLOR, width=2),
            hovertemplate="%{x|%b %d %H:%M}<br>SoC %{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    battery = result.battery
    fig.add_hline(
        y=100 * battery.soc_max,
        line=dict(dash="dot", color=_SOC_COLOR),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=100 * battery.soc_min,
        line=dict(dash="dot", color=_SOC_COLOR),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="$/MWh", zeroline=True, row=1, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=1, overlaying="y", side="right")
    fig.update_yaxes(title_text="State of charge %", range=[0, 100], row=2, col=1)

    heading = title or (
        f"{result.settlement_point} {_market(result.market)} - {result.strategy_name} "
        f"({battery.power_kw:g} kW / {battery.capacity_kwh:g} kWh)"
    )
    fig.update_layout(
        title=heading,
        height=760,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=60, t=90, b=40),
        barmode="relative",
    )
    return fig


def cumulative_revenue_figure(result: BacktestResult) -> go.Figure:
    """Cumulative net revenue, the clearest single picture of the outcome."""
    ledger = result.ledger
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ledger["interval_start"],
            y=ledger["net_usd"].cumsum(),
            name="cumulative net",
            line=dict(color=_PRICE_COLOR, width=2),
            hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Cumulative net revenue - {result.settlement_point}, {_market(result.market)}",
        height=380,
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=40),
        yaxis_title="USD",
    )
    return fig


def daily_revenue_figure(result: BacktestResult) -> go.Figure:
    """Daily net revenue as a diverging bar chart."""
    daily = result.ledger.set_index("interval_start")["net_usd"].resample("D").sum()
    colors = [
        _DISCHARGE_COLOR if v >= 0 else _CHARGE_COLOR for v in daily.to_numpy()
    ]
    fig = go.Figure(
        go.Bar(
            x=daily.index,
            y=daily.to_numpy(),
            marker_color=colors,
            name="daily net",
            hovertemplate="%{x|%b %d}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Daily net revenue - {result.settlement_point}",
        height=380,
        margin=dict(l=60, r=40, t=60, b=40),
        yaxis_title="USD",
    )
    return fig


def zone_comparison_figure(comparison: pd.DataFrame) -> go.Figure:
    """Annualized revenue per zone, weakest at the bottom.

    Zones are ordered so the reader sees the ranking immediately, and a zero
    line marks where a zone stops being profitable.
    """
    ordered = comparison.sort_values("usd_per_kw_year", ascending=True)
    colors = [
        _CHARGE_COLOR if v > 0 else _DISCHARGE_COLOR
        for v in ordered["usd_per_kw_year"]
    ]
    fig = go.Figure(
        go.Bar(
            x=ordered["usd_per_kw_year"],
            y=ordered.index,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>$%{x:,.2f} / kW-year<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#666", width=1))
    fig.update_layout(
        title="Annualized revenue per zone (same battery, same window)",
        height=340,
        margin=dict(l=110, r=40, t=60, b=40),
        xaxis_title="USD / kW-year",
        showlegend=False,
    )
    return fig


def forecast_plan_figure(plan, battery) -> go.Figure:
    """Forecast price curve for one day with the schedule planned against it."""
    frame = plan.to_frame()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=frame["interval_start"],
            y=frame["forecast_usd_per_mwh"],
            name="forecast $/MWh",
            line=dict(color=_PRICE_COLOR, width=2, dash="dash"),
            hovertemplate="%{x|%H:%M}<br>forecast $%{y:,.2f}/MWh<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=frame["interval_start"],
            y=frame["charge_kw"],
            name="planned charge kW",
            marker_color=_CHARGE_COLOR,
            hovertemplate="%{x|%H:%M}<br>charge %{y:,.1f} kW<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=frame["interval_start"],
            y=-frame["discharge_kw"],
            name="planned discharge kW",
            marker_color=_DISCHARGE_COLOR,
            hovertemplate="%{x|%H:%M}<br>discharge %{y:,.1f} kW<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="forecast $/MWh", secondary_y=False)
    fig.update_yaxes(
        title_text="kW",
        range=[-1.1 * battery.power_kw, 1.1 * battery.power_kw],
        secondary_y=True,
    )
    fig.update_layout(
        title=f"Plan for {plan.date:%a %b %d, %Y} (forecast, not a guarantee)",
        height=420,
        barmode="relative",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60, b=40),
    )
    return fig

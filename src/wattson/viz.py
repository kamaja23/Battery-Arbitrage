"""Plotly figures for the backtest ledger.

The price + SoC + dispatch chart is the primary demo visual: it shows the
strategy buying low and selling high in the same frame.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wattson.sim.engine import BacktestResult

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


def zone_comparison_figure(comparison: pd.DataFrame, battery=None) -> go.Figure:
    """Annualized revenue per zone, weakest at the bottom.

    Zones are ordered so the reader sees the ranking immediately, and a zero
    line marks where a zone stops being profitable.
    """
    from wattson.zones import name_for

    ordered = comparison.sort_values("usd_per_kw_year", ascending=True)
    colors = [
        _CHARGE_COLOR if v > 0 else _DISCHARGE_COLOR
        for v in ordered["usd_per_kw_year"]
    ]
    codes = list(ordered.index)
    # With a battery, show what that battery keeps per year rather than the
    # per-kW figure, which means nothing to most people.
    scale = battery.power_kw if battery is not None else 1.0
    values = ordered["usd_per_kw_year"] * scale
    fig = go.Figure(
        go.Bar(
            x=values,
            # Place names on the axis; the ERCOT code stays in the hover.
            y=[name_for(code) for code in codes],
            customdata=codes,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>%{customdata}<br>$%{x:,.2f}"
            + (" a year" if battery is not None else " / kW-year")
            + "<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#666", width=1))
    fig.update_layout(
        title="Annualized revenue per zone (same battery, same window)"
        if battery is None
        else "What it would earn per year after wear, by area",
        height=max(340, 30 * len(codes) + 110),
        margin=dict(r=40, t=60, b=40),
        xaxis_title="USD / kW-year" if battery is None else "Dollars per year, after battery wear",
        yaxis=dict(automargin=True),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Plain-language charts for the main page. Prices are shown in cents per kWh,
# the unit on a home electricity bill ($/MWh divided by 10).
# ---------------------------------------------------------------------------

_BUY_COLOR = "#1565c0"  # charging = buying power
_SELL_COLOR = "#ef6c00"  # discharging = selling power
_GAIN_COLOR = "#2e7d32"
_LOSS_COLOR = "#c62828"
_NEUTRAL_COLOR = "#9e9e9e"


def cents_per_kwh(usd_per_mwh):
    """$/MWh to cents per kWh: $30/MWh is 3 cents per kWh."""
    return usd_per_mwh / 10.0


def day_in_the_life_figure(day: pd.DataFrame, battery) -> go.Figure:
    """One day: the price of power, and when the battery bought and sold.

    ``day`` is the slice of a backtest ledger for a single date.
    """
    price = cents_per_kwh(day["price_usd_per_mwh"])
    buying = day["charge_kw"] > 0
    selling = day["discharge_kw"] > 0
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.08,
        subplot_titles=("Price of electricity (cents per kWh)", "How full the battery is"),
    )
    fig.add_trace(
        go.Scatter(
            x=day["interval_start"], y=price, name="Price",
            line=dict(color=_PRICE_COLOR, width=2),
            hovertemplate="%{x|%-I:%M %p}<br>%{y:.1f}¢ per kWh<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day.loc[buying, "interval_start"], y=price[buying], mode="markers",
            name="Buying (charging)", marker=dict(color=_BUY_COLOR, size=9),
            hovertemplate="%{x|%-I:%M %p}<br>bought at %{y:.1f}¢<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day.loc[selling, "interval_start"], y=price[selling], mode="markers",
            name="Selling (discharging)", marker=dict(color=_SELL_COLOR, size=9),
            hovertemplate="%{x|%-I:%M %p}<br>sold at %{y:.1f}¢<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=day["interval_start"], y=100 * day["soc_fraction"], name="Battery level",
            fill="tozeroy", line=dict(color=_SOC_COLOR, width=2),
            hovertemplate="%{x|%-I:%M %p}<br>%{y:.0f}% full<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="¢ per kWh", row=1, col=1)
    fig.update_yaxes(title_text="% full", range=[0, 100], row=2, col=1)
    fig.update_xaxes(tickformat="%-I %p", row=2, col=1)
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
        margin=dict(l=60, r=30, t=80, b=40),
    )
    return fig


def daily_earnings_figure(result) -> go.Figure:
    """Money made (green) or lost (red) each day, energy trading only."""
    daily = result.ledger.set_index("interval_start")["net_usd"].resample("D").sum()
    colors = [_GAIN_COLOR if v >= 0 else _LOSS_COLOR for v in daily.to_numpy()]
    fig = go.Figure(
        go.Bar(
            x=daily.index, y=daily.to_numpy(), marker_color=colors,
            hovertemplate="%{x|%a %b %-d}<br>$%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color="#666", width=1))
    fig.update_layout(
        height=340,
        yaxis_title="Dollars made that day",
        margin=dict(l=60, r=30, t=30, b=40),
        showlegend=False,
    )
    return fig


def strategy_figure(rows: list[tuple[str, float]]) -> go.Figure:
    """Horizontal bars: what each way of running the battery would have left after wear.

    ``rows`` are (label, dollars left after wear), drawn top to bottom. The
    last row is treated as the unreachable reference and drawn in grey.
    """
    labels = [label for label, _ in rows][::-1]
    values = [value for _, value in rows][::-1]
    colors = [
        _NEUTRAL_COLOR if i == 0 else (_GAIN_COLOR if v >= 0 else _LOSS_COLOR)
        for i, v in enumerate(values)
    ]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h", marker_color=colors,
            text=[("-$" if v < 0 else "$") + f"{abs(v):,.2f}" for v in values],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>$%{x:,.2f} left after wear<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#666", width=1))
    fig.update_layout(
        height=90 + 55 * len(rows),
        xaxis_title="Dollars left after battery wear",
        yaxis=dict(automargin=True),
        margin=dict(r=80, t=20, b=40),
        showlegend=False,
    )
    return fig


def forecast_plan_figure(plan, battery) -> go.Figure:
    """Tomorrow's expected price, with when the battery plans to buy and sell."""
    frame = plan.to_frame()
    price = cents_per_kwh(frame["forecast_usd_per_mwh"])
    buying = frame["charge_kw"] > 1e-6
    selling = frame["discharge_kw"] > 1e-6
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["interval_start"], y=price, name="Expected price",
            line=dict(color=_PRICE_COLOR, width=2, dash="dash"),
            hovertemplate="%{x|%-I:%M %p}<br>expected %{y:.1f}¢ per kWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.loc[buying, "interval_start"], y=price[buying], mode="markers",
            name="Plans to buy", marker=dict(color=_BUY_COLOR, size=9),
            hovertemplate="%{x|%-I:%M %p}<br>plans to buy<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame.loc[selling, "interval_start"], y=price[selling], mode="markers",
            name="Plans to sell", marker=dict(color=_SELL_COLOR, size=9),
            hovertemplate="%{x|%-I:%M %p}<br>plans to sell<extra></extra>",
        )
    )
    fig.update_layout(
        height=380,
        yaxis_title="Expected price (¢ per kWh)",
        xaxis=dict(tickformat="%-I %p"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=30, t=40, b=40),
    )
    return fig

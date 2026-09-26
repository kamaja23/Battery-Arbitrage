"""Streamlit front end.

Answers the homeowner question the product is built around: "if I put a
battery in my ERCOT zone, what does it earn?"

The app is deliberately honest about one thing. Every number here is
annualized from a *historical* price window, because this tool has no price
forecast. It is labelled as an annualization, not a projection, so nobody
reads it as a promise about future earnings.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from arb.config import PRESETS, BatteryConfig
from arb.data.cache import cached_fetch
from arb.data.ercot_source import ErcotLiveProvider, load_keys_file
from arb.data.providers import PriceRequest
from arb.metrics import compute_metrics, compare_strategies
from arb.sim.engine import run_backtest
from arb.strategies.threshold import ThresholdStrategy
from arb.viz import cumulative_revenue_figure, daily_revenue_figure, dispatch_figure

LOAD_ZONES = (
    "LZ_HOUSTON",
    "LZ_NORTH",
    "LZ_WEST",
    "LZ_CENTRAL",
    "LZ_SOUTH",
    "LZ_EAST",
)

# Intertrust's published range for optimized pure RTM arbitrage, used only as
# a reference point for what a well-run commercial fleet actually achieves.
BENCHMARK_LOW = 55.0
BENCHMARK_HIGH = 66.0

PRESET_LABELS = {
    "residential_13kwh": "Residential 13.5 kWh (typical home)",
    "commercial_1mw": "Commercial 1 MW / 2 MWh (fleet scale)",
}


def _load_credentials() -> bool:
    try:
        load_keys_file("ERCOT API Keys.txt")
        return True
    except (RuntimeError, OSError):
        return False


@st.cache_data(show_spinner=False, ttl=3600)
def _prices_cached(
    zone: str, start: str, end: str, market: str, ready: bool
) -> pd.DataFrame:
    if not ready:
        raise RuntimeError("credentials unavailable")
    request = PriceRequest(
        settlement_point=zone, start_date=start, end_date=end, market=market
    )
    return cached_fetch(ErcotLiveProvider(), request)


def _money(value: float, decimals: int = 0) -> str:
    """Format money so losses read as -$1 rather than $-1."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def _metrics_row(metrics, title: str) -> None:
    cols = st.columns(4)
    cols[0].metric("Net earned", _money(metrics.net_usd))
    cols[1].metric("After degradation", _money(metrics.net_after_degradation_usd))
    cols[2].metric("Per kW / year", _money(metrics.usd_per_kw_year, 2))
    payback = metrics.payback_years
    cols[3].metric(
        "Payback",
        "never" if payback is None else f"{payback:,.0f} yr",
    )
    st.caption(title)


def main() -> None:
    st.set_page_config(page_title="ERCOT Battery Arbitrage", layout="wide")
    st.title("Would a battery pay for itself in your ERCOT zone?")

    credentials_ready = _load_credentials()

    with st.sidebar:
        st.header("Your setup")
        zone = st.selectbox("Load zone", LOAD_ZONES, index=2)
        market = st.radio("Price market", ["RTM", "DAM"], horizontal=True)

        preset_key = st.selectbox(
            "Battery",
            list(PRESET_LABELS),
            format_func=lambda k: PRESET_LABELS[k],
        )
        preset = PRESETS[preset_key]

        custom = st.checkbox("Customize size", value=False)
        if custom:
            capacity = st.slider("Capacity (kWh)", 5.0, 4000.0, preset.capacity_kwh, 5.0)
            power = st.slider("Power (kW)", 1.0, 2000.0, preset.power_kw, 5.0)
            efficiency = st.slider("Round-trip efficiency", 0.80, 0.98, preset.round_trip_efficiency, 0.01)
            installed = st.number_input(
                "Installed cost ($)", 0.0, 2_000_000.0, float(preset.installed_cost_usd or 0.0), 500.0
            )
            battery = replace(
                preset,
                name=f"custom_{capacity:g}kwh",
                capacity_kwh=capacity,
                power_kw=power,
                round_trip_efficiency=efficiency,
                installed_cost_usd=installed,
            )
        else:
            battery = preset

        default_end = pd.Timestamp("today").normalize() - pd.Timedelta(days=1)
        default_start = default_end - pd.Timedelta(days=29)
        window = st.date_input(
            "Historical window",
            value=(default_start.date(), default_end.date()),
            max_value=default_end.date(),
        )
        st.caption(
            "Uses real cached ERCOT prices. The API is only called for days "
            "that aren't cached yet."
        )
        if not credentials_ready:
            st.warning(
                "No ERCOT credentials found, so only cached data will load. "
                "Add 'ERCOT API Keys.txt' to pull new dates."
            )

    if len(window) != 2:
        st.info("Pick a start and end date to run the backtest.")
        return
    start, end = window[0].isoformat(), window[1].isoformat()

    try:
        with st.spinner("Loading prices..."):
            prices = _prices_cached(zone, start, end, market, credentials_ready)
    except Exception as exc:  # noqa: BLE001 - surface any data problem in the UI
        st.error(f"Could not load prices for {zone} {start}..{end}: {exc}")
        return

    result = run_backtest(prices, battery, ThresholdStrategy())
    metrics = compute_metrics(result)
    comparison = compare_strategies(prices, battery, ThresholdStrategy())

    span = (
        f"{prices['interval_start'].iloc[0].date()} to "
        f"{prices['interval_end'].iloc[-1].date()}"
    )
    p = prices["price_usd_per_mwh"]
    st.caption(
        f"{zone} {market} · {len(prices):,} intervals · {span} · "
        f"price ${p.min():,.2f} to ${p.max():,.2f}/MWh · "
        f"{(p < 0).sum():,} negative-price intervals"
    )

    if metrics.net_usd <= 0 or metrics.net_after_degradation_usd <= 0:
        st.error(
            "**This battery did not make money in this window.** "
            "Pure energy arbitrage is a narrow game: the spread has to clear "
            "round-trip losses, degradation, and the cost of the hardware."
        )
    else:
        st.success(
            f"**{_money(metrics.net_usd)} earned** over this window, "
            f"{_money(metrics.net_usd / metrics.duration_days)}/day."
        )

    _metrics_row(
        metrics,
        f"Annualized from this window only. Battery: {battery.name}, "
        f"{battery.capacity_kwh:,.0f} kWh / {battery.power_kw:,.0f} kW.",
    )

    st.subheader("How that compares")
    st.dataframe(
        comparison.round(2).style.format(
            {
                "net_usd": "${:,.2f}",
                "net_after_degradation_usd": "${:,.2f}",
                "usd_per_kw_year": "${:,.2f}",
                "equivalent_full_cycles": "{:,.2f}",
                "capture_ratio": "{:.1%}",
            }
        ),
        width='stretch',
    )
    capture = comparison.loc[
        comparison["kind"] == "online", "capture_ratio"
    ].iloc[0]
    st.caption(
        "Without a battery there is no spread to capture, so the baseline is $0. "
        "The rule-based strategy trades only on information it could actually "
        "have had; perfect foresight sees the whole future and is unreachable."
        + (f" This run captured **{capture:.0%}** of that bound." if capture == capture else "")
    )

    per_kw = metrics.usd_per_kw_year
    if per_kw > 0:
        gap = per_kw / BENCHMARK_LOW
        st.metric(
            "vs. published fleet benchmark",
            f"{gap:.0%} of ${BENCHMARK_LOW:.0f}/kW-yr",
        )
        st.caption(
            f"Optimized ERCOT RTM arbitrage is reported around "
            f"${BENCHMARK_LOW:.0f}-${BENCHMARK_HIGH:.0f} per kW-year "
            f"(Intertrust). A single zone, a simple threshold rule, and no "
            f"demand charges or fleet coordination is well short of that."
        )

    st.subheader("Buying low, selling high")
    st.plotly_chart(
        dispatch_figure(result, f"{zone} dispatch"),
        width='stretch',
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            daily_revenue_figure(result), width='stretch'
        )
    with right:
        st.plotly_chart(
            cumulative_revenue_figure(result), width='stretch'
        )

    with st.expander("Price and dispatch detail"):
        st.dataframe(
            result.ledger[
                [
                    "interval_start",
                    "price_usd_per_mwh",
                    "charge_kw",
                    "discharge_kw",
                    "soc_fraction",
                    "net_usd",
                ]
            ].round(3),
            width='stretch',
            height=400,
        )

    st.caption(
        "All figures are annualized from historical prices, not forecasts. "
        "This model covers energy-only arbitrage: it does not value demand "
        "charge savings, solar self-consumption, or backup power, which are "
        "usually what make a home battery worth owning."
    )

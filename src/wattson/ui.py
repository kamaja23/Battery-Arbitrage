"""Wattson's Streamlit front end.

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

from wattson.config import PRESETS, BatteryConfig
from wattson.data.cache import cached_fetch, cached_windows, load
from wattson.data.ercot_source import ErcotLiveProvider, load_keys_file
from wattson.data.providers import PriceRequest
from wattson.metrics import compare_strategies, compare_zones, compute_metrics
from wattson.forecast import forecast_accuracy, plan_next_day
from wattson.sim.engine import prepare_price_series, run_backtest
from wattson.strategies.threshold import ThresholdStrategy
from wattson.viz import (
    cumulative_revenue_figure,
    daily_revenue_figure,
    dispatch_figure,
    forecast_plan_figure,
    zone_comparison_figure,
)
from wattson.zones import HUBS, LOAD_ZONES, label_for


# Intertrust's published range for optimized pure RTM arbitrage, used only as
# a reference point for what a well-run commercial fleet actually achieves.
BENCHMARK_LOW = 55.0
BENCHMARK_HIGH = 66.0

PRESET_LABELS = {
    "base_core": "Base Core · 39.2 kWh / 11 kW",
    "base_core_dual": "Base Core, two units · 78.4 kWh / 22 kW",
    "base_ground_25kwh": "Base ground-mounted · 25 kWh / 11 kW",
    "base_ground_50kwh": "Base ground-mounted, double · 50 kWh / 11 kW",
}

DEFAULT_ZONE = "LZ_AEN"

# ERCOT runs two wholesale markets. Most people have never heard of either, so
# the app names them in plain words and keeps the code for anyone who has.
MARKET_NAMES = {"RTM": "Real-time", "DAM": "Day-ahead"}
MARKET_EXPLAINERS = {
    "RTM": "What electricity actually sold for, reset every 15 minutes as "
           "supply and demand shift. It swings the most, spikes included, "
           "which is where a battery earns most of its money.",
    "DAM": "Prices agreed the day before for each hour of the next day, "
           "based on forecasts. Smoother and more predictable, with fewer "
           "spikes to profit from.",
}


def _market_name(market: str) -> str:
    """'Real-time (RTM)': the plain name first, the ERCOT code in brackets."""
    return f"{MARKET_NAMES.get(market, market)} ({market})"
MIN_DEFAULT_DAYS = 7


def _default_window() -> tuple:
    """Open on the latest window already cached for the default zone.

    A cached default means the first screen of a demo renders without the
    network. With nothing cached, fall back to the last 30 days.
    """
    yesterday = (pd.Timestamp("today").normalize() - pd.Timedelta(days=1)).date()
    windows = [
        (start, end)
        for start, end in cached_windows(DEFAULT_ZONE, "RTM")
        if (end - start).days + 1 >= MIN_DEFAULT_DAYS and end <= yesterday
    ]
    if windows:
        return windows[-1]
    return (yesterday - pd.Timedelta(days=29).to_pytimedelta(), yesterday)


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
    request = PriceRequest(
        settlement_point=zone, start_date=start, end_date=end, market=market
    )
    hit = load(request)
    if hit is not None:
        return hit
    if not ready:
        raise RuntimeError(
            f"no cached {_market_name(market)} prices for {label_for(zone)} "
            f"covering {start}..{end}, "
            "and no API credentials to fetch them"
        )
    return cached_fetch(ErcotLiveProvider(), request)


def _money(value: float, decimals: int = 0) -> str:
    """Format money so losses read as -$1 rather than $-1."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def _metrics_row(metrics, title: str, has_cost: bool = True) -> None:
    cols = st.columns(4)
    cols[0].metric("Net earned", _money(metrics.net_usd))
    cols[1].metric("After degradation", _money(metrics.net_after_degradation_usd))
    cols[2].metric("Per kW / year", _money(metrics.usd_per_kw_year, 2))
    payback = metrics.payback_years
    if not has_cost:
        cols[3].metric(
            "Payback", "n/a",
            help="Base owns the battery and pricing varies by address, so no "
                 "purchase price is assumed. Use 'Customize size' to enter one.",
        )
    else:
        cols[3].metric(
            "Payback",
            "never" if payback is None else f"{payback:,.0f} yr",
        )
    st.caption(title)


def main() -> None:
    st.set_page_config(page_title="Wattson", layout="wide")
    st.title("Wattson")
    st.header("Would a Base battery pay for itself in your ERCOT zone?")

    credentials_ready = _load_credentials()

    with st.sidebar:
        st.header("Your setup")
        view = st.radio("View", ["Your zone", "Compare zones"], horizontal=True)
        zone = st.selectbox(
            "Load zone",
            LOAD_ZONES,
            index=LOAD_ZONES.index(DEFAULT_ZONE),
            format_func=label_for,
            help="A load zone follows your utility's service territory, not your "
                 "address. Most of Austin is priced in LZ_SOUTH; only Austin "
                 "Energy's own customers are in LZ_AEN.",
        )
        market = st.radio(
            "Which prices?",
            ["RTM", "DAM"],
            horizontal=True,
            format_func=_market_name,
            help="ERCOT, the Texas grid operator, sells electricity in two "
                 "markets. Real-time (RTM): " + MARKET_EXPLAINERS["RTM"]
                 + " Day-ahead (DAM): " + MARKET_EXPLAINERS["DAM"],
        )
        st.caption(MARKET_EXPLAINERS[market])

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

        default_start, default_end = _default_window()
        latest = (pd.Timestamp("today").normalize() - pd.Timedelta(days=1)).date()
        window = st.date_input(
            "Historical window",
            value=(default_start, default_end),
            max_value=latest,
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

    if view == "Compare zones":
        _zone_comparison(start, end, market, battery, credentials_ready)
        return

    try:
        with st.spinner("Loading prices..."):
            prices = _prices_cached(zone, start, end, market, credentials_ready)
    except Exception as exc:  # noqa: BLE001 - surface any data problem in the UI
        st.error(f"Could not load prices for {label_for(zone)} {start}..{end}: {exc}")
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
        f"{label_for(zone)} · {_market_name(market)} prices · "
        f"{len(prices):,} intervals · {span} · "
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
        f"{battery.capacity_kwh:,.1f} kWh / {battery.power_kw:,.0f} kW.",
        has_cost=bool(battery.installed_cost_usd),
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
        "The rule-based and forecast strategies trade only on information they "
        "could actually have had: the rule reacts to recent prices, the forecast "
        "plans each day from earlier days' price shape. Perfect foresight sees "
        "the whole future and is unreachable."
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
            f"Optimized ERCOT real-time (RTM) arbitrage is reported around "
            f"${BENCHMARK_LOW:.0f}-${BENCHMARK_HIGH:.0f} per kW-year "
            f"(Intertrust). A single zone, a simple threshold rule, and no "
            f"demand charges or fleet coordination is well short of that."
        )

    _forecast_section(prices, result, battery)

    st.subheader("Buying low, selling high")
    st.plotly_chart(
        dispatch_figure(result, f"{label_for(zone)} dispatch"),
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


def _zone_comparison(
    start: str,
    end: str,
    market: str,
    battery,
    credentials_ready: bool,
) -> None:
    """Compare the same battery across every load zone that has data."""
    st.subheader("Same battery, every zone")
    include_hubs = st.checkbox(
        "Include hubs as well as load zones", value=False,
        help="Hubs average across a wider footprint and are not a residential option.",
    )
    targets = list(LOAD_ZONES) + list(HUBS) if include_hubs else list(LOAD_ZONES)
    st.caption(
        f"{_market_name(market)} prices · {start} to {end} · {battery.name}. "
        "Locations with no "
        "cached prices for this window are skipped."
    )

    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    progress = st.progress(0.0, text="Loading zones...")
    for i, candidate in enumerate(targets, start=1):
        try:
            frames[candidate] = _prices_cached(
                candidate, start, end, market, credentials_ready
            )
        except Exception:  # noqa: BLE001 - a missing zone must not break the rest
            skipped.append(candidate)
        progress.progress(i / len(targets), text=f"Loaded {label_for(candidate)}")

    if not frames:
        st.error(
            f"No location had cached {_market_name(market)} prices for {start}..{end}. "
            "Try a window that has been fetched, or pull it with the CLI."
        )
        return

    with st.spinner("Backtesting each zone..."):
        comparison = compare_zones(frames, battery, ThresholdStrategy())

    best = comparison.index[0]
    st.success(
        f"**{label_for(best)}** is the strongest zone in this window at "
        f"{_money(comparison.iloc[0]['usd_per_kw_year'], 2)}/kW-year."
    )

    st.plotly_chart(zone_comparison_figure(comparison), width='stretch')

    table = comparison[
        [
            "net_usd",
            "net_after_degradation_usd",
            "usd_per_kw_year",
            "equivalent_full_cycles",
            "mean_price_usd_per_mwh",
            "p95_price_usd_per_mwh",
            "intervals",
            "days",
        ]
    ].round(2)
    table.insert(0, "location", [label_for(code) for code in table.index])
    table.index.name = "settlement_point"
    st.dataframe(table, width='stretch')

    if skipped:
        st.caption(
            "No cached prices for: "
            + ", ".join(label_for(code) for code in skipped)
            + ". Fetch them with `wattson fetch` to include them here."
        )
    st.caption(
        "Annualized from this historical window only. Not a forecast, and it "
        "assumes a battery that cycles freely rather than one held back for "
        "backup or resilience."
    )


def _forecast_section(prices: pd.DataFrame, result, battery) -> None:
    """Forward look: the day after this window, planned against a forecast."""
    series = prepare_price_series(prices, result.market, result.settlement_point)
    st.subheader("Looking ahead: the next day")
    try:
        plan = plan_next_day(series, battery)
    except (ValueError, RuntimeError) as exc:
        st.info(f"Not enough history in this window to forecast from ({exc}).")
        return
    accuracy = forecast_accuracy(series)

    cols = st.columns(3)
    cols[0].metric(
        f"Planned for {plan.date:%b %d}",
        _money(plan.expected_net_after_wear_usd, 2),
        help="Energy revenue the plan expects if prices follow the forecast, "
             "after battery wear. Real prices will differ.",
    )
    if accuracy.intervals:
        cols[1].metric(
            "Typical forecast miss",
            f"${accuracy.mae_usd_per_mwh:,.2f}/MWh",
            help=f"Mean absolute error over {accuracy.days} days of this window, "
                 "each day forecast from earlier days only.",
        )
        cols[2].metric(
            "vs. 'same as yesterday'",
            f"{accuracy.skill:+.0%}",
            help="Share of the naive forecast's error removed. Positive means "
                 "the forecast beat assuming each day repeats the one before.",
        )
    st.plotly_chart(forecast_plan_figure(plan, battery), width='stretch')
    st.caption(
        "The forecast is the median price at each time of day over the "
        "previous 7 days of data, so it captures the usual daily shape but "
        "cannot anticipate a price spike. The 'forecast' row in the table "
        "above shows what trading on this forecast every day of the window "
        "would actually have earned at real prices."
    )

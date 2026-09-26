"""Wattson's Streamlit front end.

Written for someone who has never heard of ERCOT: plain words, prices in cents
per kWh (the unit on a home bill), earnings in dollars for the chosen battery,
and one idea per chart. Industry units and raw tables live in expanders.

Every figure replays historical prices and says so. Nothing here is a promise
about future earnings.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import pandas as pd
import streamlit as st

from wattson.config import CENTRAL_TIME, MAX_BASE_CORES, BatteryConfig, base_cores
from wattson.data.cache import load_range
from wattson.data.ercot_source import ErcotLiveProvider, load_keys_file
from wattson.data.providers import PriceRequest
from wattson.metrics import BASELINE_LABEL, compare_strategies, compare_zones, compute_metrics
from wattson.forecast import forecast_accuracy, plan_next_day
from wattson.parallel import (
    PARALLEL_MIN_INTERVALS,
    WORKER_FAILURES,
    backtest_job,
    default_workers,
    optimal_job,
    pool,
)
from wattson.sim.engine import prepare_price_series, run_backtest
from wattson.strategies.forecast import ForecastStrategy
from wattson.strategies.threshold import ThresholdStrategy
from wattson.viz import (
    daily_earnings_figure,
    day_in_the_life_figure,
    dispatch_figure,
    forecast_plan_figure,
    strategy_figure,
    zone_comparison_figure,
)
from wattson.zones import HUBS, LOAD_ZONES, label_for, short_name


# Intertrust's published range for optimized pure RTM arbitrage, used only as
# a reference point for what a well-run commercial fleet actually achieves.
BENCHMARK_LOW = 55.0
BENCHMARK_HIGH = 66.0

_NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _cores_phrase(count: int) -> str:
    """How the battery reads inside a sentence: 'a Base Core', 'two Base Cores'."""
    return "a Base Core" if count == 1 else f"{_NUMBER_WORDS.get(count, count)} Base Cores"


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
# Date presets. Every preset runs up to today (Texas time).
RANGE_PRESETS: dict[str, tuple[str, int] | None] = {
    "Last 7 days": ("days", 7),
    "Last month": ("months", 1),
    "Last 2 months": ("months", 2),
    "Last 3 months": ("months", 3),
    "Last 6 months": ("months", 6),
    "Last year": ("months", 12),
    "Custom dates": None,
}
DEFAULT_RANGE = "Last month"
FULL_DETAIL_MAX_DAYS = 62


def _today() -> dt.date:
    """Today in Texas. The server may run on UTC, which is ahead in the evening."""
    return pd.Timestamp.now(tz=CENTRAL_TIME).date()


def _preset_window(choice: str, today: dt.date) -> tuple[dt.date, dt.date]:
    """Start and end dates for a preset, ending today."""
    kind, n = RANGE_PRESETS[choice]
    if kind == "days":
        return today - dt.timedelta(days=n - 1), today
    start = (pd.Timestamp(today) - pd.DateOffset(months=n) + pd.Timedelta(days=1)).date()
    return start, today


def _load_credentials() -> bool:
    try:
        load_keys_file("ERCOT API Keys.txt")
        return True
    except (RuntimeError, OSError):
        return False


@st.cache_data(show_spinner=False, ttl=600)
def _prices_cached(
    zone: str, start: str, end: str, market: str, ready: bool
) -> pd.DataFrame:
    """Prices for a date range: saved months from disk, today live.

    Cached for 10 minutes so today's prices stay reasonably current. What
    could not be loaded rides along in ``frame.attrs`` for the page to explain.
    """
    fetch = (lambda request: ErcotLiveProvider().fetch(request)) if ready else None
    result = load_range(
        zone,
        market,
        dt.date.fromisoformat(start),
        dt.date.fromisoformat(end),
        today=_today(),
        fetch=fetch,
    )
    if result.frame.empty:
        raise RuntimeError(
            f"no cached {_market_name(market)} prices for {label_for(zone)} "
            f"covering {start}..{end}, "
            + ("and no API credentials to fetch them" if not ready else "and ERCOT returned none")
        )
    frame = result.frame.copy()
    frame.attrs.update(
        missing_days=list(result.missing_days),
        includes_today=result.includes_today,
        today_error=result.today_error,
    )
    return frame


@st.cache_data(show_spinner=False, ttl=600)
def _run_cached(zone: str, start: str, end: str, market: str, battery, ready: bool):
    """Everything the single-area page computes, cached so moving the day
    slider does not rerun a year of backtests."""
    prices = _prices_cached(zone, start, end, market, ready)
    point = str(prices["settlement_point"].iloc[0])
    online = optimal = None
    if len(prices) >= PARALLEL_MIN_INTERVALS:
        # Long windows: run the planner, the simple rule and perfect hindsight
        # at the same time instead of one after another.
        try:
            with pool(3) as ex:
                planned = ex.submit(backtest_job, (prices, battery, ForecastStrategy()))
                rule = ex.submit(backtest_job, (prices, battery, ThresholdStrategy()))
                best = ex.submit(optimal_job, (prices, battery, market, point))
                result, online, optimal = planned.result(), rule.result(), best.result()
        except WORKER_FAILURES:
            result, online, optimal = run_backtest(prices, battery, ForecastStrategy()), None, None
    else:
        result = run_backtest(prices, battery, ForecastStrategy())
    metrics = compute_metrics(result)
    comparison = compare_strategies(
        prices, battery, ThresholdStrategy(),
        online_result=online, forecast_result=result, optimal_solution=optimal,
    )
    series = prepare_price_series(prices, result.market, result.settlement_point)
    try:
        plan, plan_error = plan_next_day(series, battery), None
    except (ValueError, RuntimeError) as exc:
        plan, plan_error = None, str(exc)
    accuracy = forecast_accuracy(series)
    return prices, result, metrics, comparison, plan, plan_error, accuracy


@st.cache_data(show_spinner=False, ttl=600)
def _compare_cached(targets: tuple[str, ...], start: str, end: str, market: str, battery, ready: bool):
    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for code in targets:
        try:
            frames[code] = _prices_cached(code, start, end, market, ready)
        except Exception:  # noqa: BLE001 - a missing area must not break the rest
            skipped.append(code)
    comparison = (
        compare_zones(frames, battery, ForecastStrategy(), workers=default_workers())
        if frames
        else None
    )
    span_frame = next(iter(frames.values())) if frames else None
    return comparison, skipped, span_frame


def _date_ranges(days) -> str:
    """Collapse dates into readable runs: 'Sep 21–25, Sep 28'."""
    days = sorted(days)
    runs, run = [], [days[0]] if days else []
    for d in days[1:]:
        if (d - run[-1]).days == 1:
            run.append(d)
        else:
            runs.append(run)
            run = [d]
    if run:
        runs.append(run)
    def fmt(r):
        if len(r) == 1:
            return f"{r[0]:%b %-d}"
        tail = f"{r[-1]:%-d}" if r[-1].month == r[0].month else f"{r[-1]:%b %-d}"
        return f"{r[0]:%b %-d}–{tail}"

    return ", ".join(fmt(r) for r in runs)


def _data_notes(prices: pd.DataFrame, end: dt.date, today: dt.date) -> None:
    """Say plainly when part of the requested period could not be loaded."""
    missing = prices.attrs.get("missing_days") or []
    if missing:
        st.warning(
            f"No prices for {len(missing)} day{'s' if len(missing) != 1 else ''} in "
            f"this period ({_date_ranges(missing)}). You may be offline, or ERCOT "
            "hasn't published them. The numbers cover the days that loaded."
        )
    if end >= today and not prices.attrs.get("includes_today", False):
        last = prices["interval_start"].iloc[-1]
        if prices.attrs.get("today_error") == "offline":
            reason = "Today's prices need an internet connection"
        else:
            reason = "Today's prices couldn't be loaded"
        st.caption(f"{reason}, so this runs through {last:%b %-d}.")


def _md(text: str) -> str:
    """Escape dollar signs for Streamlit markdown.

    Streamlit reads text between two ``$`` as a LaTeX formula, which silently
    eats both dollar signs and mangles everything between them. Anything
    rendered as markdown (success, error, caption, info) goes through here.
    """
    return text.replace("$", r"\$")


def _money(value: float, decimals: int = 0) -> str:
    """Format money so losses read as -$1 rather than $-1."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def _cents(usd_per_mwh: float) -> str:
    """$/MWh as cents per kWh, the unit people see on their electricity bill."""
    return f"{usd_per_mwh / 10:.1f}¢"


def _hour(ts) -> str:
    return f"{ts:%-I %p}"


STRATEGY_NAMES = {
    BASELINE_LABEL: "No battery",
    "threshold": "Simple rule",
    "forecast": "Forecast planner (what Wattson uses)",
    "perfect foresight (upper bound)": "Perfect hindsight (impossible in practice)",
}

DEFAULT_WEAR_CENTS = 1.2  # cents per kWh in or out; an estimate, Base doesn't publish one

# Base's fees for a Core in Texas, the only grid Wattson covers.
BASE_MONTHLY_FEE = 19.0
BASE_INSTALL_FEE = 695.0
FEE_SOURCE = (
    "Base's fees in Texas: $695 to install and $19 a month "
    "(basepowercompany.com/pricing)."
)


def fee_coverage(per_year: float, monthly_fee: float, install_fee: float) -> str:
    """How much of Base's plan fees the battery's grid trading would pay for.

    The fees are what the homeowner pays Base; the trading earnings go to Base.
    Setting one against the other shows how far trading alone goes toward the
    plan, from Base's side of the ledger.
    """
    monthly_fee_text = _money(monthly_fee, 2 if monthly_fee % 1 else 0)
    per_month = per_year / 12
    if per_year <= 0:
        return (
            "At this pace, the battery's grid trading wouldn't pay for any of the "
            f"{monthly_fee_text} monthly fee."
        )
    share = per_year / (12 * monthly_fee)
    if share >= 1:
        text = (
            "At this pace, the battery's grid trading would pay for all of the "
            f"{monthly_fee_text} monthly fee, with about {_money(per_month - monthly_fee, 2)} "
            "a month to spare."
        )
    else:
        text = (
            "At this pace, the battery's grid trading would pay for about "
            f"{share:.0%} of the {monthly_fee_text} monthly fee: about "
            f"{_money(per_month, 2)} of it each month."
        )
    if install_fee > 0:
        first_year = install_fee + 12 * monthly_fee
        text += (
            f" Counting the {_money(install_fee)} install fee, it would cover about "
            f"{min(per_year / first_year, 1):.0%} of the first year's {_money(first_year)}."
        )
    return text


def _wear_explainer(cents: float) -> str:
    return (
        "Every time a battery charges and discharges, its lithium cells age "
        "slightly and it permanently loses a little of the energy it can hold. "
        f"Eventually it has to be replaced. Wattson counts {cents:.1f}¢ for every "
        "kWh that goes in or out as that trade's share of the replacement."
    )


OWNER_NOTE = (
    "Base owns and maintains its batteries, so battery wear is Base's cost, not "
    "the homeowner's. \"Left after wear\" is what the battery's buying and "
    "selling is worth once that cost is counted."
)

ZONE_HELP = (
    "The Texas grid is split into pricing areas. Yours depends on which "
    "company delivers your electricity, not just your address. For example, "
    "Austin Energy customers are in the Austin Energy area, but many homes "
    "around Austin are priced as South Texas."
)

GLOSSARY = {
    "ERCOT": "The organization that runs most of the Texas power grid and "
             "publishes the wholesale price of electricity.",
    "Pricing area (load zone)": "ERCOT sets a separate price for each part of "
             "Texas. Which one applies to a home depends on its electric "
             "utility.",
    "Real-time and day-ahead prices": "Real-time prices are what power "
             "actually sold for, every 15 minutes. Day-ahead prices are "
             "agreed the day before, hour by hour.",
    "kWh and kW": "A kWh (kilowatt-hour) is an amount of energy: how much the "
             "battery holds. A kW (kilowatt) is a rate: how fast it can charge "
             "or discharge. A Base Core holds 39.2 kWh and moves up to 11 kW.",
    "Cents per kWh": "The unit on a home electricity bill. Wholesale prices "
             "are usually a few cents, but can jump to dollars during a spike.",
    "Battery wear": "The slow, permanent loss of capacity as a battery's "
             "lithium cells age with use, which eventually means replacing it. "
             "Wattson counts it as a cost on every kWh in or out, so a trade only "
             "happens when the price gap is bigger than the wear. Base carries "
             "this cost, not the homeowner. It is not the same as the battery "
             "running down during use, or energy lost as heat, which is counted "
             "separately.",
    "Trading hub": "A regional average price that energy traders use. Homes "
             "are not billed at hub prices.",
}


def _how_it_works() -> None:
    cols = st.columns(3)
    cols[0].markdown(
        "**1. Power prices change all day.** In Texas, the wholesale price of "
        "electricity resets every 15 minutes. It's usually cheap overnight and "
        "can spike on hot afternoons."
    )
    cols[1].markdown(
        "**2. A battery can buy low and sell high.** It charges when power is "
        "cheap and sends it back to the grid when power is expensive. The "
        "difference is what it earns."
    )
    cols[2].markdown(
        "**3. Wattson replays real prices.** It runs a Base battery through "
        "real past prices from the Texas grid to show what it would have earned."
    )
    st.divider()


def main() -> None:
    st.set_page_config(page_title="Wattson", layout="wide")
    st.title("Wattson")
    st.header("What could a Base battery earn in your part of Texas?")
    _how_it_works()

    credentials_ready = _load_credentials()

    with st.sidebar:
        st.header("Choose what to look at")
        view = st.radio("Show", ["One area", "Compare all of Texas"], horizontal=True)
        zone = st.selectbox(
            "Where in Texas?",
            LOAD_ZONES,
            index=LOAD_ZONES.index(DEFAULT_ZONE),
            format_func=label_for,
            help=ZONE_HELP,
        )
        count = int(st.number_input(
            "How many Base Cores?",
            min_value=1,
            max_value=MAX_BASE_CORES,
            value=1,
            step=1,
            help="Each Base Core holds 39.2 kWh and charges or discharges at up "
                 "to 11 kW. Base installs one or two per home.",
        ))
        if count > 1:
            st.caption(
                f"{count} Cores hold {39.2 * count:g} kWh. Every Core sees the same "
                f"prices, so {count} Cores earn {count} times as much as one."
            )
        if count > 2:
            st.caption("Base installs one or two Cores per home; more is shown for comparison.")
        preset = base_cores(count)

        today = _today()
        range_choice = st.selectbox(
            "Dates to replay",
            list(RANGE_PRESETS),
            index=list(RANGE_PRESETS).index(DEFAULT_RANGE),
            help="Wattson replays real Texas grid prices from these dates. "
                 "Presets run up to today; today's prices fill in as the day goes on.",
        )
        if RANGE_PRESETS[range_choice] is None:
            window = st.date_input(
                "From and to",
                value=(today - dt.timedelta(days=29), today),
                max_value=today,
            )
        else:
            window = _preset_window(range_choice, today)
            st.caption(f"{window[0]:%b %-d, %Y} to today ({window[1]:%b %-d})")
            if RANGE_PRESETS[range_choice][1] >= 6 and RANGE_PRESETS[range_choice][0] == "months":
                st.caption("Long periods take a little longer to load the first time.")

        with st.expander("More options"):
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
            wear_cents = st.slider(
                "Battery wear cost (¢ per kWh in or out)",
                0.0, 5.0, DEFAULT_WEAR_CENTS, 0.1,
                help="How much each kWh the battery moves ages it, as a share of "
                     "an eventual replacement. Base doesn't publish this; "
                     f"{DEFAULT_WEAR_CENTS}¢ is Wattson's estimate for a lithium "
                     "iron phosphate battery like Base Core. Results depend "
                     "heavily on it.",
            )
            if abs(wear_cents - DEFAULT_WEAR_CENTS) > 1e-9:
                st.caption(
                    f"Using {wear_cents:.1f}¢ per kWh for battery wear instead of "
                    f"Wattson's {DEFAULT_WEAR_CENTS}¢ estimate."
                )
            custom = st.checkbox("Try a custom battery size", value=False)
            if custom:
                capacity = st.slider("How much it holds (kWh)", 5.0, 200.0, float(preset.capacity_kwh), 0.1)
                power = st.slider("How fast it charges (kW)", 1.0, 50.0, float(preset.power_kw), 0.5)
                efficiency = st.slider(
                    "Energy kept after a charge and discharge", 0.80, 0.98,
                    float(preset.round_trip_efficiency), 0.01,
                )
                installed = st.number_input(
                    "Purchase price in dollars (optional, for payback)",
                    0.0, 200_000.0, float(preset.installed_cost_usd or 0.0), 500.0,
                )
                battery = replace(
                    preset,
                    name=f"custom_{capacity:g}kwh",
                    capacity_kwh=capacity,
                    power_kw=power,
                    round_trip_efficiency=efficiency,
                    installed_cost_usd=installed or None,
                )
            else:
                battery = preset
            battery = replace(battery, degradation_cost_per_kwh=wear_cents / 100)
        if not credentials_ready:
            st.warning(
                "Offline: showing saved prices only. Add ERCOT API credentials "
                "to load new dates."
            )

    battery_name = "a custom battery" if custom else _cores_phrase(count)

    if len(window) != 2:
        st.info("Pick a start and an end date to replay.")
        return
    start, end = window[0].isoformat(), window[1].isoformat()
    end_date = window[1]

    if view == "Compare all of Texas":
        _zone_comparison(start, end, market, battery, battery_name, credentials_ready)
        return

    try:
        with st.spinner("Loading prices and running the battery..."):
            prices, result, metrics, comparison, plan, plan_error, accuracy = _run_cached(
                zone, start, end, market, battery, credentials_ready
            )
    except Exception as exc:  # noqa: BLE001 - surface any data problem in the UI
        st.error(_md(f"Couldn't load prices for {label_for(zone)} from {start} to {end}: {exc}"))
        return

    per_year = _headline(prices, metrics, battery, battery_name, zone, market)
    _fee_section(per_year, count, custom)
    _data_notes(prices, end_date, _today())
    _day_in_the_life(result)
    _day_by_day(result)
    _strategy_section(comparison)
    _forecast_section(plan, plan_error, accuracy, battery)
    _limits()
    _details(prices, result, metrics, comparison, zone, market)
    _glossary()


def _span(prices: pd.DataFrame) -> str:
    # The last interval ends at midnight after the final day, so date the
    # window by interval starts to avoid showing one day too many.
    first_day = prices["interval_start"].iloc[0]
    last_day = prices["interval_start"].iloc[-1]
    head = f"{first_day:%b %-d}" if first_day.year == last_day.year else f"{first_day:%b %-d, %Y}"
    return f"{head} – {last_day:%b %-d, %Y}"


def _headline(prices, metrics, battery, battery_name: str, zone: str, market: str) -> float:
    span = _span(prices)
    first = prices["interval_start"].iloc[0]
    last = prices["interval_start"].iloc[-1]
    days = (last.date() - first.date()).days + 1
    made = metrics.net_usd
    left = metrics.net_after_degradation_usd
    wear = made - left
    per_year = left / metrics.duration_days * 365 if metrics.duration_days else 0.0
    where = f"in the {short_name(zone)} area"

    if left > 0:
        st.success(_md(
            f"**Over these {days} days ({span}), {battery_name} {where} would "
            f"have earned {_money(made, 2)} by buying power when it was cheap and "
            f"selling it back when it was expensive.** After {_money(wear, 2)} of "
            f"battery wear, that leaves {_money(left, 2)}"
            + ("." if days >= 360 else f", about {_money(per_year)} a year at this pace.")
        ))
    else:
        st.error(_md(
            f"**Over these {days} days ({span}), {battery_name} {where} would "
            f"not have covered its wear.** It would have earned {_money(made, 2)} "
            f"buying low and selling high, but battery wear cost {_money(wear, 2)}, "
            f"leaving {_money(left, 2)}. Prices didn't swing enough to cover the wear."
        ))

    has_cost = bool(battery.installed_cost_usd)
    cols = st.columns(5 if has_cost else 4)
    cols[0].metric(
        "Earned buying low, selling high", _money(made, 2),
        help="What the battery got for the power it sold, minus what it paid "
             "to charge, over these dates.",
    )
    cols[1].metric(
        "Battery wear", _money(-wear, 2),
        help=_wear_explainer(battery.degradation_cost_per_kwh * 100)
             + " Adjust it under More options.",
    )
    cols[2].metric(
        "Left after wear", _money(left, 2),
        help="What the battery's buying and selling is worth once wear is "
             "counted. Base carries the wear, not the homeowner.",
    )
    cols[3].metric(
        "Per year at this pace", _money(per_year),
        help="What's left after wear, scaled up to a full year. Prices change a lot from "
             "month to month, so treat this as a rough guide, not a promise.",
    )
    if has_cost:
        years = battery.installed_cost_usd / per_year if per_year > 0 else None
        cols[4].metric(
            "Pays for itself in",
            f"{years:,.0f} years" if years else "Not at this pace",
            help="Purchase price divided by what's left after wear each year.",
        )
    st.caption(OWNER_NOTE)

    p = prices["price_usd_per_mwh"]
    latest = (
        f" Today's prices are included up to {last:%-I:%M %p}."
        if prices.attrs.get("includes_today")
        else ""
    )
    st.caption(_md(
        f"{label_for(zone)} · {_market_name(market)} prices from {span} ranged "
        f"from {_cents(p.min())} to {_cents(p.max())} per kWh.{latest}"
    ))
    return per_year


def _fee_section(per_year: float, count: int, custom: bool) -> None:
    """Base's side of the ledger: how far grid trading goes toward the plan fees."""
    st.markdown(_md(f"**{fee_coverage(per_year, BASE_MONTHLY_FEE, BASE_INSTALL_FEE)}**"))
    note = FEE_SOURCE
    if count > 1 and not custom:
        note += " The fees shown are for one Core; plans with two may differ."
    note += (
        " Grid trading is only part of what a Base battery is worth: backup, the "
        "electricity plan and other grid services aren't included here."
    )
    st.caption(_md(note))


def _day_in_the_life(result) -> None:
    st.subheader("A day in the life")
    st.markdown("Pick a day to see when the battery bought power and when it sold it.")
    ledger = result.ledger
    dates = ledger["interval_start"].dt.date
    daily = ledger.groupby(dates)["net_usd"].sum()
    options = list(daily.index)
    best = daily.idxmax()
    day = (
        st.select_slider(
            "Day", options=options, value=best, format_func=lambda d: f"{d:%a %b %-d}"
        )
        if len(options) > 1
        else options[0]
    )
    one = ledger[dates == day]
    st.plotly_chart(day_in_the_life_figure(one, result.battery), width="stretch")

    bought = one["charged_kwh"].sum()
    sold = one["discharged_kwh"].sum()
    made = one["net_usd"].sum()
    label = f"{day:%A, %B %-d}" + (" (the best day in this period)" if day == best else "")
    if bought > 0 and sold > 0:
        buy_price = (one["price_usd_per_mwh"] * one["charged_kwh"]).sum() / bought
        sell_price = (one["price_usd_per_mwh"] * one["discharged_kwh"]).sum() / sold
        text = (
            f"On {label}, it bought power at an average of {_cents(buy_price)} per "
            f"kWh and sold it at {_cents(sell_price)}, making {_money(made, 2)} "
            "before battery wear."
        )
    elif day == options[0] and bought == 0 and sold == 0:
        text = (
            f"On {label}, the planner had no past prices to learn from yet, so "
            "the battery waited. Pick a later day."
        )
    elif bought == 0 and sold == 0:
        text = (
            f"On {label}, prices didn't swing enough to be worth the battery "
            "wear, so the battery sat still."
        )
    else:
        action = "only charged, saving the energy for a later day" if bought else "only sold energy it had stored earlier"
        text = f"On {label}, the battery {action}."
    st.caption(_md(text))


def _day_by_day(result) -> None:
    st.subheader("What it made each day")
    st.plotly_chart(daily_earnings_figure(result), width="stretch")
    daily = result.ledger.set_index("interval_start")["net_usd"].resample("D").sum()
    total = daily.sum()
    text = "Green days made money, red days lost a little. Figures are before battery wear."
    if total > 0 and len(daily) >= 5:
        n = len(daily)
        share = daily.nlargest(3).sum() / total
        even = 3 / n  # what the best 3 days would make if every day earned the same
        if share >= 0.5:
            text += (
                f" The best 3 of {n} days made {share:.0%} of the total: most of the "
                "money comes from a few days when prices spiked."
            )
        elif share >= 2 * even:
            text += (
                f" The best 3 of {n} days made {share:.0%} of the total, far more than "
                f"their {even:.0%} share if every day earned the same: a few price "
                "spikes do a lot of the work."
            )
        else:
            text += f" The best 3 of {n} days made {share:.0%} of the total, so earnings were fairly even."
    st.caption(_md(text))


def _strategy_section(comparison: pd.DataFrame) -> None:
    st.subheader("How smart does the battery need to be?")
    st.markdown("The same battery over the same dates, run four ways. Bars show what each would have left after battery wear.")
    left = comparison["net_after_degradation_usd"]
    rows = [(STRATEGY_NAMES[label], float(left[label])) for label in comparison.index if label in STRATEGY_NAMES]
    st.plotly_chart(strategy_figure(rows), width="stretch")
    st.markdown(_md(
        "- **No battery**: nothing to buy or sell, so $0.\n"
        "- **Simple rule**: buy when power is cheaper than it has been over the "
        "past day, sell when it's pricier.\n"
        "- **Forecast planner**: each morning, predict the day's prices from the "
        "past week and plan the best times to buy and sell. The rest of this "
        "page uses this one.\n"
        "- **Perfect hindsight**: the most possible if you knew every price in "
        "advance. No one can, so it's a yardstick, not a target."
    ))


def _forecast_section(plan, plan_error, accuracy, battery) -> None:
    """Forward look: the day after this window, planned against a forecast."""
    st.subheader("What tomorrow might look like")
    if plan is None:
        st.info(_md(f"Not enough past prices here to make a forecast ({plan_error})."))
        return

    frame = plan.to_frame()
    hourly = frame.groupby(pd.DatetimeIndex(frame["interval_start"]).hour)["forecast_usd_per_mwh"].mean()
    cheap, peak = _hour(pd.Timestamp(2000, 1, 1, int(hourly.idxmin()))), _hour(pd.Timestamp(2000, 1, 1, int(hourly.idxmax())))
    left = plan.expected_net_after_wear_usd
    trades = plan.charge_kw.sum() > 1e-6
    text = (
        f"Based on the past week, Wattson expects power on {plan.date:%A, %B %-d} "
        f"to be cheapest around {cheap} and most expensive around {peak}. "
    )
    text += (
        f"Its plan: buy when it's cheap and sell into the peak, leaving about "
        f"{_money(left, 2)} after wear if prices behave as expected."
        if trades
        else "The expected price swing is too small to cover battery wear, so the plan is to sit still."
    )
    st.markdown(_md(text))
    st.plotly_chart(forecast_plan_figure(plan, battery), width="stretch")

    if accuracy.intervals:
        better = accuracy.skill >= 0
        st.caption(_md(
            f"How much to trust this: over the last {accuracy.days} days, the "
            f"forecast was typically off by {accuracy.mae_usd_per_mwh / 10:.1f}¢ "
            f"per kWh, {abs(accuracy.skill):.0%} {'better' if better else 'worse'} "
            "than just assuming each day repeats the day before. It learns the "
            "usual daily pattern but can't see sudden price spikes coming."
        ))


def _limits() -> None:
    st.info(
        "**What this leaves out.** These figures replay past prices. They aren't "
        "a promise about the future, and prices vary a lot from month to month. "
        "They also count only buying and selling power. Backup during outages, "
        "lower bills, and the grid-balancing Base does with its whole fleet "
        "aren't included, and those are the main reasons people get a Base battery."
    )


def _details(prices, result, metrics, comparison, zone: str, market: str) -> None:
    with st.expander("Show the detailed numbers"):
        table = comparison.rename(index=STRATEGY_NAMES)[
            ["net_usd", "net_after_degradation_usd", "usd_per_kw_year", "equivalent_full_cycles", "capture_ratio"]
        ].rename(
            columns={
                "net_usd": "Made ($)",
                "net_after_degradation_usd": "Left after wear ($)",
                "usd_per_kw_year": "$ per kW per year",
                "equivalent_full_cycles": "Full charge cycles",
                "capture_ratio": "Share of perfect hindsight",
            }
        )
        table.index.name = "How it was run"
        st.dataframe(
            table.style.format(
                {
                    "Made ($)": "${:,.2f}",
                    "Left after wear ($)": "${:,.2f}",
                    "$ per kW per year": "${:,.2f}",
                    "Full charge cycles": "{:,.1f}",
                    "Share of perfect hindsight": "{:.0%}",
                },
                na_rep="",
            ),
            width="stretch",
        )
        st.caption(_md(
            "Industry comparison: professionally run grid batteries doing "
            f"real-time trading in Texas have been reported at about ${BENCHMARK_LOW:.0f}-"
            f"${BENCHMARK_HIGH:.0f} per kW per year (Intertrust). "
            f"The forecast planner here: {_money(metrics.usd_per_kw_year, 2)} per kW per year."
        ))

        first = prices["interval_start"].iloc[0]
        last = prices["interval_start"].iloc[-1]
        if (last - first).days <= FULL_DETAIL_MAX_DAYS:
            st.markdown("**Every 15 minutes of the period**")
            st.plotly_chart(dispatch_figure(result, f"{label_for(zone)}"), width="stretch")
        else:
            st.caption("The every-15-minutes chart is shown for periods up to two months.")

        p = prices["price_usd_per_mwh"]
        st.caption(_md(
            f"{label_for(zone)} · {_market_name(market)} · {len(prices):,} price "
            f"intervals · {_money(p.min(), 2)} to {_money(p.max(), 2)} per MWh · "
            f"{(p < 0).sum():,} intervals with negative prices"
        ))
        ledger = result.ledger
        st.dataframe(
            pd.DataFrame(
                {
                    "Time": ledger["interval_start"],
                    "Price (¢ per kWh)": (ledger["price_usd_per_mwh"] / 10).round(2),
                    "Charging (kW)": ledger["charge_kw"].round(2),
                    "Selling (kW)": ledger["discharge_kw"].round(2),
                    "Battery level (%)": (100 * ledger["soc_fraction"]).round(1),
                    "Money ($)": ledger["net_usd"].round(3),
                }
            ),
            width="stretch",
            height=360,
        )


def _glossary() -> None:
    with st.expander("What do these words mean?"):
        st.markdown(_md("\n".join(f"- **{term}**: {meaning}" for term, meaning in GLOSSARY.items())))


def _zone_comparison(
    start: str,
    end: str,
    market: str,
    battery,
    battery_name: str,
    credentials_ready: bool,
) -> None:
    """The same battery and dates in every pricing area."""
    st.subheader("Which part of Texas pays best?")
    st.markdown(_md(
        f"{battery_name[0].upper() + battery_name[1:]}, run over the same dates "
        "in every pricing area. The only difference is how much prices swing in "
        "each place."
    ))
    include_hubs = st.checkbox(
        "Also show trading hubs", value=False,
        help="Hubs are wider regional price averages that energy traders use. "
             "Homes aren't billed at hub prices.",
    )
    targets = tuple(LOAD_ZONES) + (tuple(HUBS) if include_hubs else ())

    with st.spinner("Running the battery in every area. Long periods take a minute the first time..."):
        comparison, skipped, span_frame = _compare_cached(
            targets, start, end, market, battery, credentials_ready
        )

    if comparison is None:
        st.error(_md(
            f"No saved {_market_name(market)} prices for {start} to {end}. "
            "Try other dates, or connect to the internet to load them."
        ))
        return

    annual = comparison["usd_per_kw_year"] * battery.power_kw
    monthly_fee = BASE_MONTHLY_FEE
    best = annual.idxmax()
    if annual[best] > 0:
        share = annual[best] / (12 * monthly_fee)
        fee_share = (
            f", more than enough to cover Base's {_money(monthly_fee)} monthly fee"
            if share >= 1
            else f", enough for about {share:.0%} of Base's {_money(monthly_fee)} monthly fee"
        )
        st.success(_md(
            f"**{short_name(best)} comes out on top**: {battery_name} there "
            f"would have earned about {_money(annual[best])} a year after wear, at "
            f"this pace{fee_share}."
        ))
    else:
        st.error(_md(
            f"**No area made money after battery wear over these dates.** "
            f"{short_name(best)} lost the least."
        ))

    st.plotly_chart(zone_comparison_figure(comparison, battery), width="stretch")

    span = _span(span_frame)
    note = f"{_market_name(market)} prices, {span}. Each bar is what {battery_name} would earn per year after battery wear, at the pace of these dates."
    if skipped:
        note += " No saved prices for: " + ", ".join(label_for(code) for code in skipped) + "."
    st.caption(_md(note))

    with st.expander("Show the detailed numbers"):
        table = pd.DataFrame(
            {
                "Area": [label_for(code) for code in comparison.index],
                "Made ($)": comparison["net_usd"].round(2),
                "Left after wear ($)": comparison["net_after_degradation_usd"].round(2),
                "Per year ($)": annual.round(0),
                "Share of monthly fee": (annual / (12 * monthly_fee)).clip(lower=0).map("{:.0%}".format),
                "Average price (¢ per kWh)": (comparison["mean_price_usd_per_mwh"] / 10).round(2),
                "Priciest 5% of the time (¢ per kWh)": (comparison["p95_price_usd_per_mwh"] / 10).round(2),
            },
            index=comparison.index,
        )
        table.index.name = "code"
        st.dataframe(table, width="stretch")
    _limits()

"""Performance metrics for a backtest.

Revenue from energy-only arbitrage scales with rated power and cycles per
day, so ``usd_per_kw_year`` is the primary normalization. The per-kWh figure is
reported alongside it for completeness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from arb.config import HOURS_PER_YEAR, BatteryConfig
from arb.sim.engine import BacktestResult, prepare_price_series
from arb.strategies.base import Strategy


@dataclass(frozen=True, slots=True)
class Metrics:
    """Summary of one backtest, all money in USD."""

    settlement_point: str
    market: str
    strategy: str
    battery: str
    start: str
    end: str
    duration_days: float
    duration_years: float
    gross_revenue_usd: float
    charge_cost_usd: float
    net_usd: float
    degradation_cost_usd: float
    net_after_degradation_usd: float
    total_charged_kwh: float
    total_discharged_kwh: float
    equivalent_full_cycles: float
    cycles_per_day: float
    usd_per_kw_year: float
    usd_per_kwh_capacity_year: float
    usd_per_cycle: float
    round_trip_efficiency_realized: float
    avg_charge_price_usd_per_mwh: float
    avg_discharge_price_usd_per_mwh: float
    captured_spread_usd_per_mwh: float
    avg_daily_net_usd: float
    best_day_net_usd: float
    worst_day_net_usd: float
    share_intervals_discharging: float
    curtailed_kwh: float
    capture_ratio: float | None = None
    payback_years: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    result: BacktestResult,
    capture_ratio: float | None = None,
) -> Metrics:
    ledger = result.ledger
    battery = result.battery

    charged = float(ledger["charged_kwh"].sum())
    discharged = float(ledger["discharged_kwh"].sum())
    revenue = float(ledger["revenue_usd"].sum())
    cost = float(ledger["cost_usd"].sum())
    net = revenue - cost

    duration_hours = float(
        (ledger["interval_end"] - ledger["interval_start"]).dt.total_seconds().sum()
        / 3600.0
    )
    duration_days = duration_hours / 24.0
    years = duration_hours / HOURS_PER_YEAR if duration_hours > 0 else 0.0

    degradation = battery.degradation_cost_per_kwh * (charged + discharged)
    net_after_deg = net - degradation

    efc = discharged / battery.capacity_kwh if battery.capacity_kwh else 0.0

    daily = ledger.set_index("interval_start")["net_usd"].resample("D").sum()
    interval_hours = (
        (ledger["interval_end"] - ledger["interval_start"]).dt.total_seconds() / 3600.0
    )
    curtailment_kwh = float(
        (
            (ledger["curtailed_charge_kw"] + ledger["curtailed_discharge_kw"])
            * interval_hours
        ).sum()
    )
    realized_rte = (discharged / charged) if charged > 0 else 0.0

    charging = ledger.loc[ledger["charge_kw"] > 0, "price_usd_per_mwh"]
    discharging = ledger.loc[ledger["discharge_kw"] > 0, "price_usd_per_mwh"]
    avg_charge_price = float(charging.mean()) if len(charging) else 0.0
    avg_discharge_price = float(discharging.mean()) if len(discharging) else 0.0

    annualized = net_after_deg / years if years > 0 else 0.0
    payback = (
        battery.installed_cost_usd / annualized
        if battery.installed_cost_usd and annualized > 0
        else None
    )

    return Metrics(
        settlement_point=result.settlement_point,
        market=result.market,
        strategy=result.strategy_name,
        battery=battery.name,
        start=str(ledger["interval_start"].min()),
        end=str(ledger["interval_end"].max()),
        duration_days=duration_days,
        duration_years=years,
        gross_revenue_usd=revenue,
        charge_cost_usd=cost,
        net_usd=net,
        degradation_cost_usd=degradation,
        net_after_degradation_usd=net_after_deg,
        total_charged_kwh=charged,
        total_discharged_kwh=discharged,
        equivalent_full_cycles=efc,
        cycles_per_day=efc / duration_days if duration_days > 0 else 0.0,
        usd_per_kw_year=annualized / battery.power_kw if battery.power_kw else 0.0,
        usd_per_kwh_capacity_year=(
            annualized / battery.capacity_kwh if battery.capacity_kwh else 0.0
        ),
        usd_per_cycle=net_after_deg / efc if efc > 0 else 0.0,
        round_trip_efficiency_realized=realized_rte,
        avg_charge_price_usd_per_mwh=avg_charge_price,
        avg_discharge_price_usd_per_mwh=avg_discharge_price,
        captured_spread_usd_per_mwh=avg_discharge_price - avg_charge_price,
        avg_daily_net_usd=float(daily.mean()) if len(daily) else 0.0,
        best_day_net_usd=float(daily.max()) if len(daily) else 0.0,
        worst_day_net_usd=float(daily.min()) if len(daily) else 0.0,
        share_intervals_discharging=float((ledger["discharge_kw"] > 0).mean()),
        curtailed_kwh=curtailment_kwh,
        capture_ratio=capture_ratio,
        payback_years=payback,
    )


BASELINE_LABEL = "no battery (baseline)"


def no_battery_baseline(
    battery: BatteryConfig,
    duration_days: float,
    settlement_point: str,
    market: str,
) -> Metrics:
    """The do-nothing case, for comparison against a dispatching battery.

    This model prices energy-only arbitrage, so without a battery a buyer
    pays the retail rate and a seller receives it: there is no spread to
    capture and the incremental value of owning storage is zero. That makes
    the baseline a flat $0, and it is stated explicitly rather than omitted so
    the rule-based and perfect-foresight rows have an honest floor to beat.

    A residential customer whose bill is dominated by demand charges or solar
    self-consumption has a non-zero baseline, but charging that value requires
    modeling their load and retail tariff, which this tool does not do.
    """
    return Metrics(
        settlement_point=settlement_point,
        market=market,
        strategy=BASELINE_LABEL,
        battery=battery.name,
        start="",
        end="",
        duration_days=duration_days,
        duration_years=duration_days / 365.25,
        gross_revenue_usd=0.0,
        charge_cost_usd=0.0,
        net_usd=0.0,
        degradation_cost_usd=0.0,
        net_after_degradation_usd=0.0,
        total_charged_kwh=0.0,
        total_discharged_kwh=0.0,
        equivalent_full_cycles=0.0,
        cycles_per_day=0.0,
        usd_per_kw_year=0.0,
        usd_per_kwh_capacity_year=0.0,
        usd_per_cycle=0.0,
        round_trip_efficiency_realized=0.0,
        avg_charge_price_usd_per_mwh=0.0,
        avg_discharge_price_usd_per_mwh=0.0,
        captured_spread_usd_per_mwh=0.0,
        avg_daily_net_usd=0.0,
        best_day_net_usd=0.0,
        worst_day_net_usd=0.0,
        share_intervals_discharging=0.0,
        curtailed_kwh=0.0,
        capture_ratio=None,
        payback_years=None,
    )


def compare_strategies(
    prices: pd.DataFrame,
    battery: BatteryConfig,
    strategy: Strategy,
    *,
    include_optimal: bool = True,
) -> pd.DataFrame:
    """Backtest one battery across baseline, the online rule, and perfect foresight.

    Returns a frame indexed by label with the headline economics for each, so
    the CLI and the UI report the same comparison.
    """
    from arb.sim.engine import run_backtest
    from arb.strategies.perfect_foresight import solve_perfect_foresight

    result = run_backtest(prices, battery, strategy)
    online = compute_metrics(result)

    baseline = no_battery_baseline(
        battery, online.duration_days, result.settlement_point, result.market
    )
    rows = [
        {
            "strategy": baseline.strategy,
            "kind": "baseline",
            "net_usd": baseline.net_usd,
            "net_after_degradation_usd": baseline.net_after_degradation_usd,
            "usd_per_kw_year": baseline.usd_per_kw_year,
            "equivalent_full_cycles": baseline.equivalent_full_cycles,
            "capture_ratio": baseline.capture_ratio,
            "payback_years": baseline.payback_years,
        }
    ]

    if include_optimal:
        series = prepare_price_series(
            prices, result.market, result.settlement_point
        )
        optimal_net = solve_perfect_foresight(series, battery).net_usd
    else:
        optimal_net = 0.0

    capture = online.net_usd / optimal_net if optimal_net > 0 else None
    online_with_capture = compute_metrics(result, capture_ratio=capture)

    rows.append(
        {
            "strategy": online_with_capture.strategy,
            "kind": "online",
            "net_usd": online_with_capture.net_usd,
            "net_after_degradation_usd": online_with_capture.net_after_degradation_usd,
            "usd_per_kw_year": online_with_capture.usd_per_kw_year,
            "equivalent_full_cycles": online_with_capture.equivalent_full_cycles,
            "capture_ratio": capture,
            "payback_years": online_with_capture.payback_years,
        }
    )

    if include_optimal and optimal_net > 0:
        rows.append(
            {
                "strategy": "perfect foresight (upper bound)",
                "kind": "optimal",
                "net_usd": optimal_net,
                "net_after_degradation_usd": float("nan"),
                "usd_per_kw_year": optimal_net / online_with_capture.duration_years
                / battery.power_kw
                if online_with_capture.duration_years > 0
                else float("nan"),
                "equivalent_full_cycles": float("nan"),
                "capture_ratio": 1.0,
                "payback_years": None,
            }
        )

    return pd.DataFrame(rows).set_index("strategy")

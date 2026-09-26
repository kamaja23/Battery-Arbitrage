"""Performance metrics for a backtest.

Revenue from energy-only arbitrage scales with rated power and cycles per
day, so ``usd_per_kw_year`` is the primary normalization. The per-kWh figure is
reported alongside it for completeness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from arb.config import HOURS_PER_YEAR
from arb.sim.engine import BacktestResult


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
    curtailment_kwh = float(
        (ledger["curtailed_charge_kw"] + ledger["curtailed_discharge_kw"]).sum()
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

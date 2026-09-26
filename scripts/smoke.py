"""Smoke run: exercise the full pipeline without touching the network.

Usage:  python scripts/smoke.py [days]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arb.config import COMMERCIAL_1MW, RESIDENTIAL_13KWH, EngineConfig
from arb.data.providers import PriceRequest, SyntheticProvider
from arb.metrics import compute_metrics
from arb.sim.engine import run_backtest
from arb.strategies.perfect_foresight import run_perfect_foresight_backtest
from arb.strategies.threshold import ThresholdStrategy

INTERTRUST_USD_PER_KW_YEAR = (55.0, 66.0)


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    start = "2026-06-01"
    end = (pd.Timestamp(start) + pd.Timedelta(days=days - 1)).date().isoformat()

    request = PriceRequest(
        settlement_point="LZ_WEST", start_date=start, end_date=end, market="RTM"
    )
    prices = SyntheticProvider(seed=3).fetch(request)
    print(f"Synthetic series: {len(prices)} intervals, {start}..{end}, LZ_WEST")
    print("(synthetic data -- magnitudes are indicative, not real ERCOT results)\n")

    for cfg in (COMMERCIAL_1MW, RESIDENTIAL_13KWH):
        strategy = ThresholdStrategy(lookback_intervals=96, min_spread_usd_per_mwh=5.0)
        result = run_backtest(prices, cfg, strategy, engine=EngineConfig(market="RTM"))
        _, lp_net = run_perfect_foresight_backtest(prices, cfg)

        ratio = result.net_usd / lp_net if lp_net > 0 else None
        m = compute_metrics(result, capture_ratio=ratio)

        print(f"--- {cfg.name} ({cfg.power_kw:g} kW / {cfg.capacity_kwh:g} kWh) ---")
        print(f"  window             {m.duration_days:.1f} days")
        print(f"  net                ${m.net_usd:,.2f}")
        print(f"  after degradation  ${m.net_after_degradation_usd:,.2f}")
        print(f"  $/kW-year          ${m.usd_per_kw_year:,.2f}")
        print(f"  $/kWh-cap-year     ${m.usd_per_kwh_capacity_year:,.2f}")
        print(f"  cycles             {m.equivalent_full_cycles:.1f} ({m.cycles_per_day:.2f}/day)")
        print(f"  realized RTE       {m.round_trip_efficiency_realized:.3f}")
        print(f"  bought at          ${m.avg_charge_price_usd_per_mwh:,.2f}/MWh")
        print(f"  sold at            ${m.avg_discharge_price_usd_per_mwh:,.2f}/MWh")
        print(f"  captured spread    ${m.captured_spread_usd_per_mwh:,.2f}/MWh")
        print(f"  best/worst day     ${m.best_day_net_usd:,.2f} / ${m.worst_day_net_usd:,.2f}")
        print(f"  LP upper bound     ${lp_net:,.2f}")
        print(f"  capture ratio      {ratio:.1%}" if ratio else "  capture ratio      n/a")
        print(f"  payback            {m.payback_years:.1f} yr" if m.payback_years else "  payback            n/a")
        print()

    lo, hi = INTERTRUST_USD_PER_KW_YEAR
    print(
        f"Intertrust 2018-2025 reference for optimized pure RTM arbitrage: "
        f"${lo:.0f}-{hi:.0f}/kW-year"
    )
    print("Our rule-based figure should land meaningfully below that band; the LP")
    print("is the comparable quantity and is expected to sit in or above it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

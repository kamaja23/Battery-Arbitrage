"""Command line entry points.

Exposed as ``arb-backtest``, ``arb-verify-data`` and ``arb-fetch`` once the
package is installed; the scripts under ``scripts/`` are thin wrappers so the
same code runs from a source checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from arb.config import COMMERCIAL_1MW, RESIDENTIAL_13KWH, BatteryConfig
from arb.data.cache import cached_fetch
from arb.data.ercot_source import ErcotLiveProvider, load_keys_file
from arb.data.providers import PriceRequest
from arb.metrics import compare_strategies
from arb.strategies.threshold import ThresholdStrategy

DEFAULT_KEYS = "ERCOT API Keys.txt"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--zone", default="LZ_WEST", help="settlement point")
    parser.add_argument("--start", default="2026-06-01", help="first delivery date")
    parser.add_argument("--end", default="2026-06-30", help="last delivery date")
    parser.add_argument("--market", default="RTM", choices=["RTM", "DAM"])
    parser.add_argument("--keys", default=DEFAULT_KEYS, help="credentials file")


def _load(args: argparse.Namespace) -> pd.DataFrame:
    load_keys_file(args.keys)
    request = PriceRequest(
        settlement_point=args.zone,
        start_date=args.start,
        end_date=args.end,
        market=args.market,
    )
    return cached_fetch(ErcotLiveProvider(), request)


def _describe(prices: pd.DataFrame, args: argparse.Namespace) -> None:
    print(
        f"{args.zone} {args.market} {args.start}..{args.end}: {len(prices)} intervals, "
        f"{prices['interval_start'].iloc[0].date()} to "
        f"{prices['interval_end'].iloc[-1].date()}"
    )
    p = prices["price_usd_per_mwh"]
    print(
        f"  price $/MWh  min={p.min():.2f} p50={p.median():.2f} "
        f"p95={p.quantile(0.95):.2f} max={p.max():.2f}  "
        f"negative intervals={(p < 0).sum()}\n"
    )


def cmd_backtest(args: argparse.Namespace) -> int:
    prices = _load(args)
    _describe(prices, args)

    batteries: tuple[BatteryConfig, ...] = (COMMERCIAL_1MW, RESIDENTIAL_13KWH)
    for battery in batteries:
        frame = compare_strategies(prices, battery, ThresholdStrategy())
        view = frame[
            [
                "net_usd",
                "net_after_degradation_usd",
                "usd_per_kw_year",
                "equivalent_full_cycles",
                "capture_ratio",
                "payback_years",
            ]
        ]
        with pd.option_context("display.width", 200):
            print(f"\n{battery.name}  "
                  f"({battery.capacity_kwh:,.0f} kWh / {battery.power_kw:,.0f} kW)")
            print(view.round(2).to_string())

    print(
        "\nWithout a battery there is no spread to capture, so the baseline is $0.\n"
        "Perfect foresight sees every future price and is unreachable in practice."
    )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    prices = _load(args)
    _describe(prices, args)
    print("  cached; re-running will read from disk without hitting the API")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    load_keys_file(args.keys)
    provider = ErcotLiveProvider()

    cases = [
        ("normal", "RTM", "2026-06-15", 96),
        ("spring-fwd", "RTM", "2026-03-08", 92),
        ("fall-back", "RTM", "2025-11-02", 100),
        ("normal", "DAM", "2026-06-15", 24),
        ("spring-fwd", "DAM", "2026-03-08", 23),
        ("fall-back", "DAM", "2025-11-02", 25),
    ]

    print(f"{'case':<20} {'rows':>5} {'exp':>4} {'span':>7}  window")
    print("-" * 104)
    failures = 0
    for label, market, day, expected in cases:
        request = PriceRequest(
            settlement_point="LZ_WEST",
            start_date=day,
            end_date=day,
            market=market,
        )
        frame = cached_fetch(provider, request)
        span = frame["interval_end"].iloc[-1] - frame["interval_start"].iloc[0]
        ok = (
            len(frame) == expected
            and frame["interval_start"].is_unique
            and frame["interval_end"].iloc[-1] == frame["interval_start"].iloc[-1] + (
                frame["interval_end"].iloc[0] - frame["interval_start"].iloc[0]
            )
        )
        span_hours = span / pd.Timedelta(hours=1)
        failures += not ok
        print(
            f"{label + ' ' + market:<20} {len(frame):>5} {expected:>4} "
            f"{span_hours:>6.0f}h  {frame['interval_start'].iloc[0]} -> "
            f"{frame['interval_end'].iloc[-1]}  {'OK' if ok else 'MISMATCH'}"
        )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arb", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    backtest = sub.add_parser("backtest", help="backtest a strategy on real prices")
    _add_common(backtest)
    backtest.set_defaults(func=cmd_backtest)

    fetch = sub.add_parser("fetch", help="download and cache prices only")
    _add_common(fetch)
    fetch.set_defaults(func=cmd_fetch)

    verify = sub.add_parser("verify-data", help="check live data against ERCOT rules")
    _add_common(verify)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry points.

``wattson <command>`` exposes every command. ``wattson-backtest``, ``wattson-fetch`` and
``wattson-verify-data`` are shortcuts for ``wattson backtest`` and friends. The scripts
under ``scripts/`` are thin wrappers so the same code runs from a source
checkout.

Cached prices are used whenever they exist, so nothing needs credentials
unless it has to download a window that is not on disk yet.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from wattson.config import MAX_COUNT, PRESETS, BatteryConfig, scaled
from wattson.data.cache import cached_fetch, load
from wattson.data.ercot_source import ErcotLiveProvider, load_keys_file
from wattson.data.providers import PriceRequest
from wattson.metrics import compare_strategies, compare_zones, rank_agreement, rank_stability
from wattson.strategies.threshold import ThresholdStrategy
from wattson.zones import HUBS, LOAD_ZONES, label_for

DEFAULT_KEYS = "ERCOT API Keys.txt"


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--zone", default="LZ_AEN", help="settlement point")
    parser.add_argument("--start", default="2026-08-26", help="first delivery date")
    parser.add_argument("--end", default="2026-09-24", help="last delivery date")
    parser.add_argument("--market", default="RTM", choices=["RTM", "DAM"])
    parser.add_argument("--keys", default=DEFAULT_KEYS, help="credentials file")


DEFAULT_BATTERY = "residential_13kwh"


def _add_battery(parser: argparse.ArgumentParser, allow_all: bool = False) -> None:
    parser.add_argument(
        "--battery",
        default="all" if allow_all else DEFAULT_BATTERY,
        choices=list(PRESETS) + (["all"] if allow_all else []),
        help="battery preset" + (" (default: every preset)" if allow_all else ""),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        choices=range(1, MAX_COUNT + 1),
        metavar=f"1-{MAX_COUNT}",
        help="how many identical batteries; default 1",
    )


def _batteries(args: argparse.Namespace) -> list[BatteryConfig]:
    chosen = PRESETS.values() if args.battery == "all" else [PRESETS[args.battery]]
    return [scaled(b, args.count) for b in chosen]


def _battery(args: argparse.Namespace) -> BatteryConfig:
    return scaled(PRESETS[args.battery], args.count)


def _keys_ready(path: str) -> bool:
    """Load credentials if present. Missing credentials only matter on a cache miss."""
    try:
        load_keys_file(path)
        return True
    except (RuntimeError, OSError):
        return False


def _fetch(request: PriceRequest, keys_ready: bool) -> pd.DataFrame:
    hit = load(request)
    if hit is not None:
        return hit
    if not keys_ready:
        raise RuntimeError(
            f"no cached {request.market} prices for {request.settlement_point} "
            f"{request.start_date}..{request.end_date}, and no credentials to fetch them"
        )
    return cached_fetch(ErcotLiveProvider(), request)


def _request(args: argparse.Namespace, zone: str | None = None) -> PriceRequest:
    return PriceRequest(
        settlement_point=zone or args.zone,
        start_date=args.start,
        end_date=args.end,
        market=args.market,
    )


def _load(args: argparse.Namespace) -> pd.DataFrame:
    return _fetch(_request(args), _keys_ready(args.keys))


def _describe(prices: pd.DataFrame, args: argparse.Namespace) -> None:
    print(
        f"{label_for(args.zone)} {args.market} {args.start}..{args.end}: "
        f"{len(prices)} intervals, "
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

    for battery in _batteries(args):
        frame = compare_strategies(prices, battery, ThresholdStrategy())
        view = frame[
            [
                "net_usd",
                "net_after_degradation_usd",
                "usd_per_kw_year",
                "equivalent_full_cycles",
                "capture_ratio",
            ]
        ]
        with pd.option_context("display.width", 200):
            print(
                f"\n{battery.name}  "
                f"({battery.capacity_kwh:,.1f} kWh / {battery.power_kw:,.0f} kW)"
            )
            print(view.round(2).to_string())

    print(
        "\nWithout a battery there is no spread to capture, so the baseline is $0.\n"
        "The forecast strategy plans each day from earlier days' prices only.\n"
        "Perfect foresight sees every future price and is unreachable in practice."
    )
    return 0


def _zone_frames(
    args: argparse.Namespace, targets: list[str], keys_ready: bool
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for target in targets:
        try:
            frames[target] = _fetch(_request(args, target), keys_ready)
        except Exception as exc:  # noqa: BLE001 - one bad zone must not stop the rest
            skipped.append(f"{target} ({type(exc).__name__})")
    return frames, skipped


def cmd_compare_zones(args: argparse.Namespace) -> int:
    targets = list(LOAD_ZONES) + (list(HUBS) if args.hubs else [])
    frames, skipped = _zone_frames(args, targets, _keys_ready(args.keys))
    if not frames:
        print(f"no prices available for {args.market} {args.start}..{args.end}")
        return 1

    battery = _battery(args)
    table = compare_zones(frames, battery, ThresholdStrategy())
    view = table[
        [
            "net_usd",
            "net_after_degradation_usd",
            "usd_per_kw_year",
            "equivalent_full_cycles",
            "mean_price_usd_per_mwh",
            "p95_price_usd_per_mwh",
            "intervals",
        ]
    ]
    print(
        f"{battery.name}  {args.market}  {args.start}..{args.end}  "
        f"({len(table)} locations)\n"
    )
    with pd.option_context("display.width", 200):
        print(view.round(2).to_string())
    if skipped:
        print("\nskipped: " + ", ".join(skipped))
    print(
        "\nAnnualized from this window only; not a forecast. Locations are "
        "ranked by revenue per kW-year."
    )
    return 0


def _month_bounds(month: str) -> tuple[str, str]:
    period = pd.Period(month, freq="M")
    return period.start_time.date().isoformat(), period.end_time.date().isoformat()


def cmd_stability(args: argparse.Namespace) -> int:
    keys_ready = _keys_ready(args.keys)
    battery = _battery(args)
    targets = list(LOAD_ZONES) + (list(HUBS) if args.hubs else [])

    tables: dict[str, pd.DataFrame] = {}
    for month in args.months:
        start, end = _month_bounds(month)
        month_args = argparse.Namespace(**{**vars(args), "start": start, "end": end})
        frames, skipped = _zone_frames(month_args, targets, keys_ready)
        if skipped:
            print(f"{month}: skipped " + ", ".join(skipped))
        if frames:
            tables[month] = compare_zones(frames, battery, ThresholdStrategy())

    if len(tables) < 2:
        print("need at least two months with data to judge stability")
        return 1

    stability = rank_stability(tables)
    print(f"{battery.name}  {args.market}  revenue per kW-year by month\n")
    with pd.option_context("display.width", 200):
        print(stability.round(2).to_string())
    print(
        f"\nRank agreement between months (mean Spearman): "
        f"{rank_agreement(stability):.2f}"
        "\n1.00 means the order never changes; near 0 means one month's ranking "
        "says little about the next."
    )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    prices = _load(args)
    _describe(prices, args)
    print("  cached; re-running will read from disk without hitting the API")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    keys_ready = _keys_ready(args.keys)

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
        frame = _fetch(request, keys_ready)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wattson", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    backtest = sub.add_parser("backtest", help="backtest the strategies on real prices")
    _add_common(backtest)
    _add_battery(backtest, allow_all=True)
    backtest.set_defaults(func=cmd_backtest)

    fetch = sub.add_parser("fetch", help="download and cache prices only")
    _add_common(fetch)
    fetch.set_defaults(func=cmd_fetch)

    verify = sub.add_parser("verify-data", help="check live data against ERCOT rules")
    _add_common(verify)
    verify.set_defaults(func=cmd_verify)

    zones = sub.add_parser(
        "compare-zones", help="compare one battery across every load zone"
    )
    _add_common(zones)
    _add_battery(zones)
    zones.add_argument("--hubs", action="store_true", help="include hubs too")
    zones.set_defaults(func=cmd_compare_zones)

    stability = sub.add_parser(
        "stability", help="check whether zone rankings hold from month to month"
    )
    _add_common(stability)
    _add_battery(stability)
    stability.add_argument(
        "--months",
        nargs="+",
        default=["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"],
        help="months as YYYY-MM",
    )
    stability.add_argument("--hubs", action="store_true", help="include hubs too")
    stability.set_defaults(func=cmd_stability)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _shortcut(command: str) -> int:
    return main([command, *sys.argv[1:]])


def backtest_main() -> int:
    """``wattson-backtest``: same as ``wattson backtest``."""
    return _shortcut("backtest")


def fetch_main() -> int:
    """``wattson-fetch``: same as ``wattson fetch``."""
    return _shortcut("fetch")


def verify_main() -> int:
    """``wattson-verify-data``: same as ``wattson verify-data``."""
    return _shortcut("verify-data")


if __name__ == "__main__":
    sys.exit(main())

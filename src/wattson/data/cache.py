"""On-disk CSV cache for price frames.

CSV keeps the cache dependency-free and human-inspectable, which matters when a
demo has to survive a dead network on stage. Timestamps round-trip through
ISO-8601 with an explicit offset and are re-localized to Central Time on read.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from wattson.config import CENTRAL_TIME
from wattson.data.providers import (
    PRICE_COLUMNS,
    PriceRequest,
    PriceSeriesProvider,
    validate_price_frame,
)

DEFAULT_CACHE_DIR = Path(
    os.environ.get("WATTSON_CACHE_DIR", Path(__file__).resolve().parents[3] / "data" / "cache")
)


def cache_path(request: PriceRequest, cache_dir: Path | None = None) -> Path:
    base = cache_dir or DEFAULT_CACHE_DIR
    return base / (
        f"{request.market}_{request.settlement_point}_"
        f"{request.start_date}_{request.end_date}.csv"
    )


def load(request: PriceRequest, cache_dir: Path | None = None) -> pd.DataFrame | None:
    path = cache_path(request, cache_dir)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ("interval_start", "interval_end"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert(CENTRAL_TIME)
    return validate_price_frame(df)


def save(
    request: PriceRequest,
    df: pd.DataFrame,
    cache_dir: Path | None = None,
) -> Path:
    path = cache_path(request, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, list(PRICE_COLUMNS)].to_csv(path, index=False)
    return path


def cached_fetch(
    provider: PriceSeriesProvider,
    request: PriceRequest,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return cached data if present, otherwise fetch and cache."""
    if not refresh:
        hit = load(request, cache_dir)
        if hit is not None:
            return hit
    frame = validate_price_frame(provider.fetch(request))
    save(request, frame, cache_dir)
    return frame


_CACHE_NAME = re.compile(
    r"^(?P<market>RTM|DAM)_(?P<point>.+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.csv$"
)


def cached_windows(
    settlement_point: str,
    market: str,
    cache_dir: Path | None = None,
) -> list[tuple[dt.date, dt.date]]:
    """Date windows already on disk for one settlement point and market.

    Sorted by end date, then start date. Lets the UI open on data it can show
    without a network call.
    """
    base = cache_dir or DEFAULT_CACHE_DIR
    if not base.exists():
        return []
    found = []
    for path in base.glob(f"{market}_{settlement_point}_*.csv"):
        m = _CACHE_NAME.match(path.name)
        if m and m["point"] == settlement_point and m["market"] == market:
            found.append(
                (dt.date.fromisoformat(m["start"]), dt.date.fromisoformat(m["end"]))
            )
    return sorted(found, key=lambda w: (w[1], w[0]))


# ---------------------------------------------------------------------------
# Loading an arbitrary date range, including today.
#
# Finished days are stored in calendar-month files and never change. Today is
# still being published, so it is always fetched fresh and never written to
# disk. Any range is assembled from those pieces, which means a range that ends
# today still loads offline, just without today.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RangeResult:
    """Prices for a date range, and what could not be loaded."""

    frame: pd.DataFrame
    missing_days: tuple[dt.date, ...] = ()
    includes_today: bool = False
    today_error: str | None = None


@lru_cache(maxsize=4096)
def expected_intervals(day: dt.date, market: str) -> int:
    """Intervals in one local day: 96/92/100 for RTM, 24/23/25 for DAM."""
    start = pd.Timestamp(day).tz_localize(CENTRAL_TIME)
    end = pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(CENTRAL_TIME)
    per_hour = 4 if market == "RTM" else 1
    return int(round((end - start) / pd.Timedelta(hours=1) * per_hour))


def _local_dates(frame: pd.DataFrame) -> pd.Series:
    return frame["interval_start"].dt.tz_convert(CENTRAL_TIME).dt.date


# ERCOT publishes real-time prices within minutes; allow generous slack before
# treating a day saved after midnight as final.
SETTLE_AFTER = pd.Timedelta(hours=2)


def _day_end(day: dt.date) -> pd.Timestamp:
    return pd.Timestamp(day + dt.timedelta(days=1)).tz_localize(CENTRAL_TIME)


def _complete_days(frame: pd.DataFrame, market: str, settled: set[dt.date] = frozenset()) -> set[dt.date]:
    """Days that need no further download.

    A day is done if it has every interval, or if its data was saved after the
    day was over (``settled``). ERCOT's own history has occasional missing
    intervals that will never be filled in, so a full count alone would
    re-download those days forever. What must be caught is a day saved while
    it was still in progress, which is exactly what the save time reveals.
    """
    if frame.empty:
        return set()
    counts = _local_dates(frame).value_counts()
    return {
        day
        for day, n in counts.items()
        if n >= expected_intervals(day, market) or day in settled
    }


def _settled_days(frame: pd.DataFrame, saved_at: pd.Timestamp) -> set[dt.date]:
    if frame.empty:
        return set()
    return {d for d in set(_local_dates(frame)) if saved_at >= _day_end(d) + SETTLE_AFTER}


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))
    # Earlier frames win on duplicates, so pass fresh data first.
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset="interval_start", keep="first")
    return out.sort_values("interval_start").reset_index(drop=True)


def _month_chunks(days: list[dt.date], last_finished: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Calendar months containing ``days``, clipped so nothing unfinished is stored."""
    chunks = []
    for year, month in sorted({(d.year, d.month) for d in days}):
        first = dt.date(year, month, 1)
        nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
        chunks.append((first, min(nxt - dt.timedelta(days=1), last_finished)))
    return chunks


def load_range(
    settlement_point: str,
    market: str,
    start: dt.date,
    end: dt.date,
    *,
    today: dt.date,
    fetch=None,
    cache_dir: Path | None = None,
    now: pd.Timestamp | None = None,
) -> RangeResult:
    """Prices for ``start``..``end`` (inclusive), clipped to ``today``.

    ``fetch`` takes a :class:`PriceRequest` and returns a price frame, or is
    ``None`` when offline. Finished days come from disk when possible and are
    downloaded a month at a time when not. A cached day with fewer intervals
    than it should have (a partial download) is treated as missing.
    """
    base = cache_dir or DEFAULT_CACHE_DIR
    now = now if now is not None else pd.Timestamp.now(tz=CENTRAL_TIME)
    end = min(end, today)
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    last_finished = min(end, today - dt.timedelta(days=1))

    def in_range(frame: pd.DataFrame, lo: dt.date, hi: dt.date) -> pd.DataFrame:
        if frame.empty:
            return frame
        days = _local_dates(frame)
        return frame[(days >= lo) & (days <= hi)]

    # 1. Finished days already on disk, from any cached file that overlaps.
    cached = []
    settled: set[dt.date] = set()
    if start <= last_finished:
        for lo, hi in cached_windows(settlement_point, market, base):
            if hi >= start and lo <= last_finished:
                request = PriceRequest(
                    settlement_point=settlement_point,
                    start_date=lo.isoformat(),
                    end_date=hi.isoformat(),
                    market=market,
                )
                hit = load(request, base)
                if hit is not None:
                    part = in_range(hit, start, last_finished)
                    saved_at = pd.Timestamp(cache_path(request, base).stat().st_mtime, unit="s", tz="UTC")
                    settled |= _settled_days(part, saved_at)
                    cached.append(part)
    have = _combine(cached)

    wanted = [
        start + dt.timedelta(days=i) for i in range((last_finished - start).days + 1)
    ] if start <= last_finished else []
    done = _complete_days(have, market, settled)
    missing = [d for d in wanted if d not in done]

    # 2. Download missing finished days a month at a time, and keep them.
    fresh = []
    if missing and fetch is not None:
        # Whole months (up to yesterday), so the file serves any later range too.
        for lo, hi in _month_chunks(missing, today - dt.timedelta(days=1)):
            request = PriceRequest(
                settlement_point=settlement_point,
                start_date=lo.isoformat(),
                end_date=hi.isoformat(),
                market=market,
            )
            try:
                frame = validate_price_frame(fetch(request))
            except Exception:  # noqa: BLE001 - report the gap instead of failing
                continue
            save(request, frame, base)
            part = in_range(frame, start, last_finished)
            settled |= _settled_days(part, now)
            fresh.append(part)

    # 3. Today, always live and never stored.
    includes_today, today_error = False, None
    if end >= today:
        if fetch is None:
            today_error = "offline"
        else:
            try:
                live = validate_price_frame(
                    fetch(
                        PriceRequest(
                            settlement_point=settlement_point,
                            start_date=today.isoformat(),
                            end_date=today.isoformat(),
                            market=market,
                        )
                    )
                )
                fresh.append(in_range(live, today, today))
                includes_today = not live.empty
            except Exception as exc:  # noqa: BLE001
                today_error = f"{type(exc).__name__}: {exc}"

    frame = _combine(fresh + [have])
    done = _complete_days(frame, market, settled)
    still_missing = tuple(d for d in wanted if d not in done)
    return RangeResult(
        frame=frame,
        missing_days=still_missing,
        includes_today=includes_today,
        today_error=today_error,
    )

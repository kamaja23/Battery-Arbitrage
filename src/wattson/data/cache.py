"""On-disk CSV cache for price frames.

CSV keeps the cache dependency-free and human-inspectable, which matters when a
demo has to survive a dead network on stage. Timestamps round-trip through
ISO-8601 with an explicit offset and are re-localized to Central Time on read.
"""

from __future__ import annotations

import datetime as dt
import os
import re
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

"""Provider contract for ERCOT settlement-point price series.

Every provider returns the same normalized long-format frame:

``interval_start``      tz-aware, America/Chicago
``interval_end``        tz-aware, America/Chicago
``market``              "DAM" or "RTM"
``settlement_point``    e.g. "LZ_HOUSTON", "HB_NORTH"
``location_type``       "Load Zone" or "Hub"
``price_usd_per_mwh``   float, native ERCOT units
"""

from __future__ import annotations

import abc

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from wattson.config import CENTRAL_TIME, LocationType, Market

PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "interval_start",
    "interval_end",
    "market",
    "settlement_point",
    "location_type",
    "price_usd_per_mwh",
)

VALID_MARKETS: Final[frozenset[str]] = frozenset({"DAM", "RTM"})
VALID_LOCATION_TYPES: Final[frozenset[str]] = frozenset({"Load Zone", "Hub"})


def validate_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Assert the normalized contract; returns the frame unchanged."""
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price frame missing columns: {missing}")
    if df.empty:
        raise ValueError("price frame is empty")

    bad_market = set(df["market"].unique()) - VALID_MARKETS
    if bad_market:
        raise ValueError(f"invalid market values: {sorted(bad_market)}")

    bad_location = set(df["location_type"].unique()) - VALID_LOCATION_TYPES
    if bad_location:
        raise ValueError(f"invalid location_type values: {sorted(bad_location)}")

    for col in ("interval_start", "interval_end"):
        dtype = df[col].dtype
        if not isinstance(dtype, pd.DatetimeTZDtype):
            raise ValueError(f"{col} must be tz-aware, got {dtype}")
        if str(dtype.tz) != CENTRAL_TIME:
            raise ValueError(f"{col} must be in {CENTRAL_TIME}, got {dtype.tz}")

    if not pd.api.types.is_numeric_dtype(df["price_usd_per_mwh"]):
        raise ValueError("price_usd_per_mwh must be numeric")

    dupes = df.duplicated(
        subset=["interval_start", "market", "settlement_point"]
    ).sum()
    if dupes:
        raise ValueError(f"price frame has {dupes} duplicate keys")

    return df


@dataclass(frozen=True, slots=True)
class PriceRequest:
    """A request for one settlement point over a delivery-date range.

    Both ``start_date`` and ``end_date`` are inclusive, matching the ``ercot``
    package's own convention.
    """

    settlement_point: str
    start_date: str
    end_date: str
    market: Market = "RTM"
    location_type: LocationType = "Load Zone"


class PriceSeriesProvider(abc.ABC):
    """Anything that can produce a normalized price frame."""

    @abc.abstractmethod
    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        """Return a validated price frame for ``request``."""

    def fetch_many(self, requests: list[PriceRequest]) -> pd.DataFrame:
        frames = [self.fetch(r) for r in requests]
        return pd.concat(frames, ignore_index=True)


class SyntheticProvider:
    """Deterministic, offline ERCOT-shaped price generator.

    Emits a fixed 96 intervals/day (no DST gaps) so tests stay stable. Real
    ERCOT series drop to 92 and extend to 100 across DST transitions; that
    behaviour is exercised against the live provider instead.
    """

    def __init__(self, seed: int = 7) -> None:
        self._rng = np.random.default_rng(seed)

    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        interval_h = 1.0 if request.market == "DAM" else 0.25
        periods_per_day = int(round(24 / interval_h))

        start = pd.Timestamp(request.start_date, tz=CENTRAL_TIME)
        end = pd.Timestamp(request.end_date, tz=CENTRAL_TIME) + pd.Timedelta(days=1)
        idx = pd.date_range(start=start, end=end, freq=pd.Timedelta(hours=interval_h), inclusive="left")

        t = np.arange(len(idx), dtype=float)
        hour_of_day = (t % periods_per_day) / periods_per_day
        day_of_week = (t // periods_per_day) % 7

        daily = np.exp(-(((hour_of_day - 0.78) / 0.16) ** 2)) * 62.0
        morning = np.exp(-(((hour_of_day - 0.35) / 0.12) ** 2)) * 14.0
        weekly = 1.0 + 0.10 * np.cos(2 * np.pi * day_of_week / 7.0)
        seasonal = 1.0 + 0.35 * np.cos(2 * np.pi * (idx.dayofyear / 365.0))

        noise = self._rng.normal(0.0, 6.0, len(idx))
        spikes = self._rng.random(len(idx)) < 0.002
        spike_mag = self._rng.exponential(600.0, len(idx)) * spikes

        price = np.clip((18.0 + daily + morning) * weekly * seasonal + noise + spike_mag, -40.0, None)

        return pd.DataFrame(
            {
                "interval_start": idx,
                "interval_end": idx + pd.Timedelta(hours=interval_h),
                "market": request.market,
                "settlement_point": request.settlement_point,
                "location_type": request.location_type,
                "price_usd_per_mwh": price,
            }
        )

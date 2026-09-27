"""Rooftop solar panels alongside the battery.

Panel output follows a clear-sky model: the sun's position is computed for
each interval from the area's latitude and longitude, and panels tilted at the
latitude and facing south turn that sunlight into power. Clear skies overstate
a real year, so output is scaled so that a full year averages
``ANNUAL_KWH_PER_KW`` for every kW of panels, a typical Texas figure. The
result has the right daily and seasonal shape but no clouds: every day is an
average-weather version of a sunny day.

Solar power is valued at the same wholesale price the battery trades at, and
panels are switched off when that price is below zero (selling would cost
money). Power used at home instead of bought is usually worth more, at the
retail rate, but home use is not modelled.

The battery and the panels do not interact in this model. The battery could
store solar power, but since both are valued at the same wholesale price,
storing your own solar is worth exactly the same as charging from the grid at
that moment. The two simply add up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from wattson.config import usd_from_mwh

ANNUAL_KWH_PER_KW = 1450.0
DEFAULT_PANEL_KW = 8.0


def _clear_sky(utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Relative output (0..~1) of latitude-tilted, south-facing panels."""
    day = utc.dayofyear.to_numpy(dtype=float)
    hours = (utc.hour + utc.minute / 60 + utc.second / 3600).to_numpy(dtype=float)
    b = 2 * math.pi * (day - 81) / 364
    equation_of_time = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)  # minutes
    declination = np.radians(23.45) * np.sin(2 * math.pi * (284 + day) / 365)
    solar_time = hours + lon / 15 + equation_of_time / 60
    hour_angle = np.radians(15 * (solar_time - 12))
    phi = math.radians(lat)
    cos_zenith = math.sin(phi) * np.sin(declination) + math.cos(phi) * np.cos(declination) * np.cos(hour_angle)
    up = cos_zenith > 0.01
    air_mass = np.where(up, 1 / np.clip(cos_zenith, 0.01, None), 0.0)
    beam = np.where(up, 1.353 * 0.7 ** (air_mass ** 0.678), 0.0)  # kW per m2
    # A panel tilted at the latitude and facing south sees the sun at this angle.
    cos_incidence = np.clip(np.cos(declination) * np.cos(hour_angle), 0, None)
    return np.where(up, beam * cos_incidence + 0.1 * beam, 0.0)


@lru_cache(maxsize=64)
def _annual_clear_sky_kwh(lat: float, lon: float) -> float:
    """Clear-sky yield of 1 kW of panels over a whole year, for scaling."""
    stamps = pd.date_range("2026-01-01", "2027-01-01", freq="15min", tz="UTC", inclusive="left")
    return float(_clear_sky(stamps + pd.Timedelta(minutes=7.5), lat, lon).sum() * 0.25)


def _interval_hours(index: pd.DatetimeIndex) -> np.ndarray:
    deltas = pd.Series(index).diff()
    modal = deltas.mode()
    step = modal.iloc[0] if not modal.empty else pd.Timedelta(minutes=15)
    return deltas.fillna(step).dt.total_seconds().to_numpy() / 3600


def panel_output_kw(
    interval_starts: pd.Series | pd.DatetimeIndex,
    panel_kw: float,
    lat: float,
    lon: float,
) -> np.ndarray:
    """Average panel output (kW) over each interval."""
    starts = pd.DatetimeIndex(interval_starts)
    hours = _interval_hours(starts)
    mid = (starts + pd.to_timedelta(hours / 2, unit="h")).tz_convert("UTC")
    scale = ANNUAL_KWH_PER_KW / _annual_clear_sky_kwh(round(lat, 3), round(lon, 3))
    return panel_kw * scale * _clear_sky(mid, lat, lon)


@dataclass(frozen=True, slots=True)
class SolarResult:
    panel_kw: float
    days: float
    generated_kwh: float
    sold_kwh: float
    switched_off_hours: float
    revenue_usd: float
    avg_price_sold_usd_per_mwh: float
    avg_price_usd_per_mwh: float
    output_kw: np.ndarray

    @property
    def per_year_usd(self) -> float:
        return self.revenue_usd / self.days * 365 if self.days else 0.0


def solar_value(prices: pd.DataFrame, panel_kw: float, lat: float, lon: float) -> SolarResult:
    """What ``panel_kw`` of panels would have earned at these prices."""
    frame = prices.sort_values("interval_start")
    starts = pd.DatetimeIndex(frame["interval_start"])
    price = frame["price_usd_per_mwh"].to_numpy(dtype=float)
    hours = _interval_hours(starts)
    output = panel_output_kw(starts, panel_kw, lat, lon)
    generated = output * hours
    selling = price >= 0
    sold = np.where(selling, generated, 0.0)
    revenue = float(np.sum(usd_from_mwh(sold, price)))
    sold_total = float(sold.sum())
    return SolarResult(
        panel_kw=panel_kw,
        days=float(hours.sum() / 24),
        generated_kwh=float(generated.sum()),
        sold_kwh=sold_total,
        switched_off_hours=float(hours[(~selling) & (generated > 0)].sum()),
        revenue_usd=revenue,
        avg_price_sold_usd_per_mwh=(revenue / sold_total * 1000) if sold_total else 0.0,
        avg_price_usd_per_mwh=float(np.average(price, weights=hours)),
        output_kw=output,
    )

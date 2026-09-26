"""Day-ahead price forecasting from past prices only.

The forecast for a day is the median price in each time-of-day slot over the
previous ``history_days`` days of data. It is deliberately simple: it needs
nothing but the price history the tool already has, it cannot leak the future,
and its error is measured against the obvious baseline ("tomorrow looks like
today") so nobody has to take its usefulness on faith.

Only days before the one being forecast are used. Days are counted in the
data, so a gap in the history is skipped rather than filled.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wattson.config import CENTRAL_TIME, BatteryConfig

DEFAULT_HISTORY_DAYS = 7


def _local(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_convert(CENTRAL_TIME) if idx.tz is not None else idx


def _slots(local: pd.DatetimeIndex) -> np.ndarray:
    """Minutes since local midnight, the key that lines days up."""
    return np.asarray(local.hour * 60 + local.minute)


def _by_day(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = _local(series.index)
    frame = pd.DataFrame(
        {
            "date": local.date,
            "slot": _slots(local),
            "price": series.to_numpy(dtype=float),
        }
    )
    # The fall-back hour repeats slots; average the repeats within a day.
    table = frame.pivot_table(
        index="date", columns="slot", values="price", aggfunc="mean"
    ).sort_index()
    return frame, table


def _profile(history: pd.DataFrame) -> tuple[pd.Series, float]:
    return history.median(axis=0, skipna=True), float(np.nanmedian(history.to_numpy()))


def profile_forecast(
    series: pd.Series,
    history_days: int = DEFAULT_HISTORY_DAYS,
    min_history_days: int = 1,
) -> pd.Series:
    """Forecast every interval using only the days before it.

    Returns a series aligned with ``series``. Intervals on days with fewer
    than ``min_history_days`` of prior data are NaN: there is nothing honest
    to forecast from.
    """
    if history_days < 1 or min_history_days < 1:
        raise ValueError("history_days and min_history_days must be >= 1")

    frame, table = _by_day(series)
    positions = frame.groupby("date").indices
    out = np.full(len(frame), np.nan)

    for k, day in enumerate(table.index):
        history = table.iloc[max(0, k - history_days) : k]
        if len(history) < min_history_days:
            continue
        profile, fallback = _profile(history)
        pos = positions[day]
        values = profile.reindex(frame["slot"].to_numpy()[pos]).to_numpy()
        out[pos] = np.where(np.isnan(values), fallback, values)

    return pd.Series(out, index=series.index, name="forecast_usd_per_mwh")


@dataclass(frozen=True, slots=True)
class ForecastAccuracy:
    """How far the forecast was from what actually cleared."""

    intervals: int
    days: int
    mae_usd_per_mwh: float
    rmse_usd_per_mwh: float
    bias_usd_per_mwh: float
    naive_mae_usd_per_mwh: float

    @property
    def skill(self) -> float:
        """Share of the naive forecast's error removed (1 is perfect, <0 is worse)."""
        if self.naive_mae_usd_per_mwh <= 0:
            return float("nan")
        return 1.0 - self.mae_usd_per_mwh / self.naive_mae_usd_per_mwh


def forecast_accuracy(
    series: pd.Series, history_days: int = DEFAULT_HISTORY_DAYS
) -> ForecastAccuracy:
    """Score the forecast against realized prices and against "same as yesterday".

    Both forecasts are scored on exactly the same intervals.
    """
    forecast = profile_forecast(series, history_days)
    naive = profile_forecast(series, history_days=1)
    mask = forecast.notna() & naive.notna()
    actual = series[mask].astype(float)
    err = forecast[mask] - actual
    naive_err = naive[mask] - actual
    if err.empty:
        nan = float("nan")
        return ForecastAccuracy(0, 0, nan, nan, nan, nan)
    return ForecastAccuracy(
        intervals=int(mask.sum()),
        days=int(len(set(_local(actual.index).date))),
        mae_usd_per_mwh=float(err.abs().mean()),
        rmse_usd_per_mwh=float(np.sqrt((err**2).mean())),
        bias_usd_per_mwh=float(err.mean()),
        naive_mae_usd_per_mwh=float(naive_err.abs().mean()),
    )


@dataclass(frozen=True, slots=True)
class DayPlan:
    """A forecast price curve for one day and the schedule optimized against it."""

    date: dt.date
    timestamps: pd.DatetimeIndex
    forecast_usd_per_mwh: np.ndarray
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    soc_kwh: np.ndarray
    expected_net_usd: float
    expected_wear_usd: float

    @property
    def expected_net_after_wear_usd(self) -> float:
        return self.expected_net_usd - self.expected_wear_usd

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "interval_start": self.timestamps,
                "forecast_usd_per_mwh": self.forecast_usd_per_mwh,
                "charge_kw": self.charge_kw,
                "discharge_kw": self.discharge_kw,
                "soc_kwh": self.soc_kwh,
            }
        )


def plan_next_day(
    series: pd.Series,
    battery: BatteryConfig,
    history_days: int = DEFAULT_HISTORY_DAYS,
) -> DayPlan:
    """Forecast the day after ``series`` ends and plan the battery against it.

    The plan starts and ends at mid-window state of charge, and the optimizer
    is charged for battery wear so it only trades spreads worth taking.
    """
    from wattson.strategies.perfect_foresight import solve_perfect_foresight

    if len(series) < 2:
        raise ValueError("need at least one day of prices to forecast from")
    frame, table = _by_day(series)
    profile, fallback = _profile(table.iloc[-history_days:])

    step_minutes = int(
        pd.Series(pd.DatetimeIndex(series.index)).diff().mode().iloc[0]
        / pd.Timedelta(minutes=1)
    )
    day = table.index[-1] + dt.timedelta(days=1)
    naive = pd.date_range(
        pd.Timestamp(day), periods=24 * 60 // step_minutes, freq=f"{step_minutes}min"
    )
    stamps = naive.tz_localize(
        CENTRAL_TIME, nonexistent="shift_forward", ambiguous=True
    )
    stamps = stamps[~stamps.duplicated()]

    values = profile.reindex(_slots(stamps)).to_numpy()
    values = np.where(np.isnan(values), fallback, values)

    solution = solve_perfect_foresight(
        pd.Series(values, index=stamps),
        battery,
        throughput_cost_per_kwh=battery.degradation_cost_per_kwh,
    )
    hours = step_minutes / 60.0
    wear = battery.degradation_cost_per_kwh * float(
        np.sum(solution.charge_kw + solution.discharge_kw) * hours
    )
    return DayPlan(
        date=day,
        timestamps=stamps,
        forecast_usd_per_mwh=values,
        charge_kw=solution.charge_kw,
        discharge_kw=solution.discharge_kw,
        soc_kwh=solution.soc_kwh,
        expected_net_usd=solution.net_usd,
        expected_wear_usd=wear,
    )

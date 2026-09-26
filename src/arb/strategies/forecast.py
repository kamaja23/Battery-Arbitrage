"""Forecast-driven dispatch.

At the first interval of each day the strategy forecasts that day's prices
from earlier days only (see :mod:`arb.forecast`), optimizes a schedule against
the forecast starting from the battery's actual state of charge, and then
follows that schedule through the day. Settlement uses real prices, so every
forecast error shows up in the result.

The optimizer is charged for battery wear, so days whose forecast spread would
not cover it are left idle. Days with no prior data to forecast from are idle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from arb.config import CENTRAL_TIME
from arb.forecast import DEFAULT_HISTORY_DAYS, profile_forecast
from arb.strategies.base import Action, DecisionContext, Strategy


@dataclass(slots=True)
class ForecastStrategy(Strategy):
    """Plan each day against a price forecast, then execute the plan."""

    history_days: int = DEFAULT_HISTORY_DAYS

    name: str = field(default="forecast", init=False)

    def __post_init__(self) -> None:
        if self.history_days < 1:
            raise ValueError("history_days must be >= 1")

    def prepare(self, prices: pd.Series, timestamps: pd.DatetimeIndex) -> None:
        n = len(prices)
        self._stamps = timestamps
        self._forecast = profile_forecast(prices, self.history_days).to_numpy()
        dates = timestamps.tz_convert(CENTRAL_TIME).date
        starts = np.zeros(n, dtype=int)
        ends = np.full(n, n, dtype=int)
        day_start = 0
        for i in range(1, n + 1):
            if i == n or dates[i] != dates[i - 1]:
                starts[day_start:i] = day_start
                ends[day_start:i] = i
                day_start = i
        self._day_start, self._day_end = starts, ends
        self._charge = np.zeros(n)
        self._discharge = np.zeros(n)
        self._planned_from = -1

    def decide(self, ctx: DecisionContext) -> Action:
        i = ctx.index
        start = int(self._day_start[i])
        if start != self._planned_from:
            self._planned_from = start
            self._plan_day(start, int(self._day_end[i]), ctx)
        net = self._charge[i] - self._discharge[i]
        if net > 1e-9:
            return Action(charge_kw=float(net))
        if net < -1e-9:
            return Action(discharge_kw=float(-net))
        return Action()

    def _plan_day(self, start: int, end: int, ctx: DecisionContext) -> None:
        from arb.strategies.perfect_foresight import solve_perfect_foresight

        forecast = self._forecast[start:end]
        if end - start < 2 or np.isnan(forecast).any():
            return
        battery = ctx.battery
        soc = min(max(ctx.soc_fraction, battery.soc_min), battery.soc_max)
        plan = solve_perfect_foresight(
            pd.Series(forecast, index=self._stamps[start:end]),
            battery,
            initial_soc_fraction=soc,
            throughput_cost_per_kwh=battery.degradation_cost_per_kwh,
        )
        self._charge[start:end] = plan.charge_kw
        self._discharge[start:end] = plan.discharge_kw

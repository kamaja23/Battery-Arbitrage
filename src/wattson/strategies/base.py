"""Strategy contract.

A strategy maps the state of the world at each interval to a power request.
It never mutates the battery directly: the engine applies physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from wattson.config import BatteryConfig


@dataclass(frozen=True, slots=True)
class Action:
    """A power request for one interval. Positive values are magnitudes."""

    charge_kw: float = 0.0
    discharge_kw: float = 0.0

    @property
    def is_idle(self) -> bool:
        return self.charge_kw == 0.0 and self.discharge_kw == 0.0


@dataclass(slots=True)
class DecisionContext:
    """Everything a strategy may look at for interval ``index``."""

    index: int
    interval_hours: float
    price_usd_per_mwh: float
    soc_kwh: float
    soc_fraction: float
    battery: BatteryConfig
    prices: pd.Series = field(repr=False)
    timestamps: pd.DatetimeIndex = field(repr=False)


class Strategy:
    """Base class for dispatch strategies.

    ``prepare`` is called once before the simulation loop and is the place for
    vectorized work (rolling quantiles, forecasts). ``decide`` is then called
    per interval and should stay cheap.
    """

    name: str = "base"

    def prepare(self, prices: pd.Series, timestamps: pd.DatetimeIndex) -> None:
        """Precompute anything vectorizable ahead of the loop."""

    def decide(self, ctx: DecisionContext) -> Action:
        raise NotImplementedError

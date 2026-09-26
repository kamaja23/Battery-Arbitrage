"""Rule-based price-threshold strategy.

Charge when price sits in the bottom ``charge_quantile`` of a trailing window;
discharge when it sits in the top ``discharge_quantile``. Thresholds are
computed once, vectorized, and shifted by one interval so the rule is strictly
causal and never sees the price it is reacting to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from arb.strategies.base import Action, DecisionContext, Strategy


@dataclass(slots=True)
class ThresholdStrategy(Strategy):
    """Trailing-percentile dispatch rule."""

    lookback_intervals: int = 96
    charge_quantile: float = 0.25
    discharge_quantile: float = 0.75
    min_spread_usd_per_mwh: float = 5.0
    charge_at_negative_price: bool = True

    name: str = field(default="threshold", init=False)

    def __post_init__(self) -> None:
        if self.lookback_intervals < 2:
            raise ValueError("lookback_intervals must be >= 2")
        if not 0.0 <= self.charge_quantile < self.discharge_quantile <= 1.0:
            raise ValueError(
                "require 0 <= charge_quantile < discharge_quantile <= 1"
            )
        if self.min_spread_usd_per_mwh < 0:
            raise ValueError("min_spread_usd_per_mwh must be >= 0")

    def prepare(self, prices: pd.Series, timestamps: pd.DatetimeIndex) -> None:
        window = prices.rolling(
            self.lookback_intervals, min_periods=self.lookback_intervals
        )
        self._charge_threshold = window.quantile(self.charge_quantile).shift(1)
        self._discharge_threshold = window.quantile(self.discharge_quantile).shift(1)

    def decide(self, ctx: DecisionContext) -> Action:
        charge_at = self._charge_threshold.iat[ctx.index]
        discharge_at = self._discharge_threshold.iat[ctx.index]

        if not np.isfinite(charge_at) or not np.isfinite(discharge_at):
            return Action()

        price = ctx.price_usd_per_mwh
        spread = discharge_at - charge_at
        if spread <= self.min_spread_usd_per_mwh:
            return Action()

        if self.charge_at_negative_price and price < 0.0:
            return Action(charge_kw=ctx.battery.power_kw)

        if price <= charge_at:
            if ctx.soc_fraction < ctx.battery.soc_max:
                return Action(charge_kw=ctx.battery.power_kw)
            return Action()

        if price >= discharge_at:
            if ctx.soc_fraction > ctx.battery.soc_min:
                return Action(discharge_kw=ctx.battery.power_kw)

        return Action()

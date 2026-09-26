"""Perfect-foresight linear program.

Solves for the charge/discharge schedule that maximizes net revenue given
perfect knowledge of every future price. This is an unreachable upper bound:
no real controller can see the future. Its purpose is to quantify the capture
ratio of an online strategy, which is the strongest demo narrative available
("we captured X% of theoretical max").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from wattson.config import BatteryConfig, usd_from_mwh
from wattson.sim.battery import BatteryState

_SCALE = 1000.0


@dataclass(frozen=True, slots=True)
class PerfectForesightSolution:
    """Optimal schedule plus its realized revenue."""

    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    soc_kwh: np.ndarray
    net_usd: float

    def to_frame(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        hours = self._interval_hours(index)
        return pd.DataFrame(
            {
                "interval_start": index,
                "charge_kw": self.charge_kw,
                "discharge_kw": self.discharge_kw,
                "charged_kwh": self.charge_kw * hours,
                "discharged_kwh": self.discharge_kw * hours,
                "soc_kwh": self.soc_kwh,
            },
            index=index,
        )

    @staticmethod
    def _interval_hours(index: pd.DatetimeIndex) -> np.ndarray:
        deltas = pd.Series(index, index=index).diff()
        deltas.iloc[0] = deltas.mode().iloc[0] if not deltas.mode().empty else deltas.iloc[1]
        return deltas.dt.total_seconds().to_numpy() / 3600.0


def solve_perfect_foresight(
    prices: pd.Series,
    battery: BatteryConfig,
    initial_soc_fraction: float | None = None,
    throughput_cost_per_kwh: float = 0.0,
) -> PerfectForesightSolution:
    """Maximize net revenue over ``prices`` with full knowledge of the future.

    ``throughput_cost_per_kwh`` charges the objective for every kWh moved in
    or out, which stops the optimizer cycling on spreads that would not cover
    battery wear. It defaults to zero so the upper bound stays a pure
    energy-arbitrage ceiling; ``net_usd`` always reports energy revenue only.
    """
    values = prices.to_numpy(dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("need at least two intervals")

    index = pd.DatetimeIndex(prices.index)
    hours = PerfectForesightSolution._interval_hours(index)
    cap = battery.capacity_kwh
    soc_min = battery.soc_min * cap
    soc_max = battery.soc_max * cap
    start_frac = (
        (battery.soc_min + battery.soc_max) / 2.0
        if initial_soc_fraction is None
        else initial_soc_fraction
    )

    problem = pulp.LpProblem("perfect_foresight", pulp.LpMaximize)

    charge = [
        pulp.LpVariable(f"c_{i}", lowBound=0, upBound=battery.power_kw, cat="Continuous")
        for i in range(n)
    ]
    discharge = [
        pulp.LpVariable(f"d_{i}", lowBound=0, upBound=battery.power_kw, cat="Continuous")
        for i in range(n)
    ]
    soc = [
        pulp.LpVariable(f"s_{i}", lowBound=soc_min, upBound=soc_max, cat="Continuous")
        for i in range(n)
    ]

    eta_c, eta_d = battery.eta_charge, battery.eta_discharge

    profit = pulp.lpSum(
        usd_from_mwh(discharge[i] * hours[i], float(values[i]))
        - usd_from_mwh(charge[i] * hours[i], float(values[i]))
        for i in range(n)
    )
    if throughput_cost_per_kwh:
        profit -= pulp.lpSum(
            throughput_cost_per_kwh * (charge[i] + discharge[i]) * hours[i]
            for i in range(n)
        )
    problem += profit

    problem += soc[0] == start_frac * cap + eta_c * charge[0] * hours[0] - discharge[0] * hours[0] / eta_d
    for i in range(1, n):
        problem += (
            soc[i]
            == soc[i - 1]
            + eta_c * charge[i] * hours[i]
            - discharge[i] * hours[i] / eta_d
        )

    problem += soc[n - 1] >= start_frac * cap

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[problem.status] != "Optimal":
        raise RuntimeError(f"LP did not solve: {pulp.LpStatus[problem.status]}")

    charge_kw = np.array([float(v.value() or 0.0) for v in charge])
    discharge_kw = np.array([float(v.value() or 0.0) for v in discharge])
    soc_kwh = np.array([float(v.value() or 0.0) for v in soc])

    charged = charge_kw * hours
    discharged = discharge_kw * hours
    net = float(
        np.sum(usd_from_mwh(discharged, values) - usd_from_mwh(charged, values))
    )

    return PerfectForesightSolution(
        charge_kw=charge_kw,
        discharge_kw=discharge_kw,
        soc_kwh=soc_kwh,
        net_usd=net,
    )


def run_perfect_foresight_backtest(
    prices: pd.DataFrame,
    battery: BatteryConfig,
    market: str = "RTM",
    settlement_point: str | None = None,
) -> tuple[pd.DataFrame, float]:
    """Solve the LP and return a ledger-shaped frame plus net revenue."""
    from wattson.sim.engine import prepare_price_series

    if settlement_point is None:
        points = sorted(prices["settlement_point"].unique())
        if len(points) != 1:
            raise ValueError(
                f"settlement_point is required when the frame holds {points}"
            )
        settlement_point = points[0]

    series = prepare_price_series(prices, market, settlement_point)
    solution = solve_perfect_foresight(series, battery)
    frame = solution.to_frame(series.index).reset_index(drop=True)
    return frame, solution.net_usd

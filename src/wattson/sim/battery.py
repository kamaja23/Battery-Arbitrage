"""Battery physics: state of charge tracking with efficiency and bounds.

All energies here are AC-side kWh, i.e. what the meter and the inverter see.
Round-trip efficiency is applied as a symmetric split so that
``eta_charge * eta_discharge == round_trip_efficiency``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wattson.config import BatteryConfig, usd_from_mwh


@dataclass(slots=True)
class StepResult:
    """Outcome of advancing the battery by one interval."""

    charge_kw: float = 0.0
    discharge_kw: float = 0.0
    charged_kwh: float = 0.0
    discharged_kwh: float = 0.0
    curtailed_charge_kw: float = 0.0
    curtailed_discharge_kw: float = 0.0
    soc_start_kwh: float = 0.0
    soc_end_kwh: float = 0.0
    revenue_usd: float = 0.0
    cost_usd: float = 0.0

    @property
    def net_usd(self) -> float:
        return self.revenue_usd - self.cost_usd

    @property
    def throughput_kwh(self) -> float:
        return self.charged_kwh + self.discharged_kwh


@dataclass(slots=True)
class BatteryState:
    """Mutable state of one battery asset over a backtest."""

    config: BatteryConfig
    soc_kwh: float = 0.0
    _soc_min_kwh: float = field(init=False, default=0.0)
    _soc_max_kwh: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._soc_min_kwh = self.config.soc_min * self.config.capacity_kwh
        self._soc_max_kwh = self.config.soc_max * self.config.capacity_kwh
        if not self._soc_min_kwh <= self.soc_kwh <= self._soc_max_kwh:
            self.reset()

    @property
    def soc_fraction(self) -> float:
        return self.soc_kwh / self.config.capacity_kwh

    @property
    def headroom_charge_kwh(self) -> float:
        return max(0.0, self._soc_max_kwh - self.soc_kwh)

    @property
    def headroom_discharge_kwh(self) -> float:
        return max(0.0, self.soc_kwh - self._soc_min_kwh)

    def reset(self, soc_fraction: float | None = None) -> None:
        frac = (
            (self.config.soc_min + self.config.soc_max) / 2.0
            if soc_fraction is None
            else soc_fraction
        )
        self.soc_kwh = frac * self.config.capacity_kwh

    def step(
        self,
        charge_kw: float,
        discharge_kw: float,
        interval_hours: float,
        price_usd_per_mwh: float,
    ) -> StepResult:
        """Advance the battery by one interval.

        Requests are clamped to rated power and to the energy actually
        available in the current direction, so SoC bounds hold by
        construction. Simultaneous charge and discharge is resolved in favour
        of discharge, since doing both is always strictly wasteful.
        """
        cfg = self.config
        result = StepResult(soc_start_kwh=self.soc_kwh)

        want_c = max(0.0, charge_kw)
        want_d = max(0.0, discharge_kw)

        if want_d > 0.0:
            want_c = 0.0

        max_charge_kwh = min(
            cfg.power_kw * interval_hours,
            self.headroom_charge_kwh / cfg.eta_charge,
        )
        actual_charge_kwh = min(want_c * interval_hours, max(0.0, max_charge_kwh))

        max_discharge_kwh = min(
            cfg.power_kw * interval_hours,
            self.headroom_discharge_kwh * cfg.eta_discharge,
        )
        actual_discharge_kwh = min(want_d * interval_hours, max(0.0, max_discharge_kwh))

        result.charge_kw = actual_charge_kwh / interval_hours
        result.discharge_kw = actual_discharge_kwh / interval_hours
        result.charged_kwh = actual_charge_kwh
        result.discharged_kwh = actual_discharge_kwh
        result.curtailed_charge_kw = want_c - result.charge_kw
        result.curtailed_discharge_kw = want_d - result.discharge_kw

        self.soc_kwh += (
            actual_charge_kwh * cfg.eta_charge
            - actual_discharge_kwh / cfg.eta_discharge
        )
        self.soc_kwh = min(max(self.soc_kwh, self._soc_min_kwh), self._soc_max_kwh)
        result.soc_end_kwh = self.soc_kwh

        result.revenue_usd = usd_from_mwh(actual_discharge_kwh, price_usd_per_mwh)
        result.cost_usd = usd_from_mwh(actual_charge_kwh, price_usd_per_mwh)
        return result

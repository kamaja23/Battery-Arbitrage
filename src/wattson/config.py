"""Battery, engine, and pricing configuration.

Prices are stored natively in USD/MWh (the ERCOT convention). All money
conversion goes through :func:`usd_from_mwh` so the MWh->kWh factor exists in
exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from typing import Literal

Market = Literal["DAM", "RTM"]
LocationType = Literal["Load Zone", "Hub"]

CENTRAL_TIME = "America/Chicago"
USD_PER_MWH_TO_USD_PER_KWH = 1e-3

HOURS_PER_YEAR = 8760.0


def usd_from_mwh(energy_kwh: float, price_usd_per_mwh: float) -> float:
    """Convert an energy flow priced in USD/MWh into USD.

    ``energy_kwh`` is the AC-side energy (what the meter and the battery see).
    """
    return energy_kwh * price_usd_per_mwh * USD_PER_MWH_TO_USD_PER_KWH


@dataclass(frozen=True, slots=True)
class BatteryConfig:
    """Physical and financial parameters of one battery asset."""

    name: str
    capacity_kwh: float
    power_kw: float
    round_trip_efficiency: float = 0.90
    soc_min: float = 0.10
    soc_max: float = 0.95
    degradation_cost_per_kwh: float = 0.0
    installed_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if self.power_kw <= 0:
            raise ValueError("power_kw must be positive")
        if not 0.0 < self.round_trip_efficiency <= 1.0:
            raise ValueError("round_trip_efficiency must be in (0, 1]")
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError("require 0 <= soc_min < soc_max <= 1")

    @property
    def eta_charge(self) -> float:
        return self.round_trip_efficiency**0.5

    @property
    def eta_discharge(self) -> float:
        return self.round_trip_efficiency**0.5

    @property
    def usable_energy_kwh(self) -> float:
        return self.capacity_kwh * (self.soc_max - self.soc_min)

    @property
    def duration_hours(self) -> float:
        """Hours to traverse the usable window at rated power."""
        return self.usable_energy_kwh / self.power_kw


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Backtest-level knobs that are not properties of the battery itself."""

    market: Market = "RTM"
    location_type: LocationType = "Load Zone"
    allow_negative_price_charging: bool = True
    price_floor_usd_per_mwh: float | None = None

    def __post_init__(self) -> None:
        if self.market not in ("DAM", "RTM"):
            raise ValueError(f"market must be DAM or RTM, got {self.market!r}")


RESIDENTIAL_13KWH = BatteryConfig(
    name="residential_13kwh",
    capacity_kwh=13.5,
    power_kw=5.0,
    round_trip_efficiency=0.90,
    soc_min=0.10,
    soc_max=0.95,
    degradation_cost_per_kwh=0.015,
    installed_cost_usd=15_000.0,
)

COMMERCIAL_1MW = BatteryConfig(
    name="commercial_1mw",
    capacity_kwh=2_000.0,
    power_kw=1_000.0,
    round_trip_efficiency=0.92,
    soc_min=0.05,
    soc_max=0.95,
    degradation_cost_per_kwh=0.008,
    installed_cost_usd=960_000.0,
)

# A larger home battery: 39.2 kWh on an 11 kW inverter. Efficiency, usable
# charge window and wear use typical lithium iron phosphate assumptions. No
# purchase price is assumed; one can be entered in the app for a payback figure.
LARGE_HOME_39KWH = BatteryConfig(
    name="large_home_39kwh",
    capacity_kwh=39.2,
    power_kw=11.0,
    round_trip_efficiency=0.90,
    soc_min=0.10,
    soc_max=0.95,
    degradation_cost_per_kwh=0.012,
    installed_cost_usd=None,
)

PRESETS: dict[str, BatteryConfig] = {
    b.name: b for b in (RESIDENTIAL_13KWH, LARGE_HOME_39KWH, COMMERCIAL_1MW)
}

MAX_COUNT = 10


def scaled(battery: BatteryConfig, count: int) -> BatteryConfig:
    """``count`` identical batteries at one site, modelled as one larger battery.

    Capacity, power and any purchase price add up. Every unit sees the same
    prices, so in this energy-only model earnings scale directly with the count.
    """
    if not 1 <= count <= MAX_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_COUNT}, got {count}")
    if count == 1:
        return battery
    return replace(
        battery,
        name=f"{battery.name}_x{count}",
        capacity_kwh=round(battery.capacity_kwh * count, 6),
        power_kw=round(battery.power_kw * count, 6),
        installed_cost_usd=(
            None if battery.installed_cost_usd is None else battery.installed_cost_usd * count
        ),
    )

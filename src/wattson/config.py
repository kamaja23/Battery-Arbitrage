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

# Base Power's home battery: Base Core. Its 39.2 kWh capacity is on
# basepowercompany.com/specs/core, and Base's help center lists the 39.2 kWh
# system with an 11 kW inverter. Homes can have one Core or two. Base does not publish
# round-trip efficiency, usable SOC window, or cycle life, so those use the
# same LFP assumptions as the generic presets above. Base owns the battery and
# prices vary by address, so no installed cost is assumed.
_BASE_LFP = dict(
    round_trip_efficiency=0.90,
    soc_min=0.10,
    soc_max=0.95,
    degradation_cost_per_kwh=0.012,
    installed_cost_usd=None,
)

BASE_CORE = BatteryConfig(name="base_core", capacity_kwh=39.2, power_kw=11.0, **_BASE_LFP)
MAX_BASE_CORES = 10


def base_cores(count: int) -> BatteryConfig:
    """``count`` Base Cores at one home, modelled as one larger battery.

    Each Core adds 39.2 kWh and 11 kW. Base installs one or two per home; the
    combined power of two is not published, so adding the inverters up is an
    assumption. Every Core sees the same prices, so in this energy-only model
    earnings scale directly with the count.
    """
    if not 1 <= count <= MAX_BASE_CORES:
        raise ValueError(f"count must be between 1 and {MAX_BASE_CORES}, got {count}")
    if count == 1:
        return BASE_CORE
    return replace(
        BASE_CORE,
        name=f"base_core_x{count}",
        capacity_kwh=round(BASE_CORE.capacity_kwh * count, 6),
        power_kw=round(BASE_CORE.power_kw * count, 6),
    )


# The app offers only Base's product. The generic presets above stay defined
# because the CLI and the test suite use them as fixed reference batteries.
PRESETS: dict[str, BatteryConfig] = {BASE_CORE.name: BASE_CORE}

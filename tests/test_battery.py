from __future__ import annotations

import pytest

from arb.config import BatteryConfig
from arb.sim.battery import BatteryState, StepResult


@pytest.fixture
def battery() -> BatteryConfig:
    return BatteryConfig(
        name="test",
        capacity_kwh=100.0,
        power_kw=50.0,
        round_trip_efficiency=1.0,
        soc_min=0.0,
        soc_max=1.0,
    )


class TestStepResult:
    def test_net_is_revenue_minus_cost(self):
        r = StepResult(revenue_usd=10.0, cost_usd=4.0)
        assert r.net_usd == pytest.approx(6.0)

    def test_idle_is_net_zero(self):
        assert StepResult().net_usd == 0.0

    def test_throughput_sums_both_directions(self):
        r = StepResult(charged_kwh=3.0, discharged_kwh=2.0)
        assert r.throughput_kwh == pytest.approx(5.0)


class TestBatteryState:
    def test_starts_inside_bounds(self, battery):
        state = BatteryState(battery)
        assert battery.soc_min <= state.soc_fraction <= battery.soc_max

    def test_reset_returns_to_midpoint(self, battery):
        state = BatteryState(battery)
        state.step(50.0, 0.0, 1.0, 0.0)
        state.reset()
        assert state.soc_fraction == pytest.approx(0.5)

    def test_charge_at_unit_efficiency_adds_energy(self, battery):
        state = BatteryState(battery)
        state.reset(0.5)
        result = state.step(charge_kw=50.0, discharge_kw=0.0, interval_hours=1.0, price_usd_per_mwh=0.0)
        assert result.charged_kwh == pytest.approx(50.0)
        assert state.soc_kwh == pytest.approx(100.0)

    def test_efficiency_is_applied_on_charge(self):
        cfg = BatteryConfig(
            name="t", capacity_kwh=100.0, power_kw=50.0, round_trip_efficiency=0.81
        )
        state = BatteryState(cfg)
        state.reset(0.5)
        state.step(charge_kw=50.0, discharge_kw=0.0, interval_hours=1.0, price_usd_per_mwh=0.0)
        assert state.soc_kwh == pytest.approx(50.0 + 50.0 * 0.9)

    def test_efficiency_is_applied_on_discharge(self):
        cfg = BatteryConfig(
            name="t", capacity_kwh=100.0, power_kw=50.0, round_trip_efficiency=0.81
        )
        state = BatteryState(cfg)
        state.reset(1.0)
        state.step(0.0, 50.0, 1.0, 0.0)
        assert state.soc_kwh == pytest.approx(100.0 - 50.0 / 0.9)

    def test_full_charge_discharge_cycle_loses_energy(self):
        cfg = BatteryConfig(
            name="t", capacity_kwh=100.0, power_kw=100.0, round_trip_efficiency=0.81,
            soc_min=0.0, soc_max=1.0,
        )
        state = BatteryState(cfg)
        state.reset(0.0)
        charge = state.step(charge_kw=100.0, discharge_kw=0.0, interval_hours=2.0, price_usd_per_mwh=0.0)
        assert state.soc_kwh == pytest.approx(100.0)
        discharge = state.step(0.0, 100.0, 1.0, 0.0)
        assert discharge.discharged_kwh == pytest.approx(90.0)
        assert state.soc_kwh == pytest.approx(0.0)
        assert discharge.discharged_kwh == pytest.approx(
            charge.charged_kwh * cfg.round_trip_efficiency
        )

    def test_power_limit_is_enforced(self, battery):
        state = BatteryState(battery)
        state.reset(0.5)
        result = state.step(charge_kw=500.0, discharge_kw=0.0, interval_hours=1.0, price_usd_per_mwh=0.0)
        assert result.charge_kw == pytest.approx(battery.power_kw)

    def test_charge_is_clamped_at_full(self, battery):
        state = BatteryState(battery)
        state.reset(1.0)
        result = state.step(charge_kw=50.0, discharge_kw=0.0, interval_hours=1.0, price_usd_per_mwh=0.0)
        assert result.charged_kwh == pytest.approx(0.0)
        assert state.soc_kwh == pytest.approx(100.0)

    def test_discharge_is_clamped_at_empty(self, battery):
        state = BatteryState(battery)
        state.reset(0.0)
        result = state.step(0.0, 50.0, 1.0, 0.0)
        assert result.discharged_kwh == pytest.approx(0.0)
        assert state.soc_kwh == pytest.approx(0.0)

    def test_soc_never_leaves_bounds_under_charging(self, battery):
        state = BatteryState(battery)
        state.reset(0.0)
        for _ in range(20):
            state.step(50.0, 0.0, 0.25, 0.0)
            assert state.soc_kwh <= 100.0 + 1e-9
        assert state.soc_kwh == pytest.approx(100.0)

    def test_soc_never_leaves_bounds_under_discharging(self, battery):
        state = BatteryState(battery)
        state.reset(1.0)
        for _ in range(20):
            state.step(0.0, 50.0, 0.25, 0.0)
            assert state.soc_kwh >= -1e-9
        assert state.soc_kwh == pytest.approx(0.0)

    def test_reserve_is_respected(self):
        cfg = BatteryConfig(
            name="t", capacity_kwh=100.0, power_kw=50.0, soc_min=0.10, soc_max=0.90
        )
        state = BatteryState(cfg)
        state.reset(1.0)
        for _ in range(10):
            state.step(0.0, 50.0, 1.0, 0.0)
        assert state.soc_kwh == pytest.approx(10.0)

    def test_simultaneous_requests_resolve_to_discharge(self, battery):
        state = BatteryState(battery)
        state.reset(0.5)
        result = state.step(50.0, 50.0, 1.0, 0.0)
        assert result.charged_kwh == pytest.approx(0.0, abs=1e-9)
        assert result.discharged_kwh > 0

    def test_curtailment_is_reported(self, battery):
        state = BatteryState(battery)
        state.reset(1.0)
        result = state.step(50.0, 0.0, 1.0, 0.0)
        assert result.curtailed_charge_kw == pytest.approx(50.0)

    def test_negative_price_makes_charging_a_revenue_source(self, battery):
        state = BatteryState(battery)
        state.reset(0.5)
        result = state.step(50.0, 0.0, 1.0, -100.0)
        assert result.cost_usd == pytest.approx(-5.0)
        assert result.net_usd == pytest.approx(5.0)

    def test_interval_length_scales_energy(self, battery):
        state = BatteryState(battery)
        state.reset(0.0)
        result = state.step(50.0, 0.0, 0.25, 0.0)
        assert result.charged_kwh == pytest.approx(12.5)

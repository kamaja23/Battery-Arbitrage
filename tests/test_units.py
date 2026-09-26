"""Guards against the single most dangerous bug class in this project: a
mis-scaled price. ERCOT publishes USD/MWh; a home battery is specified in kWh.
"""

from __future__ import annotations

import pytest

from arb.config import (
    USD_PER_MWH_TO_USD_PER_KWH,
    BatteryConfig,
    usd_from_mwh,
)


def test_conversion_factor_is_one_thousandth():
    assert USD_PER_MWH_TO_USD_PER_KWH == 1e-3


def test_one_mwh_at_fifty_dollars_is_fifty_cents():
    assert usd_from_mwh(1000.0, 50.0) == pytest.approx(50.0)


def test_ten_kwh_at_fifty_dollars_per_mwh():
    assert usd_from_mwh(10.0, 50.0) == pytest.approx(0.50)


def test_thousand_kwh_at_fifty_dollars_per_mwh():
    assert usd_from_mwh(1000.0, 50.0) == pytest.approx(50.0)


def test_scarcity_price_dominates_a_normal_one():
    normal = usd_from_mwh(10.0, 50.0)
    scarcity = usd_from_mwh(10.0, 5000.0)
    assert scarcity == pytest.approx(100 * normal)


def test_zero_price_costs_nothing():
    assert usd_from_mwh(10.0, 0.0) == 0.0


def test_negative_price_is_a_revenue_source():
    assert usd_from_mwh(10.0, -20.0) == pytest.approx(-0.20)


class TestBatteryConfig:
    def test_efficiency_is_split_evenly(self):
        cfg = BatteryConfig(name="b", capacity_kwh=10.0, power_kw=5.0, round_trip_efficiency=0.81)
        assert cfg.eta_charge == pytest.approx(0.9)
        assert cfg.eta_discharge == pytest.approx(0.9)

    def test_eta_product_equals_round_trip(self):
        cfg = BatteryConfig(name="b", capacity_kwh=10.0, power_kw=5.0, round_trip_efficiency=0.90)
        assert cfg.eta_charge * cfg.eta_discharge == pytest.approx(cfg.round_trip_efficiency)

    def test_usable_window_excludes_reserved_soc(self):
        cfg = BatteryConfig(
            name="b", capacity_kwh=100.0, power_kw=50.0, soc_min=0.10, soc_max=0.90
        )
        assert cfg.usable_energy_kwh == pytest.approx(80.0)

    def test_duration_uses_usable_energy(self):
        cfg = BatteryConfig(
            name="b", capacity_kwh=100.0, power_kw=40.0, soc_min=0.0, soc_max=1.0
        )
        assert cfg.duration_hours == pytest.approx(2.5)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"capacity_kwh": 0.0},
            {"power_kw": -1.0},
            {"round_trip_efficiency": 0.0},
            {"round_trip_efficiency": 1.5},
            {"soc_min": 0.9, "soc_max": 0.5},
        ],
    )
    def test_rejects_impossible_parameters(self, kwargs):
        base = {"name": "b", "capacity_kwh": 10.0, "power_kw": 5.0}
        with pytest.raises(ValueError):
            BatteryConfig(**{**base, **kwargs})

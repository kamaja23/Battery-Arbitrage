from __future__ import annotations

import pandas as pd
import pytest

from arb.config import BatteryConfig, CENTRAL_TIME, RESIDENTIAL_13KWH
from arb.data.providers import PriceRequest, SyntheticProvider
from arb.sim.engine import run_backtest
from arb.strategies.perfect_foresight import (
    solve_perfect_foresight,
    run_perfect_foresight_backtest,
)
from arb.strategies.threshold import ThresholdStrategy


def series(prices: list[float], freq: str = "15min") -> pd.Series:
    idx = pd.date_range("2026-06-01", periods=len(prices), freq=freq, tz=CENTRAL_TIME)
    return pd.Series(prices, index=idx, dtype=float)


@pytest.fixture
def lossless() -> BatteryConfig:
    return BatteryConfig(
        name="t", capacity_kwh=10.0, power_kw=10.0, round_trip_efficiency=1.0,
        soc_min=0.0, soc_max=1.0,
    )


class TestOptimality:
    def test_two_interval_case_is_hand_computable(self, lossless):
        # 15-minute steps cap each move at 10 kW * 0.25 h = 2.5 kWh.
        # Start SoC is the 5 kWh midpoint; buy 2.5 kWh at $0/MWh and sell it
        # at $100/MWh = 2.5 * 100 * 0.001 = $0.25.
        sol = solve_perfect_foresight(series([0.0, 100.0]), lossless)
        assert sol.charge_kw.sum() == pytest.approx(10.0)
        assert sol.discharge_kw.sum() == pytest.approx(10.0)
        assert sol.net_usd == pytest.approx(0.25)

    def test_flat_prices_yield_zero_not_a_loss(self, lossless):
        sol = solve_perfect_foresight(series([50.0] * 10), lossless)
        assert sol.net_usd == pytest.approx(0.0, abs=1e-6)

    def test_flat_prices_stay_feasible(self, lossless):
        sol = solve_perfect_foresight(series([50.0] * 10), lossless)
        assert sol.soc_kwh.min() >= 0.0 - 1e-6
        assert sol.soc_kwh.max() <= 10.0 + 1e-6

    def test_does_nothing_when_prices_are_flat(self, lossless):
        sol = solve_perfect_foresight(series([50.0] * 10), lossless)
        assert sol.net_usd == pytest.approx(0.0, abs=1e-6)

    def test_exploits_a_rising_curve(self, lossless):
        sol = solve_perfect_foresight(series([0.0] * 4 + [1000.0] * 4), lossless)
        assert sol.net_usd > 0

    def test_exploits_negative_prices(self, lossless):
        sol = solve_perfect_foresight(series([-500.0, 100.0]), lossless)
        assert sol.net_usd > 1.00

    def test_respects_soc_bounds(self, lossless):
        sol = solve_perfect_foresight(series([0.0, 0.0, 900.0, 900.0]), lossless)
        assert sol.soc_kwh.min() >= 0.0 - 1e-6
        assert sol.soc_kwh.max() <= 10.0 + 1e-6

    def test_respects_power_limits(self, lossless):
        sol = solve_perfect_foresight(series([0.0, 500.0] * 5), lossless)
        assert sol.charge_kw.max() <= 10.0 + 1e-6
        assert sol.discharge_kw.max() <= 10.0 + 1e-6

    def test_never_charges_and_discharges_together(self, lossless):
        sol = solve_perfect_foresight(series([0.0, 500.0] * 5), lossless)
        both = (sol.charge_kw > 1e-9) & (sol.discharge_kw > 1e-9)
        assert not both.any()

    def test_respects_efficiency(self):
        lossy = BatteryConfig(
            name="t", capacity_kwh=10.0, power_kw=10.0, round_trip_efficiency=0.25,
            soc_min=0.0, soc_max=1.0,
        )
        sol = solve_perfect_foresight(series([0.0, 100.0]), lossy)
        assert sol.net_usd < 1.00

    def test_rejects_single_interval(self, lossless):
        with pytest.raises(ValueError, match="at least two"):
            solve_perfect_foresight(series([50.0]), lossless)


class TestUpperBound:
    @pytest.fixture
    def prices(self) -> pd.DataFrame:
        return SyntheticProvider(seed=5).fetch(
            PriceRequest(
                settlement_point="LZ_WEST",
                start_date="2026-06-01",
                end_date="2026-06-07",
            )
        )

    def test_lp_beats_the_threshold_rule(self, prices):
        _, lp_net = run_perfect_foresight_backtest(prices, RESIDENTIAL_13KWH)
        rule = run_backtest(
            prices,
            RESIDENTIAL_13KWH,
            ThresholdStrategy(lookback_intervals=96, min_spread_usd_per_mwh=0.0),
        )
        assert lp_net >= rule.net_usd

    def test_capture_ratio_is_a_sane_fraction(self, prices):
        _, lp_net = run_perfect_foresight_backtest(prices, RESIDENTIAL_13KWH)
        rule = run_backtest(
            prices,
            RESIDENTIAL_13KWH,
            ThresholdStrategy(lookback_intervals=96, min_spread_usd_per_mwh=0.0),
        )
        if rule.net_usd > 0:
            assert 0.0 < rule.net_usd / lp_net <= 1.0

    def test_lp_frame_is_ledger_shaped(self, prices):
        frame, _ = run_perfect_foresight_backtest(prices, RESIDENTIAL_13KWH)
        assert {"charge_kw", "discharge_kw", "charged_kwh", "discharged_kwh", "soc_kwh"} <= set(
            frame.columns
        )
        assert len(frame) == len(prices)

from __future__ import annotations

import pandas as pd
import pytest

from wattson.config import COMMERCIAL_1MW, RESIDENTIAL_13KWH, EngineConfig
from wattson.data.providers import SyntheticProvider
from wattson.sim.engine import prepare_price_series, run_backtest
from wattson.strategies.threshold import ThresholdStrategy


@pytest.fixture
def prices(rtm_request) -> pd.DataFrame:
    return SyntheticProvider().fetch(rtm_request)


@pytest.fixture
def strategy() -> ThresholdStrategy:
    return ThresholdStrategy(lookback_intervals=24, min_spread_usd_per_mwh=0.0)


class TestPreparePriceSeries:
    def test_returns_sorted_series(self, prices):
        series = prepare_price_series(prices, "RTM", "LZ_HOUSTON")
        assert series.index.is_monotonic_increasing

    def test_filters_to_requested_market(self, prices):
        series = prepare_price_series(prices, "RTM", "LZ_HOUSTON")
        assert len(series) == len(prices)

    def test_raises_on_unknown_point(self, prices):
        with pytest.raises(ValueError, match="no RTM rows"):
            prepare_price_series(prices, "RTM", "LZ_WEST")

    def test_raises_on_unknown_market(self, prices):
        with pytest.raises(ValueError, match="no DAM rows"):
            prepare_price_series(prices, "DAM", "LZ_HOUSTON")


class TestRunBacktest:
    def test_ledger_has_expected_columns(self, prices, strategy):
        result = run_backtest(prices, RESIDENTIAL_13KWH, strategy)
        assert {"charge_kw", "discharge_kw", "soc_end_kwh", "net_usd"} <= set(
            result.ledger.columns
        )

    def test_ledger_has_one_row_per_interval(self, prices, strategy):
        result = run_backtest(prices, RESIDENTIAL_13KWH, strategy)
        assert len(result.ledger) == len(prices)

    def test_soc_stays_within_bounds(self, prices, strategy):
        result = run_backtest(prices, RESIDENTIAL_13KWH, strategy)
        soc = result.ledger["soc_fraction"]
        assert soc.min() >= RESIDENTIAL_13KWH.soc_min - 1e-9
        assert soc.max() <= RESIDENTIAL_13KWH.soc_max + 1e-9

    def test_power_limits_respected(self, prices, strategy):
        result = run_backtest(prices, RESIDENTIAL_13KWH, strategy)
        assert result.ledger["charge_kw"].max() <= RESIDENTIAL_13KWH.power_kw + 1e-9
        assert result.ledger["discharge_kw"].max() <= RESIDENTIAL_13KWH.power_kw + 1e-9

    def test_energy_balance_holds_every_interval(self, prices, strategy):
        cfg = RESIDENTIAL_13KWH
        result = run_backtest(prices, cfg, strategy)
        ledger = result.ledger
        expected = (
            ledger["charged_kwh"] * cfg.eta_charge
            - ledger["discharged_kwh"] / cfg.eta_discharge
        )
        actual = ledger["soc_end_kwh"] - ledger["soc_start_kwh"]
        pd.testing.assert_series_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-6,
        )

    def test_never_charges_and_discharges_at_once(self, prices, strategy):
        result = run_backtest(prices, RESIDENTIAL_13KWH, strategy)
        both = (result.ledger["charge_kw"] > 0) & (result.ledger["discharge_kw"] > 0)
        assert not both.any()

    def test_rtm_intervals_are_fifteen_minutes(self, prices, strategy):
        result = run_backtest(prices, RESIDENTIAL_13KWH, strategy)
        deltas = result.ledger["interval_end"] - result.ledger["interval_start"]
        assert deltas.eq(pd.Timedelta(minutes=15)).all()

    def test_idle_battery_makes_no_money(self, prices, strategy):
        class Idle(ThresholdStrategy):
            name = "idle"

            def decide(self, ctx):
                from wattson.strategies.base import Action

                return Action()

        result = run_backtest(prices, RESIDENTIAL_13KWH, Idle())
        assert result.net_usd == pytest.approx(0.0)

    def test_price_floor_is_applied(self, prices, strategy):
        result = run_backtest(
            prices,
            RESIDENTIAL_13KWH,
            strategy,
            engine=EngineConfig(price_floor_usd_per_mwh=0.0),
        )
        assert result.ledger["price_usd_per_mwh"].min() >= 0.0

    def test_requires_point_when_frame_has_several(self, prices, strategy):
        other = prices.copy()
        other["settlement_point"] = "LZ_WEST"
        multi = pd.concat([prices, other], ignore_index=True)
        with pytest.raises(ValueError, match="settlement_point is required"):
            run_backtest(multi, RESIDENTIAL_13KWH, strategy)

    def test_accepts_explicit_point_in_multi_point_frame(self, prices, strategy):
        other = prices.copy()
        other["settlement_point"] = "LZ_WEST"
        multi = pd.concat([prices, other], ignore_index=True)
        result = run_backtest(
            multi, RESIDENTIAL_13KWH, strategy, settlement_point="LZ_WEST"
        )
        assert result.settlement_point == "LZ_WEST"

    def test_net_equals_revenue_minus_cost(self, prices, strategy):
        result = run_backtest(prices, RESIDENTIAL_13KWH, strategy)
        assert result.net_usd == pytest.approx(
            result.ledger["revenue_usd"].sum() - result.ledger["cost_usd"].sum()
        )

    def test_stronger_spread_rule_trades_less(self, prices):
        loose = run_backtest(
            prices,
            RESIDENTIAL_13KWH,
            ThresholdStrategy(lookback_intervals=24, min_spread_usd_per_mwh=0.0),
        )
        tight = run_backtest(
            prices,
            RESIDENTIAL_13KWH,
            ThresholdStrategy(lookback_intervals=24, min_spread_usd_per_mwh=200.0),
        )
        assert tight.ledger["discharge_kw"].sum() < loose.ledger["discharge_kw"].sum()

    def test_commercial_preset_runs(self, prices, strategy):
        result = run_backtest(prices, COMMERCIAL_1MW, strategy)
        assert result.ledger["discharge_kw"].max() == pytest.approx(
            COMMERCIAL_1MW.power_kw
        )


class TestNoLookahead:
    def test_future_prices_cannot_change_past_dispatch(self, prices, strategy):
        """Perturbing the future must not alter any earlier decision."""
        baseline = run_backtest(prices, RESIDENTIAL_13KWH, strategy).ledger

        split = len(prices) // 2
        tampered = prices.copy()
        tampered.loc[tampered.index[split:], "price_usd_per_mwh"] *= 50.0

        altered = run_backtest(tampered, RESIDENTIAL_13KWH, strategy).ledger

        cutoff = prices["interval_start"].iloc[split]
        past = altered[altered["interval_start"] < cutoff]

        pd.testing.assert_frame_equal(
            past.reset_index(drop=True),
            baseline[baseline["interval_start"] < cutoff].reset_index(drop=True),
        )

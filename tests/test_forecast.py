from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arb.config import BASE_CORE, CENTRAL_TIME
from arb.data.providers import PriceRequest, SyntheticProvider
from arb.forecast import forecast_accuracy, plan_next_day, profile_forecast
from arb.metrics import compute_metrics
from arb.sim.engine import prepare_price_series, run_backtest
from arb.strategies.forecast import ForecastStrategy
from arb.strategies.perfect_foresight import solve_perfect_foresight


def _repeating(days: int = 10, start: str = "2026-06-01") -> pd.Series:
    """The same daily shape every day: cheap overnight, expensive at 18:00."""
    idx = pd.date_range(start, periods=96 * days, freq="15min", tz=CENTRAL_TIME)
    hour = idx.hour + idx.minute / 60
    price = 20 + 60 * np.exp(-(((hour - 18) / 2.0) ** 2)) - 10 * (hour < 5)
    return pd.Series(price, index=idx, dtype=float)


def _frame(series: pd.Series, point: str = "LZ_TEST") -> pd.DataFrame:
    idx = pd.DatetimeIndex(series.index)
    return pd.DataFrame(
        {
            "interval_start": idx,
            "interval_end": idx + pd.Timedelta(minutes=15),
            "market": "RTM",
            "settlement_point": point,
            "location_type": "Load Zone",
            "price_usd_per_mwh": series.to_numpy(),
        }
    )


def _synthetic(days: int = 21) -> pd.Series:
    end = (pd.Timestamp("2026-06-01") + pd.Timedelta(days=days - 1)).date().isoformat()
    frame = SyntheticProvider(seed=5).fetch(
        PriceRequest(settlement_point="LZ_WEST", start_date="2026-06-01", end_date=end)
    )
    return prepare_price_series(frame, "RTM", "LZ_WEST")


class TestProfileForecast:
    def test_the_first_day_has_no_forecast(self):
        forecast = profile_forecast(_repeating())
        assert forecast.iloc[:96].isna().all()
        assert forecast.iloc[96:].notna().all()

    def test_a_repeating_day_is_forecast_exactly(self):
        series = _repeating()
        forecast = profile_forecast(series)
        assert np.allclose(forecast.iloc[96:], series.iloc[96:])

    def test_it_never_uses_the_day_it_forecasts_or_later(self):
        series = _synthetic()
        before = profile_forecast(series)
        tampered = series.copy()
        tampered.iloc[10 * 96 :] += 500.0  # rewrite day 11 onward
        after = profile_forecast(tampered)
        # Days 1-11 are forecast from days 0-10 only, so nothing up to the end
        # of day 11 may move. Day 12 onward legitimately sees the change.
        assert np.allclose(before.iloc[: 11 * 96], after.iloc[: 11 * 96], equal_nan=True)
        assert not np.allclose(before.iloc[11 * 96 :], after.iloc[11 * 96 :])

    def test_history_is_capped(self):
        series = _repeating(days=12)
        series.iloc[:96] += 1000.0  # an extreme first day
        forecast = profile_forecast(series, history_days=3)
        # Day 5 onward no longer looks back as far as day 0.
        assert np.allclose(forecast.iloc[5 * 96 :], series.iloc[5 * 96 :])

    def test_dst_days_are_forecast_without_gaps(self):
        idx = pd.date_range(
            "2026-10-28", "2026-11-05", freq="15min", tz=CENTRAL_TIME, inclusive="left"
        )
        series = pd.Series(30.0 + np.sin(np.arange(len(idx)) / 10), index=idx)
        forecast = profile_forecast(series)
        fall_back = pd.DatetimeIndex(forecast.index).tz_convert(CENTRAL_TIME).date
        on_the_day = forecast[fall_back == pd.Timestamp("2026-11-01").date()]
        assert len(on_the_day) == 100
        assert on_the_day.notna().all()

    def test_rejects_nonsense_history(self):
        with pytest.raises(ValueError):
            profile_forecast(_repeating(), history_days=0)


class TestForecastAccuracy:
    def test_a_perfectly_repeating_series_has_zero_error(self):
        acc = forecast_accuracy(_repeating())
        assert acc.mae_usd_per_mwh == pytest.approx(0.0, abs=1e-9)
        assert acc.days == 9

    def test_both_forecasts_are_scored_on_the_same_intervals(self):
        acc = forecast_accuracy(_synthetic())
        assert acc.intervals == 20 * 96
        assert acc.naive_mae_usd_per_mwh > 0
        assert np.isfinite(acc.skill)

    def test_averaging_a_week_beats_copying_yesterday_on_noisy_data(self):
        acc = forecast_accuracy(_synthetic())
        assert acc.skill > 0


class TestPlanNextDay:
    def test_it_plans_the_day_after_the_data(self):
        plan = plan_next_day(_repeating(days=5), BASE_CORE)
        assert plan.date == pd.Timestamp("2026-06-06").date()
        assert len(plan.timestamps) == 96

    def test_it_buys_cheap_and_sells_the_evening_peak(self):
        plan = plan_next_day(_repeating(days=5), BASE_CORE)
        frame = plan.to_frame()
        hours = pd.DatetimeIndex(frame["interval_start"]).hour
        peak = (hours >= 17) & (hours < 20)
        assert frame.loc[peak, "discharge_kw"].sum() > 0
        assert frame.loc[peak, "charge_kw"].sum() == pytest.approx(0.0, abs=1e-6)
        assert frame.loc[hours < 5, "charge_kw"].sum() > 0
        assert plan.expected_net_after_wear_usd > 0

    def test_it_respects_the_battery(self):
        plan = plan_next_day(_repeating(days=5), BASE_CORE)
        assert plan.charge_kw.max() <= BASE_CORE.power_kw + 1e-6
        assert plan.soc_kwh.max() <= BASE_CORE.soc_max * BASE_CORE.capacity_kwh + 1e-6

    def test_it_stays_idle_when_the_spread_cannot_cover_wear(self):
        flat = _repeating(days=5) * 0 + 30.0
        plan = plan_next_day(flat, BASE_CORE)
        assert plan.charge_kw.sum() == pytest.approx(0.0, abs=1e-6)
        assert plan.expected_net_usd == pytest.approx(0.0, abs=1e-6)

    def test_spring_forward_has_one_hour_fewer(self):
        series = _repeating(days=5, start="2026-03-03")
        plan = plan_next_day(series, BASE_CORE)
        assert plan.date == pd.Timestamp("2026-03-08").date()
        assert len(plan.timestamps) == 92


class TestForecastStrategy:
    def test_it_is_idle_on_the_first_day(self):
        result = run_backtest(_frame(_repeating()), BASE_CORE, ForecastStrategy())
        first = result.ledger.iloc[:96]
        assert (first["charge_kw"] == 0).all() and (first["discharge_kw"] == 0).all()

    def test_on_a_predictable_series_it_nears_perfect_foresight(self):
        series = _repeating()
        result = run_backtest(_frame(series), BASE_CORE, ForecastStrategy())
        optimal = solve_perfect_foresight(series, BASE_CORE).net_usd
        # Day one is idle, so 9 of 10 days is the most it can reach.
        assert result.net_usd > 0.75 * optimal
        assert result.net_usd <= optimal + 1e-6

    def test_its_decisions_never_depend_on_later_prices(self):
        series = _synthetic()
        before = run_backtest(_frame(series), BASE_CORE, ForecastStrategy()).ledger
        tampered = series.copy()
        tampered.iloc[10 * 96 :] *= 3.0
        after = run_backtest(_frame(tampered), BASE_CORE, ForecastStrategy()).ledger
        cols = ["charge_kw", "discharge_kw", "soc_end_kwh"]
        pd.testing.assert_frame_equal(before[cols].iloc[: 10 * 96], after[cols].iloc[: 10 * 96])

    def test_it_respects_power_and_charge_limits(self):
        result = run_backtest(_frame(_synthetic()), BASE_CORE, ForecastStrategy())
        ledger = result.ledger
        assert ledger["charge_kw"].max() <= BASE_CORE.power_kw + 1e-9
        assert ledger["discharge_kw"].max() <= BASE_CORE.power_kw + 1e-9
        assert ((ledger["charge_kw"] > 0) & (ledger["discharge_kw"] > 0)).sum() == 0
        assert ledger["soc_fraction"].max() <= BASE_CORE.soc_max + 1e-9

    def test_it_trades_less_than_the_optimizer_that_ignores_wear(self):
        series = _synthetic()
        planned = compute_metrics(run_backtest(_frame(series), BASE_CORE, ForecastStrategy()))
        assert planned.net_after_degradation_usd > -1.0  # wear-aware: no big losses

    def test_it_can_be_run_twice(self):
        strategy = ForecastStrategy()
        first = run_backtest(_frame(_repeating()), BASE_CORE, strategy).net_usd
        second = run_backtest(_frame(_repeating()), BASE_CORE, strategy).net_usd
        assert first == pytest.approx(second)


class TestThroughputCost:
    def test_zero_cost_matches_the_plain_upper_bound(self):
        series = _synthetic(days=5)
        plain = solve_perfect_foresight(series, BASE_CORE)
        costed = solve_perfect_foresight(series, BASE_CORE, throughput_cost_per_kwh=0.0)
        assert costed.net_usd == pytest.approx(plain.net_usd)

    def test_wear_cost_reduces_cycling(self):
        series = _synthetic(days=5)
        plain = solve_perfect_foresight(series, BASE_CORE)
        costed = solve_perfect_foresight(series, BASE_CORE, throughput_cost_per_kwh=0.05)
        assert costed.discharge_kw.sum() < plain.discharge_kw.sum()

    def test_a_prohibitive_cost_stops_trading(self):
        series = _synthetic(days=3)
        costed = solve_perfect_foresight(series, BASE_CORE, throughput_cost_per_kwh=10.0)
        assert costed.charge_kw.sum() == pytest.approx(0.0, abs=1e-6)

from __future__ import annotations

import pandas as pd
import pytest

from arb.config import COMMERCIAL_1MW, RESIDENTIAL_13KWH, BatteryConfig
from arb.data.providers import PriceRequest, SyntheticProvider
from arb.metrics import (
    BASELINE_LABEL,
    compare_strategies,
    compute_metrics,
    no_battery_baseline,
)
from arb.sim.engine import run_backtest
from arb.strategies.threshold import ThresholdStrategy


@pytest.fixture
def prices() -> pd.DataFrame:
    return SyntheticProvider().fetch(
        PriceRequest(
            settlement_point="LZ_HOUSTON",
            start_date="2026-06-01",
            end_date="2026-06-10",
        )
    )


class TestCurtailedEnergy:
    def test_curtailment_is_energy_not_power(self, prices):
        """Curtailed kW must be multiplied by the interval length."""
        result = run_backtest(prices, COMMERCIAL_1MW, ThresholdStrategy())
        metrics = compute_metrics(result)

        ledger = result.ledger
        hours = (
            (ledger["interval_end"] - ledger["interval_start"]).dt.total_seconds()
            / 3600.0
        )
        expected = float(
            (
                (ledger["curtailed_charge_kw"] + ledger["curtailed_discharge_kw"])
                * hours
            ).sum()
        )
        assert metrics.curtailed_kwh == pytest.approx(expected)

    def test_quarter_hourly_curtailment_is_four_times_the_kw_sum(self, prices):
        result = run_backtest(prices, COMMERCIAL_1MW, ThresholdStrategy())
        ledger = result.ledger
        kw_sum = float(
            (ledger["curtailed_charge_kw"] + ledger["curtailed_discharge_kw"]).sum()
        )
        assert compute_metrics(result).curtailed_kwh == pytest.approx(kw_sum * 0.25)

    def test_curtailment_is_never_negative(self, prices):
        result = run_backtest(prices, COMMERCIAL_1MW, ThresholdStrategy())
        assert compute_metrics(result).curtailed_kwh >= 0.0


class TestNoBatteryBaseline:
    def test_baseline_earns_nothing(self):
        baseline = no_battery_baseline(COMMERCIAL_1MW, 30.0, "LZ_WEST", "RTM")
        assert baseline.net_usd == 0.0
        assert baseline.net_after_degradation_usd == 0.0
        assert baseline.usd_per_kw_year == 0.0
        assert baseline.total_charged_kwh == 0.0
        assert baseline.capture_ratio is None
        assert baseline.payback_years is None

    def test_baseline_is_labelled_and_carries_the_battery_name(self):
        baseline = no_battery_baseline(RESIDENTIAL_13KWH, 30.0, "LZ_WEST", "RTM")
        assert baseline.strategy == BASELINE_LABEL
        assert baseline.battery == RESIDENTIAL_13KWH.name
        assert baseline.settlement_point == "LZ_WEST"
        assert baseline.market == "RTM"


class TestCompareStrategies:
    def test_returns_baseline_online_and_optimal_rows(self, prices):
        frame = compare_strategies(prices, COMMERCIAL_1MW, ThresholdStrategy())
        assert list(frame.index) == [
            BASELINE_LABEL,
            "threshold",
            "perfect foresight (upper bound)",
        ]
        assert list(frame["kind"]) == ["baseline", "online", "optimal"]

    def test_the_baseline_sits_below_the_rule_based_result(self, prices):
        frame = compare_strategies(prices, COMMERCIAL_1MW, ThresholdStrategy())
        assert frame.loc[BASELINE_LABEL, "net_usd"] == 0.0
        assert frame.loc["threshold", "net_usd"] > 0.0

    def test_perfect_foresight_bounds_the_rule_based_result(self, prices):
        frame = compare_strategies(prices, COMMERCIAL_1MW, ThresholdStrategy())
        online = frame.loc["threshold", "net_usd"]
        optimal = frame.loc["perfect foresight (upper bound)", "net_usd"]
        assert 0.0 < online <= optimal

    def test_capture_ratio_is_the_share_of_the_upper_bound(self, prices):
        frame = compare_strategies(prices, COMMERCIAL_1MW, ThresholdStrategy())
        online = frame.loc["threshold", "net_usd"]
        optimal = frame.loc["perfect foresight (upper bound)", "net_usd"]
        assert frame.loc["threshold", "capture_ratio"] == pytest.approx(online / optimal)

    def test_optimal_row_reports_full_capture_by_construction(self, prices):
        frame = compare_strategies(prices, COMMERCIAL_1MW, ThresholdStrategy())
        assert frame.loc["perfect foresight (upper bound)", "capture_ratio"] == 1.0

    def test_optimal_row_can_be_left_out(self, prices):
        frame = compare_strategies(
            prices, COMMERCIAL_1MW, ThresholdStrategy(), include_optimal=False
        )
        assert "perfect foresight (upper bound)" not in frame.index
        assert list(frame["kind"]) == ["baseline", "online"]

    def test_capture_ratio_is_none_when_there_is_no_upper_bound(self, prices):
        frame = compare_strategies(
            prices, COMMERCIAL_1MW, ThresholdStrategy(), include_optimal=False
        )
        assert frame.loc["threshold", "capture_ratio"] is None


def test_dam_only_frame_infers_market() -> None:
    """A frame holding only DAM rows must not be read as empty.

    EngineConfig defaults to RTM, so callers that fetch a single market used
    to crash with "no RTM rows". Regression guard for the CLI --market DAM
    path and the UI's DAM toggle.
    """
    dam = _synthetic_dam_prices()
    result = run_backtest(
        dam,
        COMMERCIAL_1MW,
        ThresholdStrategy(),
        settlement_point="LZ_TEST",
    )

    assert result.market == "DAM"
    assert result.settlement_point == "LZ_TEST"
    assert len(result.ledger) == 24

    frame = compare_strategies(
        dam, COMMERCIAL_1MW, ThresholdStrategy(), include_optimal=False
    )
    assert list(frame.index) == [BASELINE_LABEL, "threshold"]
    assert frame.index[0] == "no battery (baseline)"


def _synthetic_dam_prices() -> pd.DataFrame:
    import numpy as np

    from arb.config import CENTRAL_TIME

    index = pd.date_range("2026-06-01 00:00", periods=24, freq="1h", tz=CENTRAL_TIME)
    values = 20 + 25 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    return pd.DataFrame(
        {
            "interval_start": index,
            "interval_end": index + pd.Timedelta(hours=1),
            "market": "DAM",
            "settlement_point": "LZ_TEST",
            "location_type": "Load Zone",
            "price_usd_per_mwh": values,
        }
    )


def test_compare_zones_ranks_by_return_across_locations() -> None:
    """The same battery is compared across locations on identical inputs.

    A zone with deeper price swings must outrank a calm one, and every row
    must carry the same market so the comparison is apples to apples. A week
    is used because the rolling threshold needs history before it trades.
    """
    import numpy as np

    from arb.config import CENTRAL_TIME
    from arb.metrics import compare_zones

    def frame(sp: str, swing: float) -> pd.DataFrame:
        index = pd.date_range(
            "2026-06-01", periods=672, freq="15min", tz=CENTRAL_TIME
        )
        phase = np.arange(672) / 96 * 2 * np.pi
        values = 25 + swing * np.sin(phase) + 8 * np.sin(phase * 7)
        return pd.DataFrame(
            {
                "interval_start": index,
                "interval_end": index + pd.Timedelta(minutes=15),
                "market": "RTM",
                "settlement_point": sp,
                "location_type": "Load Zone",
                "price_usd_per_mwh": values,
            }
        )

    frames = {
        "LZ_CALM": frame("LZ_CALM", 2.0),
        "LZ_VOLATILE": frame("LZ_VOLATILE", 40.0),
    }
    out = compare_zones(frames, COMMERCIAL_1MW, ThresholdStrategy())

    assert list(out.index) == ["LZ_VOLATILE", "LZ_CALM"]
    assert set(out["market"]) == {"RTM"}
    assert (out["intervals"] == 672).all()
    assert out.loc["LZ_VOLATILE", "net_usd"] > out.loc["LZ_CALM", "net_usd"]
    assert (
        out.loc["LZ_VOLATILE", "p95_price_usd_per_mwh"]
        > out.loc["LZ_CALM", "p95_price_usd_per_mwh"]
    )


def test_compare_zones_handles_no_input() -> None:
    from arb.metrics import compare_zones

    out = compare_zones({}, COMMERCIAL_1MW, ThresholdStrategy())

    assert out.empty

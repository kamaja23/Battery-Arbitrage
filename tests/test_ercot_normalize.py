from __future__ import annotations

import pandas as pd
import pytest

from arb.config import CENTRAL_TIME
from arb.data.ercot_source import ErcotLiveProvider
from arb.data.providers import PriceRequest


def ercot_long_frame(ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Shape raw input the way the ``ercot`` package returns it."""
    return pd.DataFrame(
        {
            "date": ends.normalize().tz_localize(None),
            "settlement_point": ["LZ_HOUSTON"] * len(ends),
            "interval": range(1, len(ends) + 1),
            "interval_ending": ends,
            "price": [25.0] * len(ends),
        }
    )


def request_for(market: str = "RTM") -> PriceRequest:
    return PriceRequest(
        settlement_point="LZ_HOUSTON",
        start_date="2026-06-01",
        end_date="2026-06-02",
        market=market,
    )


class TestNormalize:
    def test_maps_to_the_contract_columns(self):
        ends = pd.date_range("2026-06-01", periods=4, freq="15min", tz=CENTRAL_TIME)
        out = ErcotLiveProvider.normalize(ercot_long_frame(ends), request_for())
        assert list(out.columns) == [
            "interval_start",
            "interval_end",
            "market",
            "settlement_point",
            "location_type",
            "price_usd_per_mwh",
        ]

    def test_interval_start_is_one_step_before_end_for_rtm(self):
        ends = pd.date_range("2026-06-01", periods=4, freq="15min", tz=CENTRAL_TIME)
        out = ErcotLiveProvider.normalize(ercot_long_frame(ends), request_for())
        assert (out["interval_end"] - out["interval_start"]).eq(
            pd.Timedelta(minutes=15)
        ).all()

    def test_interval_start_is_one_hour_before_end_for_dam(self):
        ends = pd.date_range("2026-06-01", periods=4, freq="h", tz=CENTRAL_TIME)
        out = ErcotLiveProvider.normalize(ercot_long_frame(ends), request_for("DAM"))
        assert (out["interval_end"] - out["interval_start"]).eq(
            pd.Timedelta(hours=1)
        ).all()

    def test_labels_load_zone(self):
        ends = pd.date_range("2026-06-01", periods=2, freq="15min", tz=CENTRAL_TIME)
        out = ErcotLiveProvider.normalize(ercot_long_frame(ends), request_for())
        assert set(out["location_type"]) == {"Load Zone"}

    def test_labels_hub(self):
        ends = pd.date_range("2026-06-01", periods=2, freq="15min", tz=CENTRAL_TIME)
        raw = ercot_long_frame(ends)
        raw["settlement_point"] = "HB_NORTH"
        req = PriceRequest(
            settlement_point="HB_NORTH",
            start_date="2026-06-01",
            end_date="2026-06-02",
        )
        out = ErcotLiveProvider.normalize(raw, req)
        assert set(out["location_type"]) == {"Hub"}

    def test_rejects_resource_nodes(self):
        ends = pd.date_range("2026-06-01", periods=2, freq="15min", tz=CENTRAL_TIME)
        raw = ercot_long_frame(ends)
        raw["settlement_point"] = "HZ_NORTH"
        req = PriceRequest(
            settlement_point="HZ_NORTH",
            start_date="2026-06-01",
            end_date="2026-06-02",
        )
        with pytest.raises(ValueError, match="only supports Load Zone and Hub"):
            ErcotLiveProvider.normalize(raw, req)

    def test_rejects_empty_response(self):
        raw = pd.DataFrame(columns=["date", "settlement_point", "interval", "interval_ending", "price"])
        with pytest.raises(ValueError, match="no rows"):
            ErcotLiveProvider.normalize(raw, request_for())

    def test_localizes_naive_timestamps(self):
        ends = pd.date_range("2026-06-01", periods=2, freq="15min")
        out = ErcotLiveProvider.normalize(ercot_long_frame(ends), request_for())
        assert str(out["interval_end"].dt.tz) == CENTRAL_TIME

    def test_drops_null_prices(self):
        ends = pd.date_range("2026-06-01", periods=4, freq="15min", tz=CENTRAL_TIME)
        raw = ercot_long_frame(ends)
        raw.loc[1, "price"] = None
        out = ErcotLiveProvider.normalize(raw, request_for())
        assert len(out) == 3

    def test_output_is_sorted_by_time(self):
        ends = pd.date_range("2026-06-01", periods=4, freq="15min", tz=CENTRAL_TIME)[::-1]
        out = ErcotLiveProvider.normalize(ercot_long_frame(ends), request_for())
        assert out["interval_start"].is_monotonic_increasing


class TestDaylightSavingTransitions:
    """2026: DST starts Mar 8 (23h -> 92 RTM intervals) and ends Nov 1
    (25h -> 100 RTM intervals)."""

    @pytest.mark.parametrize(
        "day,expected",
        [
            ("2026-03-08", 92),
            ("2026-06-15", 96),
            ("2026-11-01", 100),
        ],
    )
    def test_interval_counts_match_ercot(self, day, expected):
        starts = pd.date_range(f"{day} 00:00", periods=expected, freq="15min", tz=CENTRAL_TIME)
        ends = starts + pd.Timedelta(minutes=15)
        out = ErcotLiveProvider.normalize(ercot_long_frame(ends), request_for())
        assert len(out) == expected

    def test_spring_forward_skips_the_missing_hour(self):
        starts = pd.date_range("2026-03-08 00:00", periods=92, freq="15min", tz=CENTRAL_TIME)
        out = ErcotLiveProvider.normalize(
            ercot_long_frame(starts + pd.Timedelta(minutes=15)), request_for()
        )
        hours = set(out["interval_start"].dt.hour)
        assert 2 not in hours

    def test_fall_back_repeats_the_ambiguous_hour(self):
        starts = pd.date_range("2026-11-01 00:00", periods=100, freq="15min", tz=CENTRAL_TIME)
        out = ErcotLiveProvider.normalize(
            ercot_long_frame(starts + pd.Timedelta(minutes=15)), request_for()
        )
        assert (out["interval_start"].dt.hour == 1).sum() > 4

    def test_timestamps_stay_distinct_across_the_transition(self):
        starts = pd.date_range("2026-11-01 00:00", periods=100, freq="15min", tz=CENTRAL_TIME)
        out = ErcotLiveProvider.normalize(
            ercot_long_frame(starts + pd.Timedelta(minutes=15)), request_for()
        )
        assert out["interval_start"].is_unique

from __future__ import annotations

import pandas as pd
import pytest

from arb.config import CENTRAL_TIME
from arb.data.ercot_source import ErcotLiveProvider, infer_location_type
from arb.data.providers import PriceRequest

# Shapes below mirror what api.ercot.com actually returns, verified live:
#   RTM -> 7 positional columns, interval numbered 1-4 *within* each hour,
#          and a load-zone filter still returns LZ and LZEW rows.
#   DAM -> 5 positional columns, hour ending as a 'HH:00' string, no LZEW rows.
SPRING_HOURS = [h for h in range(1, 25) if h != 3]

RTM_WIDTH = 7
DAM_WIDTH = 5


def raw_rtm(
    day: str,
    hours: range = range(1, 25),
    repeated_hour: int | None = None,
    missing: tuple[int, int] | None = None,
    types: tuple[str, ...] = ("LZ", "LZEW"),
    point: str = "LZ_WEST",
    price: float = 25.0,
) -> pd.DataFrame:
    """Build a raw RTM response; ``missing`` drops one (hour, interval)."""
    rows: list[list[object]] = []
    for hour in hours:
        repeats = (False, True) if hour == repeated_hour else (False,)
        for repeat in repeats:
            for interval in range(1, 5):
                if missing is not None and (hour, interval) == missing:
                    continue
                for kind in types:
                    rows.append([day, hour, interval, point, kind, price, repeat])
    return pd.DataFrame(rows, columns=range(RTM_WIDTH))


def raw_dam(
    day: str,
    hours: range = range(1, 25),
    repeated_hour: int | None = None,
    missing_hour: int | None = None,
    point: str = "LZ_WEST",
    price: float = 45.0,
) -> pd.DataFrame:
    rows: list[list[object]] = []
    for hour in hours:
        if hour == missing_hour:
            continue
        repeats = (False, True) if hour == repeated_hour else (False,)
        for repeat in repeats:
            rows.append([day, f"{hour:02d}:00", point, price, repeat])
    return pd.DataFrame(rows, columns=range(DAM_WIDTH))


def request_for(market: str = "RTM", point: str = "LZ_WEST") -> PriceRequest:
    return PriceRequest(
        settlement_point=point,
        start_date="2026-06-15",
        end_date="2026-06-15",
        market=market,
    )


class TestRawShape:
    def test_rejects_an_unexpected_column_count(self):
        with pytest.raises(ValueError, match="positional columns"):
            ErcotLiveProvider.from_raw(raw_rtm("2026-06-15").iloc[:, :5], request_for())

    def test_rejects_an_empty_response(self):
        with pytest.raises(ValueError, match="no rows"):
            ErcotLiveProvider.from_raw(pd.DataFrame(columns=range(RTM_WIDTH)), request_for())

    def test_drops_unparseable_prices(self):
        raw = raw_rtm("2026-06-15", types=("LZ",))
        raw.loc[0, 5] = None
        out = ErcotLiveProvider.from_raw(raw, request_for())
        assert len(out) == 95


class TestLoadZoneFiltering:
    def test_a_load_zone_filter_still_returns_lz_and_lzew_rows(self):
        raw = raw_rtm("2026-06-15")
        assert set(raw[4]) == {"LZ", "LZEW"}
        assert len(raw) == 192

    def test_keeps_only_the_load_zone_rows(self):
        out = ErcotLiveProvider.from_raw(raw_rtm("2026-06-15"), request_for())
        assert len(out) == 96
        assert set(out["location_type"]) == {"Load Zone"}

    def test_a_load_zone_can_never_return_hub_rows(self):
        raw = raw_rtm("2026-06-15", types=("HU",))
        with pytest.raises(ValueError, match="no LZ rows"):
            ErcotLiveProvider.from_raw(raw, request_for())


class TestIntervalNumbering:
    """``interval`` runs 1-4 inside an hour, not 1-96 across the day."""

    def test_matches_a_plain_quarter_hourly_range(self):
        out = ErcotLiveProvider.from_raw(raw_rtm("2026-06-15"), request_for())
        expected = pd.date_range(
            "2026-06-15", periods=96, freq="15min", tz=CENTRAL_TIME
        ) + pd.Timedelta(minutes=15)
        assert list(out["interval_end"]) == list(expected)

    def test_ignores_the_row_order_of_the_response(self):
        shuffled = raw_rtm("2026-06-15").sample(frac=1.0, random_state=3)
        out = ErcotLiveProvider.from_raw(shuffled, request_for())
        assert out["interval_start"].is_monotonic_increasing
        assert len(out) == 96


class TestDaylightSaving:
    def test_a_normal_day_spans_24_hours(self):
        out = ErcotLiveProvider.from_raw(raw_rtm("2026-06-15"), request_for())
        assert len(out) == 96
        assert out["interval_start"].iloc[-1] + pd.Timedelta(minutes=15) == pd.Timestamp(
            "2026-06-16", tz=CENTRAL_TIME
        )

    def test_spring_forward_skips_an_hour_and_spans_23(self):
        out = ErcotLiveProvider.from_raw(
            raw_rtm("2026-03-08", hours=SPRING_HOURS), request_for()
        )
        assert len(out) == 92
        span = out["interval_end"].iloc[-1] - out["interval_start"].iloc[0]
        assert span == pd.Timedelta(hours=23)
        assert out["interval_end"].iloc[-1] == pd.Timestamp(
            "2026-03-09", tz=CENTRAL_TIME
        )

    def test_fall_back_repeats_an_hour_and_spans_25(self):
        out = ErcotLiveProvider.from_raw(
            raw_rtm("2025-11-02", repeated_hour=2), request_for()
        )
        assert len(out) == 100
        span = out["interval_end"].iloc[-1] - out["interval_start"].iloc[0]
        assert span == pd.Timedelta(hours=25)
        assert out["interval_end"].iloc[-1] == pd.Timestamp(
            "2025-11-03", tz=CENTRAL_TIME
        )

    def test_the_repeated_hour_lands_on_both_offsets(self):
        out = ErcotLiveProvider.from_raw(
            raw_rtm("2025-11-02", repeated_hour=2), request_for()
        )
        one_oc = out[out["interval_start"].dt.hour == 1]
        assert len(one_oc) == 8
        assert {str(t.utcoffset()) for t in one_oc["interval_start"]} == {
            "-1 day, 19:00:00",
            "-1 day, 18:00:00",
        }

    def test_timestamps_stay_unique_and_contiguous_across_transitions(self):
        for day, repeated, missing in (
            ("2026-03-08", None, 3),
            ("2025-11-02", 2, None),
        ):
            out = ErcotLiveProvider.from_raw(
                raw_rtm(
                day, hours=SPRING_HOURS if missing else range(1, 25), repeated_hour=repeated
            ),
                request_for(),
            )
            assert out["interval_start"].is_unique
            assert (out["interval_end"].iloc[:-1].to_numpy() == out[
                "interval_start"
            ].iloc[1:].to_numpy()).all()


class TestIncompleteDays:
    """A day missing an interval must not shift every later timestamp."""

    def test_later_rows_keep_their_place_on_the_timeline(self):
        complete = ErcotLiveProvider.from_raw(
            raw_rtm("2026-06-04"), request_for()
        )
        gapped = ErcotLiveProvider.from_raw(
            raw_rtm("2026-06-04", missing=(11, 3)), request_for()
        )
        assert len(complete) == 96
        assert len(gapped) == 95
        missing_stamp = set(complete["interval_end"]) - set(gapped["interval_end"])
        assert len(missing_stamp) == 1
        assert list(gapped["interval_end"]) == [
            t for t in complete["interval_end"] if t not in missing_stamp
        ]
        assert gapped["interval_end"].iloc[-1] == complete["interval_end"].iloc[-1]


class TestDam:
    def test_a_normal_day_is_24_hourly_intervals(self):
        out = ErcotLiveProvider.from_raw(raw_dam("2026-06-15"), request_for("DAM"))
        assert len(out) == 24
        assert (out["interval_end"] - out["interval_start"]).eq(
            pd.Timedelta(hours=1)
        ).all()

    def test_spring_forward_spans_23_hours(self):
        out = ErcotLiveProvider.from_raw(
            raw_dam("2026-03-08", missing_hour=2), request_for("DAM")
        )
        assert len(out) == 23
        assert out["interval_end"].iloc[-1] - out["interval_start"].iloc[0] == (
            pd.Timedelta(hours=23)
        )

    def test_fall_back_spans_25_hours(self):
        out = ErcotLiveProvider.from_raw(
            raw_dam("2025-11-02", repeated_hour=2), request_for("DAM")
        )
        assert len(out) == 25
        assert out["interval_end"].iloc[-1] - out["interval_start"].iloc[0] == (
            pd.Timedelta(hours=25)
        )

    def test_timestamps_stay_unique(self):
        out = ErcotLiveProvider.from_raw(
            raw_dam("2025-11-02", repeated_hour=2), request_for("DAM")
        )
        assert out["interval_start"].is_unique


class TestLocationTypes:
    def test_load_zone_names(self):
        assert infer_location_type("LZ_WEST") == "Load Zone"

    def test_hub_names(self):
        assert infer_location_type("HB_NORTH") == "Hub"

    def test_resource_nodes_are_rejected(self):
        with pytest.raises(ValueError, match="only supports Load Zone and Hub"):
            infer_location_type("HZ_NORTH")

    def test_hubs_are_labelled_and_kept(self):
        raw = raw_rtm("2026-06-15", point="HB_NORTH", types=("HU", "HU_EW"))
        out = ErcotLiveProvider.from_raw(
            raw, request_for(point="HB_NORTH")
        )
        assert set(out["location_type"]) == {"Hub"}
        assert len(out) == 96

from __future__ import annotations

import pandas as pd
import pytest

from arb.config import CENTRAL_TIME
from arb.data.providers import (
    PRICE_COLUMNS,
    SyntheticProvider,
    validate_price_frame,
)


class TestValidatePriceFrame:
    def test_accepts_a_good_frame(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        assert validate_price_frame(frame) is frame

    def test_rejects_missing_columns(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request).drop(columns=["price_usd_per_mwh"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_price_frame(frame)

    def test_rejects_empty_frame(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        with pytest.raises(ValueError, match="empty"):
            validate_price_frame(frame.iloc[0:0])

    def test_rejects_naive_timestamps(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        frame["interval_start"] = frame["interval_start"].dt.tz_localize(None)
        with pytest.raises(ValueError, match="tz-aware"):
            validate_price_frame(frame)

    def test_rejects_wrong_timezone(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        frame["interval_start"] = frame["interval_start"].dt.tz_convert("UTC")
        with pytest.raises(ValueError, match="America/Chicago"):
            validate_price_frame(frame)

    def test_rejects_unknown_market(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        frame["market"] = "REAL_TIME"
        with pytest.raises(ValueError, match="invalid market"):
            validate_price_frame(frame)

    def test_rejects_unknown_location_type(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        frame["location_type"] = "Resource Node"
        with pytest.raises(ValueError, match="invalid location_type"):
            validate_price_frame(frame)

    def test_rejects_non_numeric_price(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        frame["price_usd_per_mwh"] = "cheap"
        with pytest.raises(ValueError, match="numeric"):
            validate_price_frame(frame)

    def test_rejects_duplicate_keys(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            validate_price_frame(doubled)


class TestSyntheticProvider:
    def test_emits_exactly_the_contract_columns(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        assert list(frame.columns) == list(PRICE_COLUMNS)

    def test_rtm_has_four_hour_intervals(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        deltas = frame["interval_end"] - frame["interval_start"]
        assert deltas.eq(pd.Timedelta(hours=0.25)).all()

    def test_dam_has_one_hour_intervals(self, dam_request):
        frame = SyntheticProvider().fetch(dam_request)
        deltas = frame["interval_end"] - frame["interval_start"]
        assert deltas.eq(pd.Timedelta(hours=1.0)).all()

    def test_row_count_matches_requested_days(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        assert len(frame) == 3 * 96

    def test_is_deterministic_for_a_seed(self, rtm_request):
        a = SyntheticProvider(seed=11).fetch(rtm_request)
        b = SyntheticProvider(seed=11).fetch(rtm_request)
        pd.testing.assert_frame_equal(a, b)

    def test_differs_across_seeds(self, rtm_request):
        a = SyntheticProvider(seed=1).fetch(rtm_request)
        b = SyntheticProvider(seed=2).fetch(rtm_request)
        assert not a["price_usd_per_mwh"].equals(b["price_usd_per_mwh"])

    def test_prices_are_plausible_for_ercot(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        assert frame["price_usd_per_mwh"].median() > 0
        assert frame["price_usd_per_mwh"].quantile(0.95) < 500

    def test_evening_is_more_expensive_than_overnight(self):
        from arb.data.providers import PriceRequest

        req = PriceRequest(
            settlement_point="LZ_WEST",
            start_date="2026-01-01",
            end_date="2026-01-28",
        )
        frame = SyntheticProvider(seed=3).fetch(req)
        hour = frame["interval_start"].dt.hour
        peak = frame.loc[hour.isin([18, 19, 20, 21]), "price_usd_per_mwh"].median()
        trough = frame.loc[hour.isin([2, 3, 4, 5]), "price_usd_per_mwh"].median()
        assert peak > 2.5 * trough

    def test_timestamps_are_central_time(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        assert str(frame["interval_start"].dt.tz) == CENTRAL_TIME

    def test_start_and_end_dates_are_both_inclusive(self, rtm_request):
        frame = SyntheticProvider().fetch(rtm_request)
        assert frame["interval_start"].min().date() == pd.Timestamp("2026-06-01").date()
        assert frame["interval_start"].max().date() == pd.Timestamp("2026-06-03").date()

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from arb.config import CENTRAL_TIME
from arb.data.cache import cached_fetch, cache_path, load, save
from arb.data.providers import SyntheticProvider


def test_missing_cache_returns_none(rtm_request, tmp_path):
    assert load(rtm_request, tmp_path) is None


def test_round_trip_preserves_values(rtm_request, tmp_path):
    original = SyntheticProvider().fetch(rtm_request)
    save(rtm_request, original, tmp_path)
    restored = load(rtm_request, tmp_path)
    assert restored is not None
    pd.testing.assert_frame_equal(
        restored.reset_index(drop=True), original.reset_index(drop=True)
    )


def test_round_trip_preserves_central_timezone(rtm_request, tmp_path):
    original = SyntheticProvider().fetch(rtm_request)
    save(rtm_request, original, tmp_path)
    restored = load(rtm_request, tmp_path)
    assert str(restored["interval_start"].dt.tz) == CENTRAL_TIME
    assert str(restored["interval_end"].dt.tz) == CENTRAL_TIME


def test_round_trip_survives_a_dst_short_day():
    from arb.data.providers import PriceRequest

    request = PriceRequest(
        settlement_point="LZ_NORTH",
        start_date="2026-03-08",
        end_date="2026-03-08",
        market="RTM",
    )
    original = SyntheticProvider().fetch(request)
    with TemporaryDirectory() as tmp:
        save(request, original, Path(tmp))
        restored = load(request, Path(tmp))
    assert len(restored) == len(original)
    pd.testing.assert_frame_equal(
        restored.reset_index(drop=True), original.reset_index(drop=True)
    )


def test_cache_path_is_stable_and_descriptive(rtm_request, tmp_path):
    path = cache_path(rtm_request, tmp_path)
    assert path.name == "RTM_LZ_HOUSTON_2026-06-01_2026-06-03.csv"
    assert cache_path(rtm_request, tmp_path) == path


def test_dam_and_rtm_do_not_collide(dam_request, rtm_request, tmp_path):
    assert cache_path(dam_request, tmp_path) != cache_path(rtm_request, tmp_path)


def test_cached_fetch_populates_then_reuses(rtm_request, tmp_path):
    class OneShotProvider:
        calls = 0

        def fetch(self, request):
            OneShotProvider.calls += 1
            return SyntheticProvider().fetch(request)

    first = cached_fetch(OneShotProvider(), rtm_request, tmp_path)
    second = cached_fetch(OneShotProvider(), rtm_request, tmp_path)

    assert OneShotProvider.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_refresh_bypasses_the_cache(rtm_request, tmp_path):
    class CountingProvider:
        calls = 0

        def fetch(self, request):
            CountingProvider.calls += 1
            return SyntheticProvider().fetch(request)

    cached_fetch(CountingProvider(), rtm_request, tmp_path)
    cached_fetch(CountingProvider(), rtm_request, tmp_path, refresh=True)
    assert CountingProvider.calls == 2


def test_save_creates_missing_directories(rtm_request, tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    save(rtm_request, SyntheticProvider().fetch(rtm_request), nested)
    assert cache_path(rtm_request, nested).exists()


class TestCachedWindows:
    def test_it_lists_windows_for_one_point_and_market(self, tmp_path):
        import datetime as dt

        from arb.data.cache import cached_windows

        for name in (
            "RTM_LZ_AEN_2026-06-01_2026-06-30.csv",
            "RTM_LZ_AEN_2026-08-26_2026-09-24.csv",
            "DAM_LZ_AEN_2026-09-01_2026-09-24.csv",
            "RTM_LZ_AENX_2026-09-01_2026-09-24.csv",
            "RTM_LZ_AEN_notes.csv",
        ):
            (tmp_path / name).write_text("")
        windows = cached_windows("LZ_AEN", "RTM", tmp_path)
        assert windows == [
            (dt.date(2026, 6, 1), dt.date(2026, 6, 30)),
            (dt.date(2026, 8, 26), dt.date(2026, 9, 24)),
        ]

    def test_an_empty_or_missing_cache_is_fine(self, tmp_path):
        from arb.data.cache import cached_windows

        assert cached_windows("LZ_AEN", "RTM", tmp_path / "nope") == []

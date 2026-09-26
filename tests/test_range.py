from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from wattson.config import CENTRAL_TIME
from wattson.data.cache import cached_windows, expected_intervals, load_range, save
from wattson.data.providers import PriceRequest, SyntheticProvider

D = dt.date
TODAY = D(2026, 9, 26)


class FakeERCOT:
    """Synthetic prices, a call log, and 'today' published only up to noon."""

    def __init__(self, today: dt.date = TODAY, fail_today: bool = False):
        self.calls: list[tuple[str, str]] = []
        self.today = today
        self.fail_today = fail_today

    def __call__(self, request: PriceRequest) -> pd.DataFrame:
        self.calls.append((request.start_date, request.end_date))
        if request.start_date == self.today.isoformat():
            if self.fail_today:
                raise ConnectionError("no network")
            frame = SyntheticProvider(seed=1).fetch(request)
            noon = pd.Timestamp(self.today).tz_localize(CENTRAL_TIME) + pd.Timedelta(hours=12)
            return frame[frame["interval_start"] < noon]
        return SyntheticProvider(seed=1).fetch(request)


def _seed(tmp_path, start: str, end: str, drop_last: int = 0, saved_at: str | None = None,
          drop_times: tuple[str, ...] = ()) -> None:
    import os

    request = PriceRequest(settlement_point="LZ_AEN", start_date=start, end_date=end)
    frame = SyntheticProvider(seed=1).fetch(request)
    if drop_last:
        frame = frame.iloc[:-drop_last]
    for when in drop_times:
        frame = frame[frame["interval_start"] != pd.Timestamp(when, tz=CENTRAL_TIME)]
    path = save(request, frame, tmp_path)
    if saved_at:
        stamp = pd.Timestamp(saved_at, tz=CENTRAL_TIME).timestamp()
        os.utime(path, (stamp, stamp))


def _days(frame: pd.DataFrame) -> list[dt.date]:
    return sorted(set(frame["interval_start"].dt.tz_convert(CENTRAL_TIME).dt.date))


class TestExpectedIntervals:
    @pytest.mark.parametrize(
        ("day", "market", "n"),
        [
            (D(2026, 6, 15), "RTM", 96),
            (D(2026, 3, 8), "RTM", 92),
            (D(2025, 11, 2), "RTM", 100),
            (D(2026, 6, 15), "DAM", 24),
            (D(2026, 3, 8), "DAM", 23),
            (D(2025, 11, 2), "DAM", 25),
        ],
    )
    def test_dst_days_have_the_right_count(self, day, market, n):
        assert expected_intervals(day, market) == n


class TestFromDisk:
    def test_it_stitches_a_range_from_several_cached_months(self, tmp_path):
        _seed(tmp_path, "2026-07-01", "2026-07-31")
        _seed(tmp_path, "2026-08-01", "2026-08-31")
        result = load_range("LZ_AEN", "RTM", D(2026, 7, 20), D(2026, 8, 10), today=TODAY, cache_dir=tmp_path)
        assert _days(result.frame) == [D(2026, 7, 20) + dt.timedelta(days=i) for i in range(22)]
        assert result.frame["interval_start"].is_unique
        assert result.missing_days == ()

    def test_overlapping_cached_files_are_not_double_counted(self, tmp_path):
        _seed(tmp_path, "2026-08-01", "2026-08-31")
        _seed(tmp_path, "2026-08-26", "2026-09-24")
        result = load_range("LZ_AEN", "RTM", D(2026, 8, 20), D(2026, 9, 5), today=TODAY, cache_dir=tmp_path)
        assert len(result.frame) == 17 * 96

    def test_nothing_is_fetched_when_everything_is_cached(self, tmp_path):
        _seed(tmp_path, "2026-08-01", "2026-08-31")
        fake = FakeERCOT()
        load_range("LZ_AEN", "RTM", D(2026, 8, 1), D(2026, 8, 31), today=TODAY, fetch=fake, cache_dir=tmp_path)
        assert fake.calls == []


class TestDownloading:
    def test_missing_days_are_fetched_a_month_at_a_time_and_kept(self, tmp_path):
        fake = FakeERCOT()
        result = load_range("LZ_AEN", "RTM", D(2026, 6, 20), D(2026, 7, 10), today=TODAY, fetch=fake, cache_dir=tmp_path)
        assert fake.calls == [("2026-06-01", "2026-06-30"), ("2026-07-01", "2026-07-31")]
        assert result.missing_days == ()
        assert (D(2026, 6, 1), D(2026, 6, 30)) in cached_windows("LZ_AEN", "RTM", tmp_path)

        again = FakeERCOT()
        load_range("LZ_AEN", "RTM", D(2026, 6, 20), D(2026, 7, 10), today=TODAY, fetch=again, cache_dir=tmp_path)
        assert again.calls == []

    def test_the_current_month_is_stored_only_up_to_yesterday(self, tmp_path):
        fake = FakeERCOT()
        load_range("LZ_AEN", "RTM", D(2026, 9, 1), D(2026, 9, 25), today=TODAY, fetch=fake, cache_dir=tmp_path)
        assert fake.calls == [("2026-09-01", "2026-09-25")]
        assert cached_windows("LZ_AEN", "RTM", tmp_path) == [(D(2026, 9, 1), D(2026, 9, 25))]

    def test_a_day_saved_while_still_in_progress_is_fetched_again(self, tmp_path):
        # Saved at 8:15 pm on Sep 25, so the last 15 intervals didn't exist yet.
        _seed(tmp_path, "2026-09-01", "2026-09-25", drop_last=15, saved_at="2026-09-25 20:15")
        fake = FakeERCOT()
        result = load_range("LZ_AEN", "RTM", D(2026, 9, 1), D(2026, 9, 25), today=TODAY, fetch=fake, cache_dir=tmp_path)
        assert fake.calls == [("2026-09-01", "2026-09-25")]
        assert result.missing_days == ()
        assert len(result.frame) == 25 * 96


class TestToday:
    def test_today_is_fetched_live_and_never_stored(self, tmp_path):
        _seed(tmp_path, "2026-09-01", "2026-09-25")
        fake = FakeERCOT()
        result = load_range("LZ_AEN", "RTM", D(2026, 9, 20), TODAY, today=TODAY, fetch=fake, cache_dir=tmp_path)
        assert fake.calls == [("2026-09-26", "2026-09-26")]
        assert result.includes_today
        assert _days(result.frame)[-1] == TODAY
        assert len(result.frame) == 6 * 96 + 48  # six full days, then today up to noon
        assert all(hi < TODAY for _, hi in cached_windows("LZ_AEN", "RTM", tmp_path))

        again = FakeERCOT()
        load_range("LZ_AEN", "RTM", D(2026, 9, 20), TODAY, today=TODAY, fetch=again, cache_dir=tmp_path)
        assert again.calls == [("2026-09-26", "2026-09-26")]  # still live the second time

    def test_a_range_past_today_is_clipped_to_today(self, tmp_path):
        fake = FakeERCOT()
        result = load_range("LZ_AEN", "RTM", D(2026, 9, 24), D(2026, 12, 31), today=TODAY, fetch=fake, cache_dir=tmp_path)
        assert _days(result.frame)[-1] == TODAY

    def test_a_failure_fetching_today_keeps_the_finished_days(self, tmp_path):
        _seed(tmp_path, "2026-09-01", "2026-09-25")
        result = load_range(
            "LZ_AEN", "RTM", D(2026, 9, 20), TODAY, today=TODAY,
            fetch=FakeERCOT(fail_today=True), cache_dir=tmp_path,
        )
        assert not result.includes_today
        assert "no network" in result.today_error
        assert _days(result.frame)[-1] == D(2026, 9, 25)


class TestOffline:
    def test_offline_it_returns_what_is_saved_and_lists_the_gaps(self, tmp_path):
        _seed(tmp_path, "2026-09-01", "2026-09-20")
        result = load_range("LZ_AEN", "RTM", D(2026, 9, 15), TODAY, today=TODAY, cache_dir=tmp_path)
        assert _days(result.frame) == [D(2026, 9, d) for d in range(15, 21)]
        assert result.missing_days == tuple(D(2026, 9, d) for d in range(21, 26))
        assert not result.includes_today
        assert result.today_error == "offline"

    def test_offline_with_nothing_saved_gives_an_empty_frame(self, tmp_path):
        result = load_range("LZ_AEN", "RTM", D(2026, 9, 1), D(2026, 9, 5), today=TODAY, cache_dir=tmp_path)
        assert result.frame.empty
        assert len(result.missing_days) == 5

    def test_a_failed_month_download_is_reported_not_raised(self, tmp_path):
        def broken(request):
            raise ConnectionError("down")

        result = load_range("LZ_AEN", "RTM", D(2026, 9, 1), D(2026, 9, 5), today=TODAY, fetch=broken, cache_dir=tmp_path)
        assert result.frame.empty
        assert len(result.missing_days) == 5

    def test_start_after_end_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            load_range("LZ_AEN", "RTM", D(2026, 9, 10), D(2026, 9, 1), today=TODAY, cache_dir=tmp_path)


class TestErcotGaps:
    """ERCOT's history has occasional missing intervals that never get filled."""

    def test_a_gap_in_data_saved_after_the_day_is_accepted(self, tmp_path):
        _seed(
            tmp_path, "2026-08-01", "2026-08-31",
            saved_at="2026-09-10 09:00", drop_times=("2026-08-21 15:00",),
        )
        fake = FakeERCOT()
        result = load_range("LZ_AEN", "RTM", D(2026, 8, 1), D(2026, 8, 31), today=TODAY, fetch=fake, cache_dir=tmp_path)
        assert fake.calls == []  # not re-downloaded on every page load
        assert result.missing_days == ()
        assert len(result.frame) == 31 * 96 - 1

    def test_a_gap_in_a_fresh_download_is_not_fetched_again_next_time(self, tmp_path):
        class Gappy(FakeERCOT):
            def __call__(self, request):
                frame = super().__call__(request)
                return frame[frame["interval_start"] != pd.Timestamp("2026-08-21 15:00", tz=CENTRAL_TIME)]

        first = Gappy()
        now = pd.Timestamp("2026-09-26 12:00", tz=CENTRAL_TIME)
        result = load_range("LZ_AEN", "RTM", D(2026, 8, 1), D(2026, 8, 31), today=TODAY, fetch=first, cache_dir=tmp_path, now=now)
        assert first.calls == [("2026-08-01", "2026-08-31")]
        assert result.missing_days == ()

        second = Gappy()
        load_range("LZ_AEN", "RTM", D(2026, 8, 1), D(2026, 8, 31), today=TODAY, fetch=second, cache_dir=tmp_path, now=now)
        assert second.calls == []

    def test_a_day_with_no_data_at_all_is_still_reported(self, tmp_path):
        _seed(tmp_path, "2026-08-01", "2026-08-10", saved_at="2026-09-10 09:00")
        result = load_range("LZ_AEN", "RTM", D(2026, 8, 1), D(2026, 8, 12), today=TODAY, cache_dir=tmp_path)
        assert result.missing_days == (D(2026, 8, 11), D(2026, 8, 12))

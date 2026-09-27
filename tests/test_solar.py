from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from wattson.config import CENTRAL_TIME
from wattson.solar import ANNUAL_KWH_PER_KW, panel_output_kw, solar_value
from wattson.zones import HUBS, LOAD_ZONES, coordinates

AUSTIN = coordinates("LZ_AEN")


def _day(date: str, freq: str = "15min") -> pd.DatetimeIndex:
    start = pd.Timestamp(date).tz_localize(CENTRAL_TIME)
    end = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize(CENTRAL_TIME)  # local midnight
    return pd.date_range(start, end, freq=freq, inclusive="left")


def _prices(index: pd.DatetimeIndex, price) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interval_start": index,
            "price_usd_per_mwh": np.broadcast_to(np.asarray(price, dtype=float), len(index)).copy(),
        }
    )


class TestPanelOutput:
    def test_nothing_at_night_something_at_noon(self):
        day = _day("2026-06-21")
        out = pd.Series(panel_output_kw(day, 8.0, *AUSTIN), index=day)
        assert out.between_time("00:00", "05:00").sum() == 0
        assert out.between_time("22:00", "23:45").sum() == 0
        assert out.between_time("12:00", "14:00").min() > 3.0

    def test_output_is_proportional_to_panel_size(self):
        day = _day("2026-06-21")
        assert np.allclose(panel_output_kw(day, 16.0, *AUSTIN), 2 * panel_output_kw(day, 8.0, *AUSTIN))

    def test_a_year_makes_the_typical_texas_amount(self):
        year = pd.date_range("2026-01-01", "2027-01-01", freq="15min", tz=CENTRAL_TIME, inclusive="left")
        kwh = panel_output_kw(year, 1.0, *AUSTIN).sum() * 0.25
        assert kwh == pytest.approx(ANNUAL_KWH_PER_KW, rel=0.01)

    def test_summer_days_make_more_than_winter_days(self):
        summer = panel_output_kw(_day("2026-06-21"), 8.0, *AUSTIN).sum()
        winter = panel_output_kw(_day("2026-12-21"), 8.0, *AUSTIN).sum()
        assert summer > 1.2 * winter

    def test_the_sun_peaks_later_in_west_texas(self):
        day = _day("2026-06-21")
        houston = day[panel_output_kw(day, 8.0, *coordinates("LZ_HOUSTON")).argmax()]
        west = day[panel_output_kw(day, 8.0, *coordinates("LZ_WEST")).argmax()]
        assert west > houston

    def test_hourly_prices_give_the_same_daily_energy(self):
        quarter = panel_output_kw(_day("2026-06-21"), 8.0, *AUSTIN).sum() * 0.25
        hourly = panel_output_kw(_day("2026-06-21", "1h"), 8.0, *AUSTIN).sum() * 1.0
        assert hourly == pytest.approx(quarter, rel=0.03)

    def test_dst_days_work(self):
        spring = _day("2026-03-08")
        assert len(spring) == 92
        assert panel_output_kw(spring, 8.0, *AUSTIN).sum() > 0

    def test_every_area_has_coordinates_in_texas(self):
        for code in LOAD_ZONES + HUBS:
            lat, lon = coordinates(code)
            assert 25.5 < lat < 36.6 and -106.7 < lon < -93.5, code


class TestSolarValue:
    def test_earnings_are_output_times_price(self):
        day = _day("2026-06-21")
        result = solar_value(_prices(day, 50.0), 8.0, *AUSTIN)
        assert result.revenue_usd == pytest.approx(result.generated_kwh * 0.05)
        assert result.avg_price_sold_usd_per_mwh == pytest.approx(50.0)
        assert result.switched_off_hours == 0

    def test_panels_switch_off_below_zero(self):
        day = _day("2026-06-21")
        price = np.where((day.hour >= 11) & (day.hour < 13), -20.0, 40.0)
        result = solar_value(_prices(day, price), 8.0, *AUSTIN)
        assert result.switched_off_hours == pytest.approx(2.0)
        assert result.sold_kwh < result.generated_kwh
        assert result.revenue_usd > 0  # negative hours cost nothing

    def test_midday_cheap_prices_pull_the_solar_price_below_average(self):
        day = _day("2026-06-21")
        price = np.where((day.hour >= 9) & (day.hour < 16), 20.0, 60.0)
        result = solar_value(_prices(day, price), 8.0, *AUSTIN)
        assert result.avg_price_sold_usd_per_mwh < result.avg_price_usd_per_mwh

    def test_per_year_scales_the_window(self):
        day = _day("2026-06-21")
        result = solar_value(_prices(day, 50.0), 8.0, *AUSTIN)
        assert result.days == pytest.approx(1.0)
        assert result.per_year_usd == pytest.approx(result.revenue_usd * 365)

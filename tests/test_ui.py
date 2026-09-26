from __future__ import annotations

import re
from pathlib import Path

import datetime as dt

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from wattson.data.providers import PriceRequest, SyntheticProvider

APP = str(Path(__file__).resolve().parents[1] / "app.py")
COMPARE = "Compare all of Texas"


def _synthetic(price: float | None = None) -> pd.DataFrame:
    frame = SyntheticProvider(seed=11).fetch(
        PriceRequest(settlement_point="LZ_WEST", start_date="2026-06-01", end_date="2026-06-14")
    )
    return frame if price is None else frame.assign(price_usd_per_mwh=price)


@pytest.fixture
def offline_ui(monkeypatch):
    """Run the app against synthetic prices so tests never touch the network."""
    import wattson.ui as ui

    st.cache_data.clear()  # cached results must not leak between tests
    frame = _synthetic()
    monkeypatch.setattr(ui, "_prices_cached", lambda *a, **k: frame)
    monkeypatch.setattr(ui, "_load_credentials", lambda: False)
    return AppTest.from_file(APP, default_timeout=120)


@pytest.fixture
def losing_ui(monkeypatch):
    """Flat prices: nothing to gain, so the battery cannot make money."""
    import wattson.ui as ui

    st.cache_data.clear()
    frame = _synthetic(price=30.0)
    monkeypatch.setattr(ui, "_prices_cached", lambda *a, **k: frame)
    monkeypatch.setattr(ui, "_load_credentials", lambda: False)
    return AppTest.from_file(APP, default_timeout=120)


def _metric(app, label: str):
    return next(m for m in app.metric if m.label == label)


def _captions(app) -> str:
    return " ".join(c.value for c in app.caption)


def _expander(app, label: str):
    return next(e for e in app.expander if e.label == label)


class TestRenders:
    def test_it_runs_without_exceptions(self, offline_ui):
        offline_ui.run()
        assert not offline_ui.exception

    def test_it_asks_the_question_in_plain_words(self, offline_ui):
        offline_ui.run()
        assert offline_ui.title[0].value == "Wattson"
        assert "What could a Base battery earn" in offline_ui.header[0].value

    def test_it_explains_the_idea_before_any_numbers(self, offline_ui):
        offline_ui.run()
        intro = " ".join(m.value for m in offline_ui.markdown[:3])
        assert "every 15 minutes" in intro
        assert "buy low and sell high" in intro
        assert "real past prices" in intro

    def test_it_shows_the_headline_economics_in_plain_words(self, offline_ui):
        offline_ui.run()
        labels = [m.label for m in offline_ui.metric]
        for label in ("Earned buying low, selling high", "Battery wear", "Left after wear", "Per year at this pace"):
            assert label in labels

    def test_it_walks_through_the_story_in_order(self, offline_ui):
        offline_ui.run()
        assert [s.value for s in offline_ui.subheader] == [
            "A day in the life",
            "What it made each day",
            "How smart does the battery need to be?",
            "What tomorrow might look like",
        ]

    def test_prices_are_in_cents_per_kwh_on_the_main_page(self, offline_ui):
        offline_ui.run()
        price_line = next(c.value for c in offline_ui.caption if "ranged from" in c.value)
        assert re.search(r"ranged from -?[\d.]+¢ to [\d.]+¢ per kWh", price_line)
        assert "MWh" not in price_line

    def test_it_says_these_are_past_prices_not_a_promise(self, offline_ui):
        offline_ui.run()
        info = " ".join(i.value for i in offline_ui.info)
        assert "aren't a promise about the future" in info
        assert "Backup during outages" in info

    def test_it_warns_when_credentials_are_missing(self, offline_ui):
        offline_ui.run()
        assert any("credentials" in w.value for w in offline_ui.warning)


class TestPlainLanguage:
    JARGON = re.compile(
        r"arbitrage|kW-year|\bper kW\b|MWh|\bSoC\b|state of charge|dispatch|capture"
        r"|degradation|baseline|threshold|annuali[sz]ed|settlement|foresight",
        re.IGNORECASE,
    )

    @staticmethod
    def _main_page_text(app) -> list[str]:
        """Text a first-time visitor reads: everything outside the expanders."""
        kinds = ("success", "error", "info", "caption", "markdown", "subheader", "header")
        tucked_away = {
            el.value
            for exp in app.expander
            for kind in kinds
            for el in getattr(exp, kind)
        }
        return [
            el.value
            for kind in kinds
            for el in getattr(app.main, kind)
            if el.value not in tucked_away
        ]

    def test_the_main_page_avoids_industry_jargon(self, offline_ui):
        offline_ui.run()
        hits = [t for t in self._main_page_text(offline_ui) if self.JARGON.search(t)]
        assert hits == []

    def test_the_compare_view_avoids_industry_jargon(self, offline_ui):
        offline_ui.run()
        offline_ui.radio[0].set_value(COMPARE).run()
        hits = [t for t in self._main_page_text(offline_ui) if self.JARGON.search(t)]
        assert hits == []

    def test_there_is_a_glossary(self, offline_ui):
        offline_ui.run()
        glossary = _expander(offline_ui, "What do these words mean?")
        text = " ".join(m.value for m in glossary.markdown)
        for term in ("ERCOT", "Pricing area", "kWh and kW", "Battery wear"):
            assert term in text


class TestDayInTheLife:
    def test_it_opens_on_the_best_day(self, offline_ui):
        offline_ui.run()
        slider = offline_ui.select_slider[0]
        assert "the best day in this period" in _captions(offline_ui)
        assert slider.value in slider.options or slider.value is not None

    def test_it_says_what_the_battery_paid_and_got(self, offline_ui):
        offline_ui.run()
        text = _captions(offline_ui)
        assert "bought power at an average of" in text
        assert "per kWh and sold it at" in text

    def test_the_first_day_explains_why_the_battery_waited(self, offline_ui):
        import datetime as dt

        offline_ui.run()
        offline_ui.select_slider[0].set_value(dt.date(2026, 6, 1)).run()
        assert not offline_ui.exception
        assert "no past prices to learn from" in _captions(offline_ui)


class TestDayByDay:
    def test_it_explains_the_colors_and_the_takeaway(self, offline_ui):
        offline_ui.run()
        text = _captions(offline_ui)
        assert "Green days made money" in text
        assert re.search(r"best 3 of 14 days made \d+% of the total", text)

    def test_the_takeaway_matches_how_concentrated_earnings_are(self):
        import wattson.ui as ui

        captured = []

        class Result:
            def __init__(self, values):
                idx = pd.date_range("2026-06-01", periods=len(values) * 96, freq="15min", tz="America/Chicago")
                per_day = [v / 96 for v in values for _ in range(96)]
                self.ledger = pd.DataFrame({"interval_start": idx, "net_usd": per_day})

        orig = (ui.st.subheader, ui.st.plotly_chart, ui.st.caption)
        ui.st.subheader = lambda *a, **k: None
        ui.st.plotly_chart = lambda *a, **k: None
        ui.st.caption = lambda text, **k: captured.append(text)
        try:
            ui._day_by_day(Result([1.0] * 30))                 # perfectly even
            ui._day_by_day(Result([10, 10, 10] + [1.0] * 27))   # 3 days = 53%
            ui._day_by_day(Result([5, 5, 5] + [1.0] * 27))      # 3 days = 36%, 3.6x even
        finally:
            ui.st.subheader, ui.st.plotly_chart, ui.st.caption = orig
        assert "fairly even" in captured[0]
        assert "most of the money comes from a few days" in captured[1]
        assert "a few price spikes do a lot of the work" in captured[2]


class TestStrategies:
    def test_it_explains_each_way_of_running_the_battery(self, offline_ui):
        offline_ui.run()
        text = " ".join(m.value for m in offline_ui.markdown)
        for name in ("No battery", "Simple rule", "Forecast planner", "Perfect hindsight"):
            assert name in text

    def test_the_detailed_table_uses_plain_names(self, offline_ui):
        offline_ui.run()
        table = _expander(offline_ui, "Show the detailed numbers").dataframe[0].value
        assert list(table.index) == [
            "No battery",
            "Simple rule",
            "Forecast planner (what Wattson uses)",
            "Perfect hindsight (impossible in practice)",
        ]
        assert "Left after wear ($)" in table.columns
        assert table.loc["No battery", "Made ($)"] == 0.0

    def test_the_audit_trail_is_still_available(self, offline_ui):
        offline_ui.run()
        ledger = _expander(offline_ui, "Show the detailed numbers").dataframe[-1].value
        assert {"Time", "Price (¢ per kWh)", "Charging (kW)", "Selling (kW)", "Battery level (%)"} <= set(
            ledger.columns
        )


class TestInteraction:
    def test_two_cores_keep_twice_as_much_as_one(self, offline_ui):
        offline_ui.run()
        one = float(_metric(offline_ui, "Left after wear").value.replace("$", "").replace(",", ""))
        offline_ui.number_input[0].set_value(2).run()
        two = float(_metric(offline_ui, "Left after wear").value.replace("$", "").replace(",", ""))
        assert two == pytest.approx(2 * one, abs=0.02)

    def test_every_count_runs_without_error(self, offline_ui):
        from wattson.config import MAX_BASE_CORES

        offline_ui.run()
        for n in (1, 2, 3, MAX_BASE_CORES):
            offline_ui.number_input[0].set_value(n).run()
            assert not offline_ui.exception, n

    def test_the_compare_view_runs(self, offline_ui):
        offline_ui.run()
        offline_ui.radio[0].set_value(COMPARE).run()
        assert not offline_ui.exception
        assert offline_ui.subheader[0].value == "Which part of Texas pays best?"
        assert "comes out on top" in offline_ui.success[0].value
        assert "a year after wear, at this pace" in offline_ui.success[0].value

    def test_day_ahead_prices_run(self, offline_ui):
        offline_ui.run()
        offline_ui.radio[1].set_value("DAM").run()
        assert not offline_ui.exception

    def test_a_purchase_price_adds_a_payback_figure(self, offline_ui):
        offline_ui.run()
        offline_ui.checkbox[0].set_value(True).run()
        offline_ui.number_input[1].set_value(15_000.0).run()
        assert not offline_ui.exception
        assert "Pays for itself in" in [m.label for m in offline_ui.metric]


class TestDemoDefaults:
    def test_it_opens_on_austin_energy(self, offline_ui):
        from wattson.zones import label_for

        offline_ui.run()
        assert offline_ui.selectbox[0].value == "LZ_AEN"
        assert "Austin Energy" in label_for(offline_ui.selectbox[0].value)

    def test_it_asks_how_many_base_cores_and_starts_at_one(self, offline_ui):
        from wattson.config import MAX_BASE_CORES

        offline_ui.run()
        picker = offline_ui.number_input[0]
        assert picker.label == "How many Base Cores?"
        assert picker.value == 1
        assert (picker.min, picker.max) == (1, MAX_BASE_CORES)
        assert "39.2 kWh" in (picker.help or "")

    @pytest.mark.parametrize(
        ("count", "phrase"),
        [(1, "a Base Core in the"), (2, "two Base Cores in the"), (3, "three Base Cores in the")],
    )
    def test_the_count_reads_naturally_in_the_headline(self, offline_ui, count, phrase):
        offline_ui.run()
        offline_ui.number_input[0].set_value(count).run()
        assert phrase in offline_ui.success[0].value

    def test_it_explains_that_earnings_scale_and_flags_more_than_two(self, offline_ui):
        offline_ui.run()
        assert "earn 2 times as much" not in _captions(offline_ui)
        offline_ui.number_input[0].set_value(2).run()
        assert "2 Cores earn 2 times as much as one" in _captions(offline_ui)
        assert "shown for comparison" not in _captions(offline_ui)
        offline_ui.number_input[0].set_value(3).run()
        assert "Base installs one or two Cores per home" in _captions(offline_ui)

    def test_the_comparison_uses_the_count_too(self, offline_ui):
        offline_ui.run()
        offline_ui.number_input[0].set_value(2).run()
        offline_ui.radio[0].set_value(COMPARE).run()
        assert "two Base Cores there would have earned" in offline_ui.success[0].value

    def test_payback_is_not_claimed_without_a_price(self, offline_ui):
        offline_ui.run()
        assert "Pays for itself in" not in [m.label for m in offline_ui.metric]


class TestFailureHandling:
    def test_a_data_problem_is_reported_not_raised(self, offline_ui, monkeypatch):
        import wattson.ui as ui

        def boom(*args, **kwargs):
            raise ValueError("no data for that window")

        monkeypatch.setattr(ui, "_prices_cached", boom)
        offline_ui.run()
        assert not offline_ui.exception
        assert any("Couldn't load prices" in e.value for e in offline_ui.error)


class TestCachedPricesWithoutCredentials:
    """A seeded cache should be enough to run, even with no API access."""

    def test_it_serves_cached_prices_when_credentials_are_missing(self, monkeypatch, tmp_path):
        import wattson.data.cache as cache
        import wattson.ui as ui

        request = PriceRequest(
            settlement_point="LZ_WEST", start_date="2026-06-01", end_date="2026-06-14"
        )
        expected = SyntheticProvider(seed=3).fetch(request)
        monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path)
        cache.save(request, expected, tmp_path)

        got = ui._prices_cached.__wrapped__("LZ_WEST", "2026-06-01", "2026-06-14", "RTM", False)
        assert len(got) == len(expected)
        assert got["price_usd_per_mwh"].abs().sum() == pytest.approx(
            expected["price_usd_per_mwh"].abs().sum()
        )

    def test_it_explains_itself_when_there_is_neither_cache_nor_credentials(
        self, monkeypatch, tmp_path
    ):
        import wattson.data.cache as cache
        import wattson.ui as ui

        monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path)
        with pytest.raises(RuntimeError) as excinfo:
            ui._prices_cached.__wrapped__("LZ_WEST", "2026-06-01", "2026-06-14", "RTM", False)
        message = str(excinfo.value)
        assert "no cached Real-time (RTM) prices" in message
        assert "LZ_WEST" in message
        assert "credentials" in message


class TestLocationLabels:
    def test_the_area_picker_offers_readable_names(self, offline_ui):
        from wattson.zones import LOAD_ZONES, label_for

        offline_ui.run()
        options = offline_ui.selectbox[0].options
        assert set(options) == {label_for(code) for code in LOAD_ZONES}
        assert all(code in " ".join(options) for code in LOAD_ZONES)

    def test_the_picker_explains_that_areas_follow_the_utility(self, offline_ui):
        offline_ui.run()
        picker = offline_ui.selectbox[0]
        assert picker.label == "Where in Texas?"
        assert "which company delivers your electricity" in (picker.help or "")


class TestForecastSection:
    def test_it_describes_tomorrow_in_a_sentence(self, offline_ui):
        offline_ui.run()
        text = " ".join(m.value for m in offline_ui.markdown)
        assert "cheapest around" in text and "most expensive around" in text

    def test_it_says_how_much_to_trust_the_forecast(self, offline_ui):
        offline_ui.run()
        text = _captions(offline_ui)
        assert "How much to trust this" in text
        assert "than just assuming each day repeats the day before" in text


class TestDatePresets:
    TODAY = dt.date(2026, 9, 26)

    def test_it_offers_presets_and_custom_dates(self, offline_ui):
        offline_ui.run()
        picker = offline_ui.selectbox[1]
        assert picker.label == "Dates to replay"
        assert picker.options == [
            "Last 7 days", "Last month", "Last 2 months", "Last 3 months",
            "Last 6 months", "Last year", "Custom dates",
        ]

    def test_it_opens_on_the_last_month_up_to_today(self, offline_ui, monkeypatch):
        import wattson.ui as ui

        monkeypatch.setattr(ui, "_today", lambda: self.TODAY)
        offline_ui.run()
        assert offline_ui.selectbox[1].value == "Last month"
        assert "Aug 27, 2026 to today (Sep 26)" in _captions(offline_ui)

    @pytest.mark.parametrize(
        ("choice", "start"),
        [
            ("Last 7 days", dt.date(2026, 9, 20)),
            ("Last month", dt.date(2026, 8, 27)),
            ("Last 2 months", dt.date(2026, 7, 27)),
            ("Last 6 months", dt.date(2026, 3, 27)),
            ("Last year", dt.date(2025, 9, 27)),
        ],
    )
    def test_every_preset_ends_today(self, choice, start):
        import wattson.ui as ui

        assert ui._preset_window(choice, self.TODAY) == (start, self.TODAY)

    def test_custom_dates_can_run_up_to_today(self, offline_ui, monkeypatch):
        import wattson.ui as ui

        monkeypatch.setattr(ui, "_today", lambda: self.TODAY)
        offline_ui.run()
        assert not offline_ui.date_input
        offline_ui.selectbox[1].set_value("Custom dates").run()
        picker = offline_ui.date_input[0]
        assert picker.value == (dt.date(2026, 8, 28), self.TODAY)
        assert picker.max == self.TODAY
        assert not offline_ui.exception

    def test_today_is_texas_time(self, monkeypatch):
        import wattson.ui as ui

        # 11:30 pm in Texas is already the next day in UTC.
        late = pd.Timestamp("2026-09-26 23:30", tz="America/Chicago")
        monkeypatch.setattr(ui.pd.Timestamp, "now", classmethod(lambda cls, tz=None: late.tz_convert(tz)))
        assert ui._today() == dt.date(2026, 9, 26)


class TestDataNotes:
    @staticmethod
    def _with_attrs(**attrs):
        frame = _synthetic()
        frame.attrs.update(attrs)
        return frame

    def _run(self, monkeypatch, frame):
        import wattson.ui as ui

        st.cache_data.clear()
        monkeypatch.setattr(ui, "_prices_cached", lambda *a, **k: frame)
        monkeypatch.setattr(ui, "_load_credentials", lambda: False)
        app = AppTest.from_file(APP, default_timeout=120)
        app.run()
        return app

    def test_missing_days_are_named(self, monkeypatch):
        frame = self._with_attrs(missing_days=[dt.date(2026, 9, d) for d in (21, 22, 23, 25)])
        app = self._run(monkeypatch, frame)
        warning = " ".join(w.value for w in app.warning)
        assert "No prices for 4 days in this period (Sep 21–23, Sep 25)" in warning

    def test_offline_it_says_why_today_is_missing(self, monkeypatch):
        app = self._run(monkeypatch, self._with_attrs(today_error="offline"))
        assert "Today's prices need an internet connection, so this runs through Jun 14" in _captions(app)

    def test_when_today_loads_it_says_how_current_it_is(self, monkeypatch):
        app = self._run(monkeypatch, self._with_attrs(includes_today=True))
        assert "Today's prices are included up to 11:45 PM" in _captions(app)
        assert "need an internet connection" not in _captions(app)


class TestMarketLabels:
    def test_the_price_picker_uses_plain_names(self, offline_ui):
        offline_ui.run()
        picker = offline_ui.radio[1]
        assert picker.options == ["Real-time (RTM)", "Day-ahead (DAM)"]
        assert "ERCOT" in (picker.help or "")

    def test_the_explanation_follows_the_choice(self, offline_ui):
        offline_ui.run()
        assert "every 15 minutes as supply and demand shift" in _captions(offline_ui)
        offline_ui.radio[1].set_value("DAM").run()
        text = _captions(offline_ui)
        assert "the day before" in text
        assert "every 15 minutes as supply and demand shift" not in text


def _markdown_texts(app) -> list[str]:
    """Every piece of text Streamlit renders as markdown."""
    kinds = ("success", "error", "warning", "info", "caption", "markdown", "header", "subheader")
    return [el.value for kind in kinds for el in getattr(app, kind)]


def _unescaped_dollars(app) -> list[str]:
    return [t for t in _markdown_texts(app) if re.search(r"(?<!\\)\$", t)]


class TestDollarSigns:
    """Two bare $ signs make Streamlit render a LaTeX formula and eat the text."""

    def test_no_unescaped_dollars_on_the_main_view(self, offline_ui):
        offline_ui.run()
        assert _unescaped_dollars(offline_ui) == []

    def test_no_unescaped_dollars_when_the_battery_loses_money(self, losing_ui):
        losing_ui.run()
        assert losing_ui.error
        assert _unescaped_dollars(losing_ui) == []

    def test_no_unescaped_dollars_in_the_zone_comparison(self, offline_ui):
        offline_ui.run()
        offline_ui.radio[0].set_value(COMPARE).run()
        assert _unescaped_dollars(offline_ui) == []


class TestHeadline:
    def test_it_leads_with_what_the_battery_earned_then_wear(self, offline_ui):
        offline_ui.run()
        text = offline_ui.success[0].value
        assert "a Base Core in the Austin Energy area would have earned \\$" in text
        assert "of battery wear, that leaves \\$" in text
        assert "a year at this pace" in text
        assert "buying power when it was cheap and selling it back" in text

    def test_the_window_is_dated_by_its_last_day_not_the_midnight_after(self, offline_ui):
        offline_ui.run()
        text = offline_ui.success[0].value
        assert "Over these 14 days (Jun 1 – Jun 14, 2026)" in text

    def test_the_loss_message_shows_the_numbers(self, losing_ui):
        losing_ui.run()
        text = losing_ui.error[0].value
        assert "would not have covered its wear" in text
        assert "battery wear cost" in text
        assert "Prices didn't swing enough" in text


class TestLongPeriods:
    def test_a_span_across_years_names_both_years(self):
        import wattson.ui as ui

        idx = pd.DatetimeIndex(["2025-09-27 00:00", "2026-09-26 12:00"]).tz_localize("America/Chicago")
        assert ui._span(pd.DataFrame({"interval_start": idx})) == "Sep 27, 2025 – Sep 26, 2026"

    def test_a_span_within_a_year_names_it_once(self):
        import wattson.ui as ui

        idx = pd.DatetimeIndex(["2026-08-27 00:00", "2026-09-26 12:00"]).tz_localize("America/Chicago")
        assert ui._span(pd.DataFrame({"interval_start": idx})) == "Aug 27 – Sep 26, 2026"

    def test_a_full_year_does_not_repeat_itself(self, monkeypatch):
        import wattson.ui as ui

        st.cache_data.clear()
        year = SyntheticProvider(seed=11).fetch(
            PriceRequest(settlement_point="LZ_WEST", start_date="2025-09-27", end_date="2026-09-26")
        )
        monkeypatch.setattr(ui, "_prices_cached", lambda *a, **k: year)
        monkeypatch.setattr(ui, "_load_credentials", lambda: False)
        app = AppTest.from_file(APP, default_timeout=300)
        app.run()
        assert not app.exception
        text = (app.success or app.error)[0].value
        assert "Over these 365 days (Sep 27, 2025 – Sep 26, 2026)" in text
        assert "a year at this pace" not in text


def test_worker_processes_do_not_rerun_the_app(monkeypatch):
    """Parallel workers import the entry script as __mp_main__; it must stay inert."""
    import importlib.util

    import wattson.ui as ui

    calls = []
    monkeypatch.setattr(ui, "main", lambda: calls.append(1))
    spec = importlib.util.spec_from_file_location("__mp_main__", APP)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert calls == []



class TestBatteryWear:
    def test_the_explanation_says_what_actually_wears_out(self, offline_ui):
        offline_ui.run()
        help_text = _metric(offline_ui, "Battery wear").help or ""
        assert "lithium cells age" in help_text
        assert "permanently loses" in help_text
        assert "1.2¢ for every kWh" in help_text

    def test_the_glossary_separates_wear_from_other_losses(self, offline_ui):
        offline_ui.run()
        text = " ".join(m.value for m in _expander(offline_ui, "What do these words mean?").markdown)
        assert "lithium cells age" in text
        assert "not the same as the battery running down" in text

    def test_it_says_who_carries_the_cost(self, offline_ui):
        offline_ui.run()
        assert "battery wear is Base's cost, not the homeowner's" in _captions(offline_ui)

    def test_the_wear_cost_is_adjustable_and_starts_at_the_estimate(self, offline_ui):
        offline_ui.run()
        slider = offline_ui.slider[0]
        assert slider.label == "Battery wear cost (¢ per kWh in or out)"
        assert slider.value == pytest.approx(1.2)
        assert "instead of Wattson's" not in _captions(offline_ui)

    def test_changing_it_changes_the_result_and_says_so(self, offline_ui):
        def dollars(label):
            return float(_metric(offline_ui, label).value.replace("$", "").replace(",", ""))

        offline_ui.run()
        default_wear = dollars("Battery wear")
        offline_ui.slider[0].set_value(2.4).run()
        assert not offline_ui.exception
        assert "Using 2.4¢ per kWh for battery wear instead of Wattson's 1.2¢ estimate" in _captions(offline_ui)
        # More expensive wear means more wear cost, and a planner that trades less.
        assert dollars("Battery wear") != default_wear
        assert "2.4¢ for every kWh" in (_metric(offline_ui, "Battery wear").help or "")

    def test_zero_wear_leaves_everything_it_earned(self, offline_ui):
        def dollars(label):
            return float(_metric(offline_ui, label).value.replace("$", "").replace(",", ""))

        offline_ui.run()
        offline_ui.slider[0].set_value(0.0).run()
        assert dollars("Battery wear") == pytest.approx(0.0)
        assert dollars("Left after wear") == pytest.approx(dollars("Earned buying low, selling high"))

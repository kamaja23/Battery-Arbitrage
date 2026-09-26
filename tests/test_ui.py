from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from arb.data.providers import PriceRequest, SyntheticProvider
from arb.metrics import BASELINE_LABEL

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def offline_ui(monkeypatch):
    """Run the app against synthetic prices so tests never touch the network."""
    import arb.ui as ui

    frame = SyntheticProvider(seed=11).fetch(
        PriceRequest(
            settlement_point="LZ_WEST",
            start_date="2026-06-01",
            end_date="2026-06-14",
        )
    )
    monkeypatch.setattr(ui, "_prices_cached", lambda *a, **k: frame)
    monkeypatch.setattr(ui, "_load_credentials", lambda: False)
    return AppTest.from_file(APP, default_timeout=120)


class TestRenders:
    def test_it_runs_without_exceptions(self, offline_ui):
        offline_ui.run()
        assert not offline_ui.exception

    def test_it_answers_the_homeowner_question_up_front(self, offline_ui):
        offline_ui.run()
        assert "battery pay for itself" in offline_ui.title[0].value

    def test_it_shows_the_headline_economics(self, offline_ui):
        offline_ui.run()
        labels = [m.label for m in offline_ui.metric]
        assert "Net earned" in labels
        assert "Per kW / year" in labels
        assert "Payback" in labels

    def test_it_shows_the_dispatch_chart(self, offline_ui):
        offline_ui.run()
        assert any("Buying low" in s.value for s in offline_ui.subheader)

    def test_it_lists_the_baseline_alongside_the_strategies(self, offline_ui):
        offline_ui.run()
        table = next(df.value for df in offline_ui.dataframe)
        assert BASELINE_LABEL in table.index
        assert "threshold" in table.index

    def test_it_explains_why_the_baseline_is_zero(self, offline_ui):
        offline_ui.run()
        captions = " ".join(c.value for c in offline_ui.caption)
        assert "no spread to capture" in captions

    def test_it_flags_that_figures_are_annualized_not_forecast(self, offline_ui):
        offline_ui.run()
        captions = " ".join(c.value for c in offline_ui.caption)
        assert "not forecasts" in captions

    def test_it_warns_when_credentials_are_missing(self, offline_ui):
        offline_ui.run()
        assert any("credentials" in w.value for w in offline_ui.warning)

    def test_it_shows_a_ledger_for_auditability(self, offline_ui):
        offline_ui.run()
        ledger = offline_ui.dataframe[-1].value
        assert {"price_usd_per_mwh", "charge_kw", "soc_fraction"} <= set(
            ledger.columns
        )


class TestInteraction:
    def test_switching_base_models_changes_the_result(self, offline_ui):
        offline_ui.run()
        core = offline_ui.metric[0].value
        offline_ui.selectbox[1].set_value("base_core_dual").run()
        dual = offline_ui.metric[0].value
        assert core != dual

    def test_every_base_model_runs_without_error(self, offline_ui):
        from arb.config import PRESETS

        offline_ui.run()
        for key in PRESETS:
            offline_ui.selectbox[1].set_value(key).run()
            assert not offline_ui.exception, key

    def test_the_compare_view_runs(self, offline_ui):
        offline_ui.run()
        offline_ui.radio[0].set_value("Compare zones").run()
        assert not offline_ui.exception
        assert offline_ui.success


class TestDemoDefaults:
    def test_it_opens_on_austin_energy(self, offline_ui):
        from arb.zones import label_for

        offline_ui.run()
        assert offline_ui.selectbox[0].value == "LZ_AEN"
        assert "Austin Energy" in label_for(offline_ui.selectbox[0].value)

    def test_it_offers_only_base_batteries(self, offline_ui):
        offline_ui.run()
        options = offline_ui.selectbox[1].options
        assert options and all(o.startswith("Base ") for o in options)
        assert offline_ui.selectbox[1].value == "base_core"

    def test_payback_is_not_claimed_without_a_price(self, offline_ui):
        offline_ui.run()
        payback = next(m for m in offline_ui.metric if m.label == "Payback")
        assert payback.value == "n/a"


class TestFailureHandling:
    def test_a_data_problem_is_reported_not_raised(self, offline_ui, monkeypatch):
        import arb.ui as ui

        def boom(*args, **kwargs):
            raise ValueError("no data for that window")

        monkeypatch.setattr(ui, "_prices_cached", boom)
        offline_ui.run()
        assert not offline_ui.exception
        assert any("Could not load prices" in e.value for e in offline_ui.error)


class TestCachedPricesWithoutCredentials:
    """A seeded cache should be enough to run, even with no API access."""

    def test_it_serves_cached_prices_when_credentials_are_missing(
        self, monkeypatch, tmp_path
    ):
        import arb.data.cache as cache
        import arb.ui as ui

        request = PriceRequest(
            settlement_point="LZ_WEST",
            start_date="2026-06-01",
            end_date="2026-06-14",
        )
        expected = SyntheticProvider(seed=3).fetch(request)
        monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path)
        cache.save(request, expected, tmp_path)

        monkeypatch.setattr(ui, "load", cache.load)
        got = ui._prices_cached.__wrapped__(
            "LZ_WEST", "2026-06-01", "2026-06-14", "RTM", False
        )
        assert len(got) == len(expected)
        assert got["price_usd_per_mwh"].abs().sum() == pytest.approx(
            expected["price_usd_per_mwh"].abs().sum()
        )

    def test_it_explains_itself_when_there_is_neither_cache_nor_credentials(
        self, monkeypatch, tmp_path
    ):
        import arb.data.cache as cache
        import arb.ui as ui

        monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path)
        monkeypatch.setattr(ui, "load", cache.load)
        with pytest.raises(RuntimeError) as excinfo:
            ui._prices_cached.__wrapped__(
                "LZ_WEST", "2026-06-01", "2026-06-14", "RTM", False
            )
        message = str(excinfo.value)
        assert "no cached RTM prices" in message
        assert "LZ_WEST" in message
        assert "credentials" in message


class TestLocationLabels:
    def test_the_zone_selector_offers_readable_names(self, offline_ui):
        from arb.zones import LOAD_ZONES, label_for

        offline_ui.run()
        options = offline_ui.selectbox[0].options
        # The widget renders the readable name, not the bare code.
        assert set(options) == {label_for(code) for code in LOAD_ZONES}
        # ...but the raw code stays visible for traceability.
        assert all(code in " ".join(options) for code in LOAD_ZONES)

    def test_the_selector_warns_that_zones_follow_the_utility(self, offline_ui):
        offline_ui.run()
        help_text = offline_ui.selectbox[0].help or ""
        assert "service territory" in help_text


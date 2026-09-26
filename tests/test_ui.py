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
    def test_switching_to_the_commercial_preset_changes_the_result(self, offline_ui):
        offline_ui.run()
        residential = offline_ui.metric[0].value
        offline_ui.selectbox[1].set_value("commercial_1mw").run()
        commercial = offline_ui.metric[0].value
        assert residential != commercial

    def test_the_commercial_run_does_not_error(self, offline_ui):
        offline_ui.run()
        offline_ui.selectbox[1].set_value("commercial_1mw").run()
        assert not offline_ui.exception


class TestFailureHandling:
    def test_a_data_problem_is_reported_not_raised(self, offline_ui, monkeypatch):
        import arb.ui as ui

        def boom(*args, **kwargs):
            raise ValueError("no data for that window")

        monkeypatch.setattr(ui, "_prices_cached", boom)
        offline_ui.run()
        assert not offline_ui.exception
        assert any("Could not load prices" in e.value for e in offline_ui.error)

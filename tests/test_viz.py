from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from wattson.config import RESIDENTIAL_13KWH
from wattson.data.providers import PriceRequest, SyntheticProvider
from wattson.sim.engine import run_backtest
from wattson.strategies.threshold import ThresholdStrategy
from wattson.viz import cumulative_revenue_figure, daily_revenue_figure, dispatch_figure


@pytest.fixture
def result() -> "object":
    prices = SyntheticProvider(seed=2).fetch(
        PriceRequest(
            settlement_point="LZ_NORTH",
            start_date="2026-06-01",
            end_date="2026-06-10",
        )
    )
    return run_backtest(
        prices,
        RESIDENTIAL_13KWH,
        ThresholdStrategy(lookback_intervals=96, min_spread_usd_per_mwh=0.0),
    )


class TestDispatchFigure:
    def test_returns_three_traces(self, result):
        fig = dispatch_figure(result)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4

    def test_has_a_title(self, result):
        assert dispatch_figure(result).layout.title.text

    def test_accepts_a_custom_title(self, result):
        fig = dispatch_figure(result, title="hello")
        assert fig.layout.title.text == "hello"

    def test_price_trace_uses_ledger_prices(self, result):
        fig = dispatch_figure(result)
        assert list(fig.data[0].y) == list(result.ledger["price_usd_per_mwh"])

    def test_soc_trace_is_a_percentage(self, result):
        soc = fig_y(dispatch_figure(result), name="SoC %")
        assert max(soc) <= 100.0
        assert min(soc) >= 0.0

    def test_discharge_is_plotted_negative_for_stacking(self, result):
        discharge = fig_y(dispatch_figure(result), name="discharging kW")
        assert all(v <= 0 for v in discharge)

    def test_survives_a_one_day_window(self):
        prices = SyntheticProvider(seed=2).fetch(
            PriceRequest(
                settlement_point="LZ_NORTH", start_date="2026-06-01", end_date="2026-06-01"
            )
        )
        res = run_backtest(prices, RESIDENTIAL_13KWH, ThresholdStrategy(lookback_intervals=96))
        assert dispatch_figure(res) is not None


def fig_y(fig: go.Figure, name: str) -> list[float]:
    for trace in fig.data:
        if trace.name == name:
            return list(trace.y)
    raise AssertionError(f"no trace named {name!r}")


class TestRevenueFigures:
    def test_cumulative_ends_at_total_net(self, result):
        fig = cumulative_revenue_figure(result)
        assert fig_y(fig, "cumulative net")[-1] == pytest.approx(result.net_usd)

    def test_daily_has_one_bar_per_day(self, result):
        daily = result.ledger.set_index("interval_start")["net_usd"].resample("D").sum()
        fig = daily_revenue_figure(result)
        assert len(fig.data[0].y) == len(daily)


class TestZoneComparisonLabels:
    @staticmethod
    def _comparison(codes_and_values):
        return pd.DataFrame(
            {"usd_per_kw_year": [v for _, v in codes_and_values]},
            index=pd.Index([c for c, _ in codes_and_values], name="settlement_point"),
        )

    def test_bars_are_labelled_with_place_names_not_codes(self):
        from wattson.viz import zone_comparison_figure

        fig = zone_comparison_figure(
            self._comparison([("LZ_AEN", 4.4), ("LZ_WEST", 9.1), ("HB_PAN", -1.0)])
        )
        labels = list(fig.data[0].y)
        assert "Austin Energy (Austin / Travis Co.)" in labels
        assert "West Texas (Midland–Odessa)" in labels
        assert "Texas Panhandle hub (Amarillo)" in labels
        assert not any(label.startswith(("LZ_", "HB_")) for label in labels)

    def test_the_code_is_still_in_the_hover(self):
        from wattson.viz import zone_comparison_figure

        fig = zone_comparison_figure(self._comparison([("LZ_AEN", 4.4), ("LZ_WEST", 9.1)]))
        assert list(fig.data[0].customdata) == ["LZ_AEN", "LZ_WEST"]  # weakest first
        assert "%{customdata}" in fig.data[0].hovertemplate

    def test_an_unknown_code_is_shown_rather_than_dropped(self):
        from wattson.viz import zone_comparison_figure

        fig = zone_comparison_figure(self._comparison([("LZ_NEW", 1.0)]))
        assert list(fig.data[0].y) == ["LZ_NEW"]

"""Backtest engine: walks a price series and applies a strategy interval by
interval, recording a dispatch ledger.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wattson.config import BatteryConfig, EngineConfig
from wattson.data.providers import validate_price_frame
from wattson.sim.battery import BatteryState
from wattson.strategies.base import DecisionContext, Strategy

LEDGER_COLUMNS = (
    "interval_start",
    "interval_end",
    "price_usd_per_mwh",
    "charge_kw",
    "discharge_kw",
    "charged_kwh",
    "discharged_kwh",
    "soc_start_kwh",
    "soc_end_kwh",
    "soc_fraction",
    "revenue_usd",
    "cost_usd",
    "net_usd",
    "curtailed_charge_kw",
    "curtailed_discharge_kw",
)


@dataclass(slots=True)
class BacktestResult:
    """A dispatch ledger plus the inputs that produced it."""

    ledger: pd.DataFrame
    battery: BatteryConfig
    strategy_name: str
    market: str
    settlement_point: str

    @property
    def net_usd(self) -> float:
        return float(self.ledger["net_usd"].sum())


def prepare_price_series(
    prices: pd.DataFrame,
    market: str,
    settlement_point: str,
) -> pd.Series:
    """Validate and reduce a price frame to one sorted series."""
    validate_price_frame(prices)
    subset = prices[
        (prices["market"] == market)
        & (prices["settlement_point"] == settlement_point)
    ]
    if subset.empty:
        available = sorted(prices["settlement_point"].unique())
        raise ValueError(
            f"no {market} rows for {settlement_point!r}; available: {available}"
        )
    series = (
        subset.sort_values("interval_start")
        .set_index("interval_start")["price_usd_per_mwh"]
        .astype(float)
    )
    return series[~series.index.duplicated(keep="first")]


def _resolve_market(
    prices: pd.DataFrame, requested: str, settlement_point: str
) -> str:
    """Return the market actually available for ``settlement_point``.

    ``EngineConfig`` defaults to RTM, so a frame holding only DAM rows would
    otherwise look empty. Callers fetch one market at a time, so when the
    requested market is absent and exactly one is present, use that one.
    """
    at_point = set(
        prices.loc[prices["settlement_point"] == settlement_point, "market"]
    )
    if requested in at_point:
        return requested
    if len(at_point) == 1:
        return str(next(iter(at_point)))
    present = sorted(prices["market"].unique())
    raise ValueError(
        f"no {requested} rows for {settlement_point!r}; available: {present}"
    )


def run_backtest(
    prices: pd.DataFrame,
    battery: BatteryConfig,
    strategy: Strategy,
    engine: EngineConfig | None = None,
    settlement_point: str | None = None,
) -> BacktestResult:
    """Run ``strategy`` over ``prices`` for one battery and one settlement point."""
    engine = engine or EngineConfig()

    if settlement_point is None:
        points = sorted(prices["settlement_point"].unique())
        if len(points) != 1:
            raise ValueError(
                f"settlement_point is required when the frame holds {points}"
            )
        settlement_point = points[0]

    market = _resolve_market(prices, engine.market, settlement_point)
    series = prepare_price_series(prices, market, settlement_point)
    timestamps = pd.DatetimeIndex(series.index)
    values = series.to_numpy(dtype=float)

    durations = pd.Series(timestamps, index=series.index).diff()
    default_hours = 1.0 if market == "DAM" else 0.25
    modal = durations.mode()
    step = (
        pd.Timedelta(hours=default_hours)
        if modal.empty
        else pd.Timedelta(modal.iloc[0])
    )
    durations.iloc[0] = step
    hours = durations.dt.total_seconds().to_numpy() / 3600.0

    if engine.price_floor_usd_per_mwh is not None:
        values = values.clip(min=engine.price_floor_usd_per_mwh)

    strategy.prepare(series, timestamps)
    state = BatteryState(battery)
    state.reset()

    rows: list[dict] = []
    for i in range(len(values)):
        price = float(values[i])
        ctx = DecisionContext(
            index=i,
            interval_hours=float(hours[i]),
            price_usd_per_mwh=price,
            soc_kwh=state.soc_kwh,
            soc_fraction=state.soc_fraction,
            battery=battery,
            prices=series,
            timestamps=timestamps,
        )
        action = strategy.decide(ctx)
        step = state.step(
            charge_kw=action.charge_kw,
            discharge_kw=action.discharge_kw,
            interval_hours=float(hours[i]),
            price_usd_per_mwh=price,
        )
        rows.append(
            {
                "interval_start": timestamps[i],
                "interval_end": timestamps[i] + pd.Timedelta(hours=float(hours[i])),
                "price_usd_per_mwh": price,
                "charge_kw": step.charge_kw,
                "discharge_kw": step.discharge_kw,
                "charged_kwh": step.charged_kwh,
                "discharged_kwh": step.discharged_kwh,
                "soc_start_kwh": step.soc_start_kwh,
                "soc_end_kwh": step.soc_end_kwh,
                "soc_fraction": state.soc_fraction,
                "revenue_usd": step.revenue_usd,
                "cost_usd": step.cost_usd,
                "net_usd": step.net_usd,
                "curtailed_charge_kw": step.curtailed_charge_kw,
                "curtailed_discharge_kw": step.curtailed_discharge_kw,
            }
        )

    ledger = pd.DataFrame(rows, columns=list(LEDGER_COLUMNS))
    return BacktestResult(
        ledger=ledger,
        battery=battery,
        strategy_name=strategy.name,
        market=market,
        settlement_point=settlement_point,
    )

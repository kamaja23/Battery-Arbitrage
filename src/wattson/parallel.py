"""Run independent backtests on several CPU cores.

A year of prices is about 35,000 intervals, and the forecast planner solves an
optimization for every day of it, so comparing eight areas one after another
takes the better part of a minute. The work is mostly Python, which threads
cannot spread across cores, so this uses processes.

Workers start from a clean "forkserver" process rather than a fork of the
caller: Streamlit runs several threads, and forking a threaded process can
deadlock.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from wattson.config import BatteryConfig
from wattson.sim.engine import BacktestResult, prepare_price_series, run_backtest
from wattson.strategies.base import Strategy

# Below this many intervals (about two months of 15-minute prices) starting
# worker processes costs more than it saves.
PARALLEL_MIN_INTERVALS = 6_000


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 1)))


_CONTEXT = multiprocessing.get_context("forkserver")
# Preload only what workers need. The default would re-import the caller's
# __main__, which under Streamlit is Streamlit's own launcher.
_CONTEXT.set_forkserver_preload(["wattson.parallel"])


def pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=_CONTEXT)


def backtest_job(args: tuple[pd.DataFrame, BatteryConfig, Strategy]) -> BacktestResult:
    prices, battery, strategy = args
    # Strategies keep per-run state; every run gets its own copy.
    return run_backtest(prices, battery, copy.deepcopy(strategy))


def optimal_job(args: tuple[pd.DataFrame, BatteryConfig, str, str]):
    from wattson.strategies.perfect_foresight import solve_perfect_foresight

    prices, battery, market, point = args
    return solve_perfect_foresight(prepare_price_series(prices, market, point), battery)

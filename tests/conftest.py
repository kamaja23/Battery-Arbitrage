from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arb.data.providers import PriceRequest  # noqa: E402


@pytest.fixture
def rtm_request() -> PriceRequest:
    return PriceRequest(
        settlement_point="LZ_HOUSTON",
        start_date="2026-06-01",
        end_date="2026-06-03",
        market="RTM",
        location_type="Load Zone",
    )


@pytest.fixture
def dam_request() -> PriceRequest:
    return PriceRequest(
        settlement_point="LZ_HOUSTON",
        start_date="2026-06-01",
        end_date="2026-06-03",
        market="DAM",
        location_type="Load Zone",
    )

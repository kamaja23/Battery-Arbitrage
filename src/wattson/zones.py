"""Settlement point geography for ERCOT load zones and trading hubs.

The raw codes (``LZ_AEN``, ``HB_PAN``) are what the API accepts and what
engineers search for, so the code is always kept alongside a readable name.

Names and groupings are taken from ERCOT sources rather than inferred from the
abbreviations:

* Nodal Protocol Section 3.4.3(2)(b) names the four Non-Opt-In Entity load
  zones approved before the nodal market went live: Austin Energy, City Public
  Service, Rayburn County Electric Cooperative, and Lower Colorado River
  Authority.
* The remaining four are the competitive load zones: North, South, West, and
  Houston.

One caveat worth keeping in front of users: a load zone follows a utility's
service territory, not a postal address. Most of Austin sits in LZ_SOUTH, and
only Austin Energy's own customers are priced in LZ_AEN.
"""

from __future__ import annotations

from dataclasses import dataclass

COMPETITIVE = "competitive"
MUNICIPAL = "municipal utility"
NOIE = "cooperative / river authority"
HUB = "trading hub"


@dataclass(frozen=True, slots=True)
class SettlementPoint:
    """One price location: what it is called, and roughly where it is."""

    code: str
    name: str
    area: str
    kind: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.area})"

    @property
    def full_label(self) -> str:
        return f"{self.label} · {self.code}"


SETTLEMENT_POINTS: dict[str, SettlementPoint] = {
    sp.code: sp
    for sp in (
        # Competitive load zones.
        SettlementPoint("LZ_NORTH", "North Texas", "Dallas–Fort Worth", COMPETITIVE),
        SettlementPoint("LZ_SOUTH", "South Texas", "Corpus Christi / Laredo", COMPETITIVE),
        SettlementPoint("LZ_WEST", "West Texas", "Midland–Odessa", COMPETITIVE),
        SettlementPoint("LZ_HOUSTON", "Houston", "Houston metro", COMPETITIVE),
        # Non-Opt-In Entity load zones.
        SettlementPoint("LZ_AEN", "Austin Energy", "Austin / Travis Co.", MUNICIPAL),
        SettlementPoint("LZ_CPS", "CPS Energy", "San Antonio / Bexar Co.", MUNICIPAL),
        SettlementPoint("LZ_LCRA", "Lower Colorado River Authority", "Central Texas", NOIE),
        SettlementPoint("LZ_RAYBN", "Rayburn County Electric Co-op", "East Texas", NOIE),
        # Trading hubs.
        SettlementPoint("HB_PAN", "Texas Panhandle hub", "Amarillo", HUB),
        SettlementPoint("HB_WEST", "West Texas hub", "Midland–Odessa", HUB),
        SettlementPoint("HB_NORTH", "North Texas hub", "Dallas–Fort Worth", HUB),
        SettlementPoint("HB_SOUTH", "South Texas hub", "Corpus Christi", HUB),
        SettlementPoint("HB_HOUSTON", "Houston hub", "Houston metro", HUB),
    )
}

LOAD_ZONES: tuple[str, ...] = tuple(
    code for code, sp in SETTLEMENT_POINTS.items() if code.startswith("LZ_")
)
HUBS: tuple[str, ...] = tuple(
    code for code, sp in SETTLEMENT_POINTS.items() if code.startswith("HB_")
)


def label_for(code: str) -> str:
    """Readable label for a settlement point, falling back to the raw code."""
    point = SETTLEMENT_POINTS.get(code)
    return point.full_label if point else code


def name_for(code: str) -> str:
    """Place name without the raw code, for tight spaces like chart axes.

    ``"LZ_AEN"`` -> ``"Austin Energy (Austin / Travis Co.)"``. Falls back to
    the code for anything unknown.
    """
    point = SETTLEMENT_POINTS.get(code)
    return point.label if point else code


def short_name(code: str) -> str:
    """Just the name, for use inside a sentence: ``"LZ_AEN"`` -> ``"Austin Energy"``."""
    point = SETTLEMENT_POINTS.get(code)
    return point.name if point else code

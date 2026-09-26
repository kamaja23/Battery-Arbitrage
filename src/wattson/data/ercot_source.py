"""Live ERCOT provider.

The API returns positional columns whose arity depends on the report, and
``ercot`` 0.1.2's own formatter mis-handles the 7-column RTM shape, so this
module calls the endpoint directly and reshapes the response itself.

Verified against the live API:

* ``np6-905-cd/spp_node_zone_hub`` (RTM, 15-min) returns 7 columns --
  deliveryDate, hourEnding, interval, settlementPoint, settlementPointType,
  settlementPointPrice, repeatHourFlag. A load-zone filter still returns both
  ``LZ`` and ``LZEW`` rows, so rows must be narrowed to one type.
* ``np4-190-cd/dam_stlmnt_pnt_prices`` (DAM, hourly) returns 5 columns and
  needs no type narrowing.
* Interval counts per delivery day are 96 normal, 92 on the spring-forward
  transition, and 100 on the fall-back transition.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd

from arb.config import CENTRAL_TIME
from arb.data.providers import (
    PriceRequest,
    PriceSeriesProvider,
    validate_price_frame,
)

_BASE_URL = "https://api.ercot.com/api/public-reports"

_ENDPOINTS = {
    "DAM": "/np4-190-cd/dam_stlmnt_pnt_prices",
    "RTM": "/np6-905-cd/spp_node_zone_hub",
}

_RTM_SCHEMA = (
    "delivery_date",
    "hour_ending",
    "interval",
    "settlement_point",
    "settlement_point_type",
    "price_usd_per_mwh",
    "repeat_hour",
)

_DAM_SCHEMA = (
    "delivery_date",
    "hour_ending",
    "settlement_point",
    "price_usd_per_mwh",
    "repeat_hour",
)

_SETTLEMENT_POINT_TYPES = {"LZ": "Load Zone", "HU": "Hub"}

_ENV_KEYS = (
    "ERCOT_API_USERNAME",
    "ERCOT_API_PASSWORD",
    "ERCOT_API_SUBSCRIPTION_KEY",
)

_LABEL_ALIASES = {
    "primary key": "ERCOT_API_SUBSCRIPTION_KEY",
    "subscription key": "ERCOT_API_SUBSCRIPTION_KEY",
    "secondary key": "ERCOT_API_SECONDARY_KEY",
    "username": "ERCOT_API_USERNAME",
    "user name": "ERCOT_API_USERNAME",
    "email": "ERCOT_API_USERNAME",
    "login": "ERCOT_API_USERNAME",
    "password": "ERCOT_API_PASSWORD",
}


def _match_label(label: str) -> str | None:
    cleaned = label.strip().strip("*").strip().lower()
    if cleaned in _ENV_KEYS:
        return cleaned
    return _LABEL_ALIASES.get(cleaned)


def load_keys_file(path: str | os.PathLike[str], strict: bool = True) -> set[str]:
    """Load credentials from a text file into the environment.

    Understands ``KEY=value``, ``Label: value``, and the ERCOT API Explorer
    layout where a label sits on one line and its value on the next. Returns
    the set of recognised keys, and with ``strict`` raises if any of username,
    password, or subscription key is still missing.
    """
    lines = Path(path).read_text().splitlines()
    pending: str | None = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        match = re.match(r"^([^:=]+)\s*[:=]\s*(.*)$", line)
        if match:
            label, value = match.group(1), match.group(2).strip()
            env_key = _match_label(label)
            if env_key is None:
                continue
            if value:
                os.environ[env_key] = value
                pending = None
            else:
                pending = env_key
            continue

        env_key = pending if pending is not None else _match_label(line)
        if env_key:
            os.environ[env_key] = line
            pending = None
        elif len(line) >= 16 and re.fullmatch(r"[A-Za-z0-9_\-]+", line):
            os.environ.setdefault("ERCOT_API_SUBSCRIPTION_KEY", line)
            pending = None

    found = {k for k in _ENV_KEYS if os.environ.get(k)}
    if strict:
        missing = [k for k in _ENV_KEYS if k not in found]
        if missing:
            raise RuntimeError(
                f"credentials file {path} is missing: {', '.join(missing)}"
            )
    return found


def infer_location_type(settlement_point: str) -> str:
    """Derive the location type from an ERCOT settlement point name.

    ``LZ_`` names are load zones, ``HB_`` names are hubs, and ``HZ_`` names
    are resource nodes, which this project does not model.
    """
    name = settlement_point.strip().upper()
    if name.startswith("LZ_"):
        return "Load Zone"
    if name.startswith("HB_"):
        return "Hub"
    raise ValueError(
        f"{settlement_point!r} is not a supported settlement point; "
        "this tool only supports Load Zone and Hub names (LZ_ and HB_)"
    )


def _hour_ending_to_minutes(value: object) -> int:
    """Parse an hour-ending of ``'07:00'`` or ``24:00`` into minutes past midnight."""
    if isinstance(value, (int, float)):
        return int(value) * 60
    text = str(value).strip()
    match = re.match(r"^(\d{1,2}):(\d{2})$", text)
    if not match:
        raise ValueError(f"unparseable hour ending {value!r}")
    return int(match.group(1)) * 60 + int(match.group(2))


def _daily_interval_ends(
    frame: pd.DataFrame, market: str
) -> list[pd.Timestamp]:
    """Place each row on a real Central Time timeline using its ERCOT labels.

    Each row's offset from local midnight comes from its own
    ``hour_ending``/``interval`` label rather than from its position in the
    frame, so a day that is missing an interval does not shift the timestamps
    of every later row.

    Offsets are applied as absolute durations to a tz-aware midnight. Wall
    clock arithmetic cannot express the fall-back day's repeated 01:00 hour,
    whose second pass is an hour of real time later; advancing a tz-aware
    timestamp handles that, and lets 23, 24, and 25 hour days land correctly
    without special-casing either transition.

    ERCOT keeps hour endings numbered 1-24 on a transition day and signals the
    transition in the data itself: the fall-back hour is present twice, and
    the spring-forward hour is absent. Either way every row after the
    transition sits one hour off its naive label, in opposite directions, so
    the correction is derived per day from the labels present.
    """
    if market == "RTM":
        hours = pd.to_numeric(frame["hour_ending"], errors="coerce")
        minutes = (hours - 1) * 60 + pd.to_numeric(
            frame["interval"], errors="coerce"
        ) * 15
        per_hour_expected = 4
    else:
        hours = frame["hour_ending"].map(_hour_ending_to_minutes) / 60.0
        minutes = frame["hour_ending"].map(_hour_ending_to_minutes)
        per_hour_expected = 1

    if minutes.isna().any() or hours.isna().any():
        bad = int((minutes.isna() | hours.isna()).sum())
        raise ValueError(f"could not resolve an interval label on {bad} rows")

    repeat = frame["repeat_hour"].astype(bool)
    ends: list[pd.Timestamp] = []
    day_labels = frame["delivery_date"].astype(str)

    for day in day_labels.unique():
        on_day = day_labels == day
        day_hours = hours[on_day]
        n_rows = int(on_day.sum())
        counts = day_hours.value_counts()
        repeated = sorted(h for h, n in counts.items() if n > per_hour_expected)
        present = set(counts.index)

        if repeated:
            boundary = repeated[0]
            shift = 60 * (repeat[on_day] | (day_hours > boundary))
        else:
            missing = [
                h for h in range(int(min(present)), int(max(present)) + 1)
                if h not in present
            ]
            # Only a full-length day that is short by exactly one hour ending
            # 2 or 3 can be a spring-forward day. A truncated or gappy day is a
            # data gap, and treating it as a transition would shift every hour
            # after the gap by an hour and mislabel the rest of the day.
            short_by_one = n_rows == 24 * per_hour_expected - per_hour_expected
            is_dst_hour = len(missing) == 1 and missing[0] in (2, 3)
            shift = -60 * (day_hours > missing[0]) if short_by_one and is_dst_hour else 0

        midnight = pd.Timestamp(str(day)[:10]).tz_localize(CENTRAL_TIME)
        offsets = (minutes[on_day] + shift).to_numpy()
        ends.extend(midnight + pd.to_timedelta(offsets, unit="m"))
    return ends


class ErcotLiveProvider(PriceSeriesProvider):
    """Fetches settlement point prices from the official ERCOT public API."""

    def __init__(self, page_size: int = 100_000) -> None:
        self._client = None
        self._page_size = page_size

    def _get_client(self):
        if self._client is None:
            import ercot as er

            missing = [k for k in _ENV_KEYS if not os.environ.get(k)]
            if missing:
                raise RuntimeError(f"missing ERCOT credentials: {', '.join(missing)}")
            er.configure(
                username=os.environ["ERCOT_API_USERNAME"],
                password=os.environ["ERCOT_API_PASSWORD"],
                subscription_key=os.environ["ERCOT_API_SUBSCRIPTION_KEY"],
                save=True,
            )
            self._client = er.client()
        return self._client

    def fetch(self, request: PriceRequest) -> pd.DataFrame:
        client = self._get_client()
        params = client._date_params(request.start_date, request.end_date)
        client._add_sp_filter(params, [request.settlement_point])
        endpoint = "dam_spp" if request.market == "DAM" else "rtm_spp"
        raw = self._request_with_backoff(client, endpoint, params)
        return self.from_raw(raw, request)

    @staticmethod
    def _request_with_backoff(
        client, endpoint: str, params: dict, attempts: int = 6
    ) -> pd.DataFrame:
        """Retry throttled and transiently failing ERCOT requests.

        The public API rate limits aggressively and answers 429 once a caller
        exceeds its budget, so a single backoff loop keeps long fetches from
        failing outright. A 400 is a caller error and is raised immediately.
        """
        import requests as _requests

        delay = 2.0
        for attempt in range(attempts):
            try:
                return client.request(endpoint, params)
            except _requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 429 or 500 <= status < 600:
                    if attempt == attempts - 1:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise
            except _requests.RequestException:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError(f"ERCOT request for {endpoint} never completed")

    @staticmethod
    def from_raw(raw: pd.DataFrame, request: PriceRequest) -> pd.DataFrame:
        """Reshape a raw ERCOT response into the normalized price frame."""
        if raw is None or raw.empty:
            raise ValueError(
                f"ERCOT returned no rows for {request.settlement_point} "
                f"{request.market} {request.start_date}..{request.end_date}"
            )

        schema = _RTM_SCHEMA if request.market == "RTM" else _DAM_SCHEMA
        if len(raw.columns) != len(schema):
            raise ValueError(
                f"expected {len(schema)} positional columns for {request.market}, "
                f"got {len(raw.columns)}"
            )

        frame = raw.copy()
        frame.columns = list(schema)
        frame["price_usd_per_mwh"] = pd.to_numeric(
            frame["price_usd_per_mwh"], errors="coerce"
        )
        frame["repeat_hour"] = frame["repeat_hour"].fillna(False).astype(bool)
        frame = frame.dropna(subset=["price_usd_per_mwh"])
        location_type = infer_location_type(request.settlement_point)

        if request.market == "RTM":
            wanted = "LZ" if location_type == "Load Zone" else "HU"
            frame = frame[frame["settlement_point_type"] == wanted]
            if frame.empty:
                raise ValueError(
                    f"no {wanted} rows for {request.settlement_point}; "
                    "the API returned only other settlement point types"
                )

        # Collapse echoed rows before the DST analysis below. An echo adds an
        # extra interval to an hour, which would make that hour look repeated
        # and trigger a false fall-back shift that runs into the next day.
        # repeat_hour stays in the key so a genuine DST repeated hour, which
        # carries a different repeat_hour, is preserved.
        key_cols = (
            ["delivery_date", "hour_ending", "interval", "repeat_hour"]
            if request.market == "RTM"
            else ["delivery_date", "hour_ending", "repeat_hour"]
        )
        echoed = frame.duplicated(subset=key_cols, keep=False)
        if echoed.any():
            conflicting = frame.loc[echoed].groupby(key_cols)[
                "price_usd_per_mwh"
            ].nunique()
            if (conflicting > 1).any():
                bad = conflicting[conflicting > 1].index.tolist()[:3]
                raise ValueError(
                    f"conflicting prices for repeated intervals at {bad}"
                )
            frame = frame.drop_duplicates(subset=key_cols, keep="first")

        step = pd.Timedelta(hours=1 if request.market == "DAM" else 0.25)
        if request.market == "RTM":
            frame = frame.sort_values(
                ["delivery_date", "hour_ending", "interval", "repeat_hour"],
                kind="stable",
            )
        else:
            frame = frame.sort_values(
                ["delivery_date", "hour_ending", "repeat_hour"], kind="stable"
            )
        ends = _daily_interval_ends(frame, request.market)

        out = pd.DataFrame(
            {
                "interval_start": [t - step for t in ends],
                "interval_end": ends,
                "market": request.market,
                "settlement_point": frame["settlement_point"].astype(str).str.strip(),
                "location_type": location_type,
                "price_usd_per_mwh": frame["price_usd_per_mwh"].to_numpy(dtype=float),
            }
        )
        out = out.sort_values("interval_start").reset_index(drop=True)
        if out["interval_start"].duplicated().any():
            # ERCOT occasionally echoes a row for a settlement point. Echoes
            # agree on price and are safe to collapse; genuine disagreements
            # are a real data conflict and must not be resolved silently.
            repeated = out["interval_start"].duplicated(keep=False)
            conflicting = out.loc[repeated].groupby("interval_start")[
                "price_usd_per_mwh"
            ].nunique()
            if (conflicting > 1).any():
                bad = conflicting[conflicting > 1].index.tolist()[:5]
                raise ValueError(
                    f"conflicting prices for duplicate intervals at {bad}"
                )
            out = out[~out["interval_start"].duplicated(keep="first")]
            out = out.reset_index(drop=True)
        return validate_price_frame(out)

    @staticmethod
    def normalize(raw: pd.DataFrame, request: PriceRequest) -> pd.DataFrame:
        """Reshape an ``ercot``-formatted long frame into the contract."""
        if raw is None or raw.empty:
            raise ValueError(
                f"ERCOT returned no rows for {request.settlement_point} "
                f"{request.market} {request.start_date}..{request.end_date}"
            )

        step = pd.Timedelta(hours=1.0 if request.market == "DAM" else 0.25)
        ends = pd.to_datetime(raw["interval_ending"])
        if ends.dt.tz is None:
            ends = ends.dt.tz_localize(CENTRAL_TIME)
        else:
            ends = ends.dt.tz_convert(CENTRAL_TIME)
        location_type = infer_location_type(request.settlement_point)

        out = pd.DataFrame(
            {
                "interval_start": ends - step,
                "interval_end": ends,
                "market": request.market,
                "settlement_point": raw["settlement_point"].astype(str).str.strip(),
                "location_type": location_type,
                "price_usd_per_mwh": pd.to_numeric(raw["price"], errors="coerce"),
            }
        )
        out = out.dropna(subset=["price_usd_per_mwh"])
        out = out.sort_values("interval_start").reset_index(drop=True)
        return validate_price_frame(out)

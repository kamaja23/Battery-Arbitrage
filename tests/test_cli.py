from __future__ import annotations

import pytest

import arb.cli as cli
import arb.data.cache as cache
from arb.config import PRESETS
from arb.data.providers import PriceRequest, SyntheticProvider

MISSING_KEYS = "/nonexistent/ERCOT API Keys.txt"


@pytest.fixture
def seeded_cache(tmp_path, monkeypatch):
    """A cache holding synthetic prices, and no credentials anywhere."""
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path)
    for zone in ("LZ_AEN", "LZ_WEST"):
        for start, end in (("2026-06-01", "2026-06-30"), ("2026-07-01", "2026-07-31")):
            request = PriceRequest(settlement_point=zone, start_date=start, end_date=end)
            cache.save(request, SyntheticProvider(seed=len(zone) + int(start[6])).fetch(request), tmp_path)
    return tmp_path


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


class TestOffline:
    def test_backtest_runs_from_cache_without_credentials(self, seeded_cache, capsys):
        code, out, _ = _run(
            capsys, "backtest", "--zone", "LZ_AEN", "--start", "2026-06-01",
            "--end", "2026-06-30", "--keys", MISSING_KEYS,
        )
        assert code == 0
        for name in PRESETS:
            assert name in out
        assert "forecast" in out
        assert "Austin Energy" in out

    def test_one_battery_can_be_chosen(self, seeded_cache, capsys):
        code, out, _ = _run(
            capsys, "backtest", "--zone", "LZ_AEN", "--start", "2026-06-01",
            "--end", "2026-06-30", "--keys", MISSING_KEYS, "--battery", "base_core",
        )
        assert code == 0
        assert "base_core " in out and "base_core_dual" not in out

    def test_a_cache_miss_without_credentials_is_a_clear_error(self, seeded_cache, capsys):
        code, _, err = _run(
            capsys, "backtest", "--zone", "LZ_AEN", "--start", "2025-01-01",
            "--end", "2025-01-31", "--keys", MISSING_KEYS,
        )
        assert code == 1
        assert "no cached RTM prices" in err and "credentials" in err

    def test_compare_zones_skips_what_is_not_cached(self, seeded_cache, capsys):
        code, out, _ = _run(
            capsys, "compare-zones", "--start", "2026-06-01", "--end", "2026-06-30",
            "--keys", MISSING_KEYS,
        )
        assert code == 0
        assert "(2 locations)" in out
        assert "skipped" in out

    def test_stability_ranks_zones_across_months(self, seeded_cache, capsys):
        code, out, _ = _run(
            capsys, "stability", "--months", "2026-06", "2026-07", "--keys", MISSING_KEYS,
        )
        assert code == 0
        assert "2026-06" in out and "2026-07" in out
        assert "Rank agreement" in out

    def test_stability_needs_two_months(self, seeded_cache, capsys):
        code, out, _ = _run(
            capsys, "stability", "--months", "2026-06", "--keys", MISSING_KEYS,
        )
        assert code == 1
        assert "at least two months" in out


class TestDefaults:
    def test_the_default_battery_is_base_core(self):
        args = cli.build_parser().parse_args(["compare-zones"])
        assert args.battery == "base_core"

    def test_the_default_zone_is_austin_energy(self):
        args = cli.build_parser().parse_args(["backtest"])
        assert args.zone == "LZ_AEN"
        assert args.battery == "all"

    def test_generic_batteries_are_not_offered(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["compare-zones", "--battery", "commercial_1mw"])


class TestShortcuts:
    @pytest.mark.parametrize(
        ("entry", "command"),
        [
            (cli.backtest_main, "backtest"),
            (cli.fetch_main, "fetch"),
            (cli.verify_main, "verify-data"),
        ],
    )
    def test_each_shortcut_supplies_its_subcommand(self, entry, command, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli, "main", lambda argv: seen.setdefault("argv", argv) and 0)
        monkeypatch.setattr(cli.sys, "argv", ["prog", "--zone", "LZ_WEST"])
        entry()
        assert seen["argv"] == [command, "--zone", "LZ_WEST"]

    def test_the_installed_names_point_at_the_shortcuts(self):
        import tomllib
        from pathlib import Path

        project = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
        )
        scripts = project["project"]["scripts"]
        assert scripts == {
            "arb": "arb.cli:main",
            "arb-backtest": "arb.cli:backtest_main",
            "arb-verify-data": "arb.cli:verify_main",
            "arb-fetch": "arb.cli:fetch_main",
        }

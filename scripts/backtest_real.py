"""Thin wrapper so the CLI runs from a source checkout.

Installed as ``wattson-backtest`` / ``wattson-verify-data`` / ``wattson-fetch``; from here,
pass the subcommand through, e.g. ``python scripts/backtest_real.py backtest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattson.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

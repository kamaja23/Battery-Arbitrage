"""Thin wrapper so the data verification CLI runs from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattson.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["verify-data", *sys.argv[1:]]))

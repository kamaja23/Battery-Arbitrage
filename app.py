"""Streamlit entry point.

Run from a source checkout with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from arb.ui import main  # noqa: E402

main()

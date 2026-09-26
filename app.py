"""Streamlit entry point.

Run from a source checkout with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wattson.ui import main  # noqa: E402

# Streamlit runs this file as __main__. Worker processes started for parallel
# backtests re-import it as __mp_main__, and must not run the whole app again.
if __name__ == "__main__":
    main()

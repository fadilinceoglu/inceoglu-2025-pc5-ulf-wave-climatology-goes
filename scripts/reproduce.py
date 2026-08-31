#!/usr/bin/env python3
"""Repository-checkout entry point for every reproduction stage."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pc5_climatology.cli import main, main_for_stage  # noqa: E402

__all__ = ["main", "main_for_stage"]

if __name__ == "__main__":
    raise SystemExit(main())

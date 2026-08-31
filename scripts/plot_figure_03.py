#!/usr/bin/env python3
"""Regenerate paper Figure 3 from local checkpoints."""

from reproduce import main_for_stage

if __name__ == "__main__":
    raise SystemExit(main_for_stage("figure", figure_number=3))

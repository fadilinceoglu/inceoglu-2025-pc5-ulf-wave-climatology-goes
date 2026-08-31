#!/usr/bin/env python3
"""Run only GOES acquisition and preparation."""

from reproduce import main_for_stage

if __name__ == "__main__":
    raise SystemExit(main_for_stage("acquire"))

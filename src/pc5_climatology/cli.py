"""Command-line interface shared by every public script."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from .config import (
    DEFAULT_MONTE_CARLO_SAMPLES,
    DEFAULT_RANDOM_SEED,
    RepositoryPaths,
)
from .pipeline import PipelineOptions, ReproductionPipeline

STAGES = ("all", "acquire", "detect", "catalog", "figures", "figure", "status")

_FIXED_STAGE_DESCRIPTIONS = {
    "all": "Run or resume the complete GOES Pc5 reproduction chain.",
    "acquire": "Acquire and prepare the GOES magnetometer inputs.",
    "detect": "Detect Pc5 events in the prepared GOES inputs.",
    "catalog": "Build the up-to-three significant-peak event catalogs.",
    "figures": "Regenerate all four paper figures and the correlation table.",
    "figure": "Regenerate one paper figure.",
    "status": "Report the available reproduction-stage artifacts as JSON.",
}


def build_parser(
    *,
    fixed_stage: str | None = None,
    fixed_figure_number: int | None = None,
) -> argparse.ArgumentParser:
    """Build either the general CLI or one fixed-stage wrapper CLI."""

    if fixed_stage is not None and fixed_stage not in STAGES:
        raise ValueError(f"Unknown fixed stage: {fixed_stage}")
    if fixed_figure_number is not None and (
        fixed_stage != "figure" or fixed_figure_number not in range(1, 5)
    ):
        raise ValueError("A fixed figure number from 1 through 4 requires stage 'figure'")

    description = (
        "Reproduce the 2025 GOES Pc5 ULF-wave climatology study."
        if fixed_stage is None
        else _FIXED_STAGE_DESCRIPTIONS[fixed_stage]
    )
    parser = argparse.ArgumentParser(description=description)
    if fixed_stage is None:
        parser.add_argument("stage", nargs="?", default="all", choices=STAGES)
        parser.add_argument("figure_number", nargs="?", type=int, choices=range(1, 5))
    else:
        parser.set_defaults(stage=fixed_stage)
        if fixed_stage == "figure" and fixed_figure_number is None:
            parser.add_argument("figure_number", type=int, choices=range(1, 5))
        else:
            parser.set_defaults(figure_number=fixed_figure_number)

    parser.set_defaults(
        checkpoint_dir=None,
        omni_file=None,
        figures_dir=None,
        tables_dir=None,
        workers=max(1, min(4, os.cpu_count() or 1)),
        monte_carlo_samples=DEFAULT_MONTE_CARLO_SAMPLES,
        random_seed=DEFAULT_RANDOM_SEED,
        full=False,
        force=False,
    )
    parser.add_argument("--root", type=Path, default=_default_root())
    parser.add_argument("--checkpoint-dir", type=Path)

    general = fixed_stage is None
    figure_uses_omni = fixed_stage == "figure" and (
        fixed_figure_number is None or fixed_figure_number in (3, 4)
    )
    if general or fixed_stage in ("all", "acquire", "figures", "status") or figure_uses_omni:
        parser.add_argument("--omni-file", type=Path)
    # Every stage can invalidate the complete-output manifest. Accept both
    # output roots even when that stage does not render an output itself.
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--tables-dir", type=Path)
    if general or fixed_stage in ("all", "detect"):
        parser.add_argument("--workers", type=_positive_int)
    if general or fixed_stage in ("all", "detect", "status"):
        parser.add_argument("--monte-carlo-samples", type=_positive_int)
        parser.add_argument(
            "--random-seed",
            type=_nonnegative_int,
            help=f"Repository Monte Carlo seed (default: {DEFAULT_RANDOM_SEED}).",
        )
    if general or fixed_stage == "all":
        parser.add_argument(
            "--full",
            action="store_true",
            help="Allow the resource-intensive full-interval upstream stages.",
        )
    if general or fixed_stage in ("all", "acquire", "detect", "catalog"):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Recompute the selected stage instead of resuming it.",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the general command-line interface."""

    args = build_parser().parse_args(argv)
    return _run_with_cli_boundary(args)


def main_for_stage(
    stage: str,
    argv: Sequence[str] | None = None,
    *,
    figure_number: int | None = None,
) -> int:
    """Run a wrapper whose stage identity is fixed by its script name."""

    args = build_parser(
        fixed_stage=stage,
        fixed_figure_number=figure_number,
    ).parse_args(argv)
    return _run_with_cli_boundary(args)


def _run_with_cli_boundary(args: argparse.Namespace) -> int:
    """Render expected operational failures without masking code defects."""

    try:
        return _run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.stage != "status":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
    if args.stage == "figure" and args.figure_number is None:
        raise SystemExit("The figure stage requires a number from 1 through 4.")
    if args.stage != "figure" and args.figure_number is not None:
        raise SystemExit("A figure number is accepted only after the 'figure' stage.")
    if args.full and args.stage != "all":
        raise SystemExit("--full is meaningful only with the 'all' stage.")
    if args.stage == "all" and args.force and not args.full:
        raise SystemExit("For safety, forcing the complete upstream chain requires --full --force.")

    paths = RepositoryPaths.from_root(
        args.root,
        checkpoint_dir=args.checkpoint_dir,
        omni_file=args.omni_file,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
    )
    pipeline = ReproductionPipeline(
        paths,
        PipelineOptions(
            workers=args.workers,
            monte_carlo_samples=args.monte_carlo_samples,
            random_seed=args.random_seed,
            force=args.force,
        ),
    )

    if args.stage == "status":
        print(json.dumps(pipeline.status(), indent=2, sort_keys=True))
    elif args.stage == "all":
        pipeline.all(full=args.full)
    elif args.stage == "acquire":
        pipeline.acquire()
    elif args.stage == "detect":
        pipeline.detect()
    elif args.stage == "catalog":
        pipeline.catalog()
    elif args.stage == "figures":
        pipeline.figures()
    elif args.stage == "figure":
        pipeline.figure(args.figure_number)
    return 0


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed

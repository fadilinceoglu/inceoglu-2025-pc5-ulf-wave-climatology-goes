"""Build Pc5 catalogs from up to the three strongest significant peaks.

The component checkpoints contain one table per processed day.  For every
one-hour window, this stage retains the ranked significant peaks 1, 2, and 3,
maps them to the shared ``date``, ``t1``, ``t2``, ``freq``, and ``power``
columns, and concatenates them in rank order.  Input acquisition determines the
available date range; this stage does not apply another date filter.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Final

import joblib
import pandas as pd

from .config import RETAINED_PEAKS_PER_WINDOW
from .io import atomic_write_json, sha256_file

_COMPONENTS: Final = {
    "radial": {
        "input": "Frequency_Power_radial_new_1h.pkl",
        "output": "radial_powers_freq_mlt_date.pkl",
        "frequency_prefix": "peak_freq_rad_",
        "amplitude_prefix": "peak_pow_rad_",
    },
    "azimuthal": {
        "input": "Frequency_Power_azimuthal_new_1h.pkl",
        "output": "az_powers_freq_mlt_date.pkl",
        "frequency_prefix": "peak_freq_az_",
        "amplitude_prefix": "peak_pow_az_",
    },
    "parallel": {
        "input": "Frequency_Power_parallel_new_1h.pkl",
        "output": "par_powers_freq_mlt_date.pkl",
        "frequency_prefix": "peak_freq_par_",
        "amplitude_prefix": "peak_pow_par_",
    },
}

_COMPLETION_FILE = "catalog_complete.json"
_INCOMPLETE_FILE = "catalog_incomplete.json"


def _flatten_detection_batches(payload: object, *, source: Path) -> pd.DataFrame:
    """Flatten daily detection tables in their stored order.

    Empty list objects denote failed daily work and are skipped. DataFrame
    objects are concatenated in their stored order.
    """

    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"Expected a list-like detection checkpoint in {source}")

    nonempty_lists = [item for item in payload if isinstance(item, list) and item]
    if nonempty_lists:
        raise TypeError(f"Nonempty list objects are not valid detection placeholders in {source}")
    tables = [item for item in payload if not isinstance(item, list)]
    if not tables:
        return pd.DataFrame()
    if not all(isinstance(item, pd.DataFrame) for item in tables):
        unexpected = sorted(
            {type(item).__name__ for item in tables if not isinstance(item, pd.DataFrame)}
        )
        raise TypeError(f"Unsupported detection objects in {source}: {unexpected}")

    return pd.concat(tables).reset_index(drop=True)


def _top_three_catalog(
    detections: pd.DataFrame,
    *,
    frequency_prefix: str,
    amplitude_prefix: str,
) -> pd.DataFrame:
    """Retain ranks 1--3 of the highest-amplitude significant peaks."""

    output_columns = ["date", "t1", "t2", "freq", "power"]
    if detections.empty and not len(detections.columns):
        return pd.DataFrame(columns=output_columns)

    base_columns = ["date", "t1", "t2"]
    missing_base = [column for column in base_columns if column not in detections]
    if missing_base:
        raise KeyError(f"Detection checkpoint is missing columns: {missing_base}")

    ranked: list[pd.DataFrame] = []
    absent_rank_seen = False
    for rank in range(1, RETAINED_PEAKS_PER_WINDOW + 1):
        frequency_column = f"{frequency_prefix}{rank}"
        amplitude_column = f"{amplitude_prefix}{rank}"
        present = [column in detections.columns for column in (frequency_column, amplitude_column)]
        if not any(present):
            absent_rank_seen = True
            continue
        if absent_rank_seen:
            raise ValueError("Detection checkpoint contains a non-contiguous peak rank")
        if not all(present):
            missing = [
                column
                for column in (frequency_column, amplitude_column)
                if column not in detections.columns
            ]
            raise KeyError(f"Detection checkpoint is missing columns: {missing}")

        selection = detections.loc[:, [*base_columns, frequency_column, amplitude_column]].copy()
        selection.columns = output_columns
        ranked.append(selection.reset_index(drop=True))

    if not ranked:
        if len(detections):
            raise KeyError("Detection checkpoint has event rows but no peak rank columns")
        return pd.DataFrame(columns=output_columns)
    return pd.concat(ranked).dropna().reset_index(drop=True)


def _atomic_joblib_dump(value: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            joblib.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _component_paths(checkpoint_dir: Path, field: str) -> dict[str, Path]:
    return {
        component: checkpoint_dir / str(specification[field])
        for component, specification in _COMPONENTS.items()
    }


def _hashes_by_name(
    paths: dict[str, Path],
    *,
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, str]:
    return {path.name: hash_file(path) for path in paths.values()}


def catalog_outputs_are_usable(
    checkpoint_dir: Path,
    *,
    hash_file: Callable[[Path], str] = sha256_file,
) -> bool:
    """Accept a complete catalog run or an unmarked supplied catalog set."""

    checkpoint_dir = Path(checkpoint_dir)
    outputs = _component_paths(checkpoint_dir, "output")
    if not all(path.is_file() for path in outputs.values()):
        return False
    if (checkpoint_dir / _INCOMPLETE_FILE).exists():
        return False

    completion_path = checkpoint_dir / _COMPLETION_FILE
    if not completion_path.exists():
        return True
    try:
        import json

        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        if not (
            payload.get("schema_version") == 1
            and payload.get("status") == "complete"
            and payload.get("retained_peaks_per_window") == RETAINED_PEAKS_PER_WINDOW
            and payload.get("output_sha256") == _hashes_by_name(outputs, hash_file=hash_file)
        ):
            return False

        inputs = _component_paths(checkpoint_dir, "input")
        recorded_inputs = payload.get("input_sha256")
        expected_input_names = {path.name for path in inputs.values()}
        if not isinstance(recorded_inputs, dict) or set(recorded_inputs) != expected_input_names:
            return False
        for path in inputs.values():
            if path.is_file() and recorded_inputs.get(path.name) != hash_file(path):
                return False

    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def build_event_catalogs(
    checkpoint_dir: Path,
    *,
    force: bool = False,
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, Path]:
    """Build three component catalogs containing up to ranked peaks 1--3.

    Parameters
    ----------
    checkpoint_dir:
        Directory containing the component significant-peak checkpoints.  The
        generated catalogs use the documented filenames consumed by
        Figures 1--4.  Stored rows are processed without a second date filter.
    force:
        Replace existing catalog checkpoints when true.  By default an existing
        component catalog is left untouched.
    hash_file:
        Stable file-digest function used to bind completion metadata.

    Returns
    -------
    dict
        Mapping from ``radial``, ``azimuthal``, and ``parallel`` to their output
        checkpoint paths.
    """

    checkpoint_dir = Path(checkpoint_dir)
    outputs = _component_paths(checkpoint_dir, "output")
    if not force and catalog_outputs_are_usable(checkpoint_dir, hash_file=hash_file):
        return outputs

    inputs = _component_paths(checkpoint_dir, "input")
    for component, source in inputs.items():
        if not source.is_file():
            raise FileNotFoundError(f"Missing {component} detection checkpoint: {source}")

    completion_path = checkpoint_dir / _COMPLETION_FILE
    incomplete_path = checkpoint_dir / _INCOMPLETE_FILE
    input_hashes = _hashes_by_name(inputs, hash_file=hash_file)
    atomic_write_json(
        {
            "schema_version": 1,
            "status": "incomplete",
            "retained_peaks_per_window": RETAINED_PEAKS_PER_WINDOW,
            "input_sha256": input_hashes,
        },
        incomplete_path,
    )
    completion_path.unlink(missing_ok=True)

    staging_dir = Path(tempfile.mkdtemp(prefix=".catalog-output-", dir=checkpoint_dir))
    try:
        for component, specification in _COMPONENTS.items():
            source = inputs[component]
            detections = _flatten_detection_batches(joblib.load(source), source=source)
            catalog = _top_three_catalog(
                detections,
                frequency_prefix=str(specification["frequency_prefix"]),
                amplitude_prefix=str(specification["amplitude_prefix"]),
            )
            _atomic_joblib_dump(catalog, staging_dir / outputs[component].name)

        for component, destination in outputs.items():
            os.replace(staging_dir / destination.name, destination)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    atomic_write_json(
        {
            "schema_version": 1,
            "status": "complete",
            "retained_peaks_per_window": RETAINED_PEAKS_PER_WINDOW,
            "input_sha256": input_hashes,
            "output_sha256": _hashes_by_name(outputs, hash_file=hash_file),
        },
        completion_path,
    )
    incomplete_path.unlink(missing_ok=True)

    return outputs

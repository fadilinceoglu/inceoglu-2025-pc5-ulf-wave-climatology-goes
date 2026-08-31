"""Pc5 wave extraction from the prepared daily GOES checkpoints.

This module implements the paper's scientific thresholds and variable-width
peak-table schema with three operational guarantees:

* every product is associated with a daily DataFrame by satellite and UTC date;
* every local checkpoint is atomically replaced; and
* completed per-day temporary results are reused unless ``force=True``.

The paper's Monte Carlo seed was not recorded.  The repository default seed
derives an independent deterministic stream for every daily task; changing the
seed can change detections close to the significance threshold.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .config import (
    CADENCE_SECONDS,
    CANDIDATE_AMPLITUDE_MIN_NT,
    COMPONENTS,
    DEFAULT_MONTE_CARLO_SAMPLES,
    DEFAULT_RANDOM_SEED,
    DETECTION_ALGORITHM_VERSION,
    DETECTION_HIGHPASS_CUTOFF_MINUTES,
    DETECTION_HIGHPASS_ORDER,
    DETECTION_STEP_MINUTES,
    DETECTION_WINDOW_MINUTES,
    FFT_PADDED_SAMPLES,
    MAXIMUM_CLEAN_ITERATIONS,
    MAXIMUM_FALSE_ALARM_PROBABILITY,
    PC5_FREQUENCY_MAX_HZ,
    PC5_FREQUENCY_MIN_HZ,
    PEAK_FIT_HALF_WINDOW_MAX,
    PEAK_FIT_HALF_WINDOW_MIN,
    SATELLITE_NUMBERS,
    STUDY_END,
    STUDY_START,
)
from .io import sha256_file

LOGGER = logging.getLogger(__name__)


SAMPLE_INTERVAL_SECONDS = float(CADENCE_SECONDS)
HIGH_PASS_PERIOD_MINUTES = float(DETECTION_HIGHPASS_CUTOFF_MINUTES)
HIGH_PASS_ORDER = DETECTION_HIGHPASS_ORDER
WINDOW_HOURS = DETECTION_WINDOW_MINUTES / 60.0
WINDOW_STEP_HOURS = DETECTION_STEP_MINUTES / 60.0
ZERO_PADDED_SAMPLES = FFT_PADDED_SAMPLES
PEAK_HEIGHT_NT = CANDIDATE_AMPLITUDE_MIN_NT
MAX_CLEAN_ITERATIONS = MAXIMUM_CLEAN_ITERATIONS
PC5_MIN_HZ = PC5_FREQUENCY_MIN_HZ
PC5_MAX_HZ = PC5_FREQUENCY_MAX_HZ
SIGNIFICANCE_LEVEL = MAXIMUM_FALSE_ALARM_PROBABILITY
FIT_MAX_HALF_WINDOW = PEAK_FIT_HALF_WINDOW_MAX
FIT_MIN_HALF_WINDOW = PEAK_FIT_HALF_WINDOW_MIN

_FILENAME_PATTERN = re.compile(r"dn_magn-l2-hires_(g\d{1,2})_d(\d{8})_v(\d+[_-]\d+[_-]\d+)\.nc")
_COMPONENTS = COMPONENTS
_OUTPUT_FILES = {
    "radial": "Frequency_Power_radial_new_1h.pkl",
    "azimuthal": "Frequency_Power_azimuthal_new_1h.pkl",
    "parallel": "Frequency_Power_parallel_new_1h.pkl",
}


@dataclass(frozen=True)
class ProductRecord:
    """Parsed identity of one entry in ``processed_data.pkl``."""

    global_index: int
    url: str
    satellite: int
    observation_date: date
    version: tuple[int, int, int]

    @property
    def key(self) -> tuple[int, date]:
        """Return the satellite-date key used for product version selection."""

        return self.satellite, self.observation_date


@dataclass(frozen=True)
class DetectionTask:
    """Serializable input for one daily multiprocessing task."""

    index: int
    frame: pd.DataFrame
    url: str
    satellite_date: str
    temp_dir: Path
    monte_carlo_samples: int
    day_seed: int | None
    prepared_checkpoint_sha256: str


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of one computed or reused daily task."""

    index: int
    reused: bool
    error: str | None = None


def _joblib() -> Any:
    """Import Joblib only when a checkpoint is read or written."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - installation error
        raise RuntimeError("Pc5 detection requires the 'joblib' package") from exc
    return joblib


def _atomic_joblib_dump(value: Any, path: Path, *, compress: int = 0) -> None:
    """Atomically replace a Joblib checkpoint in the target directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _joblib().dump(value, temporary, compress=compress)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(value: Any, path: Path) -> None:
    """Atomically replace a deterministic UTF-8 JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_list(path: Path) -> list[Any]:
    """Load a required list checkpoint."""

    if not path.exists():
        raise FileNotFoundError(f"Required prepared GOES checkpoint is missing: {path}")
    value = _joblib().load(path)
    if not isinstance(value, list):
        raise TypeError(f"Expected a list checkpoint at {path}")
    return value


def _parse_product_url(global_index: int, url: str) -> ProductRecord:
    """Parse satellite, UTC date, and semantic version from a NOAA filename."""

    filename = Path(urlparse(url).path).name
    match = _FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Unsupported GOES high-resolution filename: {filename}")
    satellite_token, date_token, version_token = match.groups()
    return ProductRecord(
        global_index=global_index,
        url=url,
        satellite=int(satellite_token[1:]),
        observation_date=datetime.strptime(date_token, "%Y%m%d").date(),
        version=tuple(int(part) for part in re.split(r"[_-]", version_token)),
    )


def _frame_observation_date(frame: pd.DataFrame, satellite: int) -> date:
    """Validate one daily frame and return its sole UTC observation date."""

    required = {
        "time",
        "mlt",
        "b_radial",
        "b_azimuthal",
        "b_parallel",
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"df_{satellite:02d}.pkl contains a non-DataFrame or empty entry")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"df_{satellite:02d}.pkl entry is missing columns: {sorted(missing)}")

    timestamps = pd.to_datetime(frame["time"], errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError(f"df_{satellite:02d}.pkl entry has no valid UTC time")
    dates = {timestamp.date() for timestamp in timestamps}
    if len(dates) != 1:
        raise ValueError(f"df_{satellite:02d}.pkl entry spans multiple UTC dates")

    if "satellite" in frame.columns:
        values = {
            str(value).lower().replace("goes", "").lstrip("0") or "0"
            for value in frame["satellite"].dropna().unique()
        }
        if values and values != {str(satellite)}:
            raise ValueError(f"df_{satellite:02d}.pkl contains satellite values {sorted(values)}")
    return dates.pop()


def _validated_latest_products(
    checkpoint_dir: Path,
) -> tuple[list[str], list[tuple[ProductRecord, pd.DataFrame]]]:
    """Validate product/frame identity and retain the latest product version.

    Both sides are grouped by satellite and actual UTC date, and equal
    multiplicity is required. Newly prepared frames carry their source URL for
    direct validation. Unmarked checkpoint frames with multiple versions for
    the same identity are paired in their stored append order.
    """

    processed_values = _load_list(checkpoint_dir / "processed_data.pkl")
    if not all(isinstance(value, str) for value in processed_values):
        raise TypeError("processed_data.pkl must contain only URL strings")
    processed_urls = [str(value) for value in processed_values]
    duplicate_urls = sorted(url for url, count in Counter(processed_urls).items() if count > 1)
    if duplicate_urls:
        raise ValueError(
            "processed_data.pkl repeats exact product URLs: " + ", ".join(duplicate_urls[:10])
        )
    records = [_parse_product_url(index, url) for index, url in enumerate(processed_urls)]

    records_by_key: dict[tuple[int, date], list[ProductRecord]] = {}
    for record in records:
        if record.satellite not in SATELLITE_NUMBERS:
            raise ValueError(f"Unsupported GOES satellite in {record.url}")
        if not STUDY_START <= record.observation_date <= STUDY_END:
            raise ValueError(
                "processed_data.pkl contains a product date outside the inclusive "
                f"paper interval {STUDY_START.isoformat()}..{STUDY_END.isoformat()}: "
                f"{record.observation_date.isoformat()}"
            )
        records_by_key.setdefault(record.key, []).append(record)

    frames_by_key: dict[tuple[int, date], list[pd.DataFrame]] = {}
    for satellite in SATELLITE_NUMBERS:
        satellite_path = checkpoint_dir / f"df_{satellite:02d}.pkl"
        frames = _load_list(satellite_path) if satellite_path.exists() else []
        for frame in frames:
            observation_date = _frame_observation_date(frame, satellite)
            if not STUDY_START <= observation_date <= STUDY_END:
                raise ValueError(
                    f"df_{satellite:02d}.pkl contains a product date outside the "
                    "inclusive paper interval "
                    f"{STUDY_START.isoformat()}..{STUDY_END.isoformat()}: "
                    f"{observation_date.isoformat()}"
                )
            frames_by_key.setdefault((satellite, observation_date), []).append(frame)

    url_keys = set(records_by_key)
    frame_keys = set(frames_by_key)
    if url_keys != frame_keys:
        missing_frames = sorted(url_keys - frame_keys)
        missing_urls = sorted(frame_keys - url_keys)
        raise ValueError(
            "Prepared GOES URL/DataFrame identity mismatch; "
            f"URL keys without frames={missing_frames[:10]}, "
            f"frame keys without URLs={missing_urls[:10]}"
        )

    frame_for_record: dict[int, pd.DataFrame] = {}
    for key, key_records in records_by_key.items():
        key_frames = frames_by_key[key]
        if len(key_records) != len(key_frames):
            raise ValueError(
                "Prepared GOES URL/DataFrame multiplicity mismatch for "
                f"GOES {key[0]:02d} on {key[1]}: "
                f"{len(key_records)} URLs != {len(key_frames)} frames"
            )
        for record, frame in zip(key_records, key_frames):
            frame_url = frame.attrs.get("source_url")
            if frame_url is not None and frame_url != record.url:
                raise ValueError(
                    "Prepared frame source URL does not match its product record "
                    f"for GOES {key[0]:02d} on {key[1]}"
                )
            frame_for_record[record.global_index] = frame

    best_by_key: dict[tuple[int, date], ProductRecord] = {}
    key_order: list[tuple[int, date]] = []
    for record in records:
        if record.key not in best_by_key:
            key_order.append(record.key)
            best_by_key[record.key] = record
        elif record.version > best_by_key[record.key].version:
            best_by_key[record.key] = record

    selected = [
        (best_by_key[key], frame_for_record[best_by_key[key].global_index]) for key in key_order
    ]
    # Stable scientific identity controls task numbering, per-day seeded RNG
    # streams, and
    # reusable temporary filenames.
    selected.sort(
        key=lambda pair: (
            pair[0].satellite,
            pair[0].observation_date,
            pair[0].version,
            pair[0].url,
        )
    )
    return processed_urls, selected


def hanning_peak_model(x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
    """Squared-sinc model used to subtract a Hanning-window spectral peak."""

    return amplitude * np.sinc((x - center) / width) ** 2


def high_pass_filter(data: pd.Series) -> pd.Series:
    """Apply the fifth-order 30-minute high-pass filter.

    The daily series is sampled once per minute. NaNs are replaced by zero for
    ``filtfilt`` and restored afterward, which can influence samples adjacent
    to gaps.
    """

    if not isinstance(data, pd.Series):
        raise TypeError("data must be a pandas.Series")
    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:  # pragma: no cover - installation error
        raise RuntimeError("Pc5 detection requires the 'scipy' package") from exc

    sampling_frequency = 1.0 / SAMPLE_INTERVAL_SECONDS
    nyquist = 0.5 * sampling_frequency
    normalized_cutoff = (1.0 / (HIGH_PASS_PERIOD_MINUTES * SAMPLE_INTERVAL_SECONDS)) / nyquist
    numerator, denominator = butter(HIGH_PASS_ORDER, normalized_cutoff, btype="high")
    missing = data.isna()
    filtered = filtfilt(numerator, denominator, data.fillna(0).to_numpy())
    result = pd.Series(filtered, index=data.index)
    result.loc[missing] = np.nan
    return result


def clean_ulf(
    data_frame: pd.DataFrame,
    component: str,
    *,
    monte_carlo_samples: int,
    rng: Any,
) -> pd.DataFrame:
    """Extract significant Pc5 peaks from one component/window.

    This is the paper's enhanced-CLEAN calculation.  A one-hour
    Hanning-windowed signal is centered in a 1,440-sample zero-padded array.
    ``4 / N`` scales FFT magnitude to the nT amplitude called ``power``
    downstream.  Each iteration removes the fitted peak from the magnitude
    spectrum while retaining the original phase array; the complex residual is
    not transformed back and re-analysed.  Peaks above 1 nT are removed even
    when they fail p <= 0.05 or lie outside 1.6--6.7 mHz, which affects later
    iterations and must not be optimized away.
    """

    if component not in _COMPONENTS:
        raise ValueError(f"Unknown magnetic component: {component}")
    if monte_carlo_samples <= 0:
        raise ValueError("monte_carlo_samples must be positive")
    try:
        from scipy.optimize import curve_fit
        from scipy.signal import find_peaks
    except ImportError as exc:  # pragma: no cover - installation error
        raise RuntimeError("Pc5 detection requires the 'scipy' package") from exc

    column = f"b_{component}_hp"
    empty = pd.DataFrame(columns=["freq", "amp", "phase"])
    if column not in data_frame or data_frame[column].isna().all():
        return empty

    signal = data_frame[column]
    signal_size = len(signal)
    if signal_size > ZERO_PADDED_SAMPLES:
        raise ValueError(
            f"Signal has {signal_size} samples, exceeding zero-pad length {ZERO_PADDED_SAMPLES}"
        )
    time_step_seconds = float(np.mean(np.diff(data_frame["time_hours"]))) * 3600.0
    hanning_window = np.hanning(signal_size)
    pad_left = (ZERO_PADDED_SAMPLES - signal_size) // 2
    pad_right = ZERO_PADDED_SAMPLES - signal_size - pad_left

    windowed_signal = hanning_window * signal.to_numpy()
    padded_signal = np.pad(windowed_signal, (pad_left, pad_right), mode="constant")
    frequencies = np.fft.rfftfreq(ZERO_PADDED_SAMPLES, time_step_seconds)
    complex_spectrum = np.fft.rfft(padded_signal)
    amplitudes = 4.0 / signal_size * np.abs(complex_spectrum)
    phases = np.angle(complex_spectrum)

    noise_samples = rng.normal(
        np.nanmean(signal), np.nanstd(signal), (monte_carlo_samples, signal_size)
    )
    windowed_noise = noise_samples * hanning_window
    padded_noise = np.pad(
        windowed_noise,
        ((0, 0), (pad_left, pad_right)),
        mode="constant",
    )
    noise_amplitudes = 4.0 / signal_size * np.abs(np.fft.rfft(padded_noise, axis=1)).transpose()

    rows: list[dict[str, float]] = []
    for _ in range(MAX_CLEAN_ITERATIONS):
        peaks, _ = find_peaks(amplitudes, height=PEAK_HEIGHT_NT)
        if peaks.size == 0:
            break
        peak_index = int(peaks[np.argmax(amplitudes[peaks])])
        peak_amplitude = float(amplitudes[peak_index])
        peak_frequency = float(frequencies[peak_index])
        peak_phase = float(phases[peak_index])

        fitted_parameters: np.ndarray | None = None
        for half_window in range(FIT_MAX_HALF_WINDOW, FIT_MIN_HALF_WINDOW - 1, -1):
            left = max(0, peak_index - half_window)
            right = min(len(frequencies), peak_index + half_window + 1)
            x_values = frequencies[left:right]
            y_values = amplitudes[left:right]
            if len(x_values) < 3:
                continue
            initial_width = 2.0 / (signal_size * time_step_seconds)
            try:
                fitted_parameters, _ = curve_fit(
                    hanning_peak_model,
                    x_values,
                    y_values,
                    p0=[float(np.max(y_values)), frequencies[peak_index], initial_width],
                    bounds=(
                        [0.0, frequencies[0], 0.0],
                        [np.inf, frequencies[-1], np.inf],
                    ),
                )
                peak_frequency = float(fitted_parameters[1])
                break
            except (RuntimeError, ValueError):
                fitted_parameters = None

        significance = float(
            np.sum(noise_amplitudes[peak_index] > peak_amplitude) / monte_carlo_samples
        )
        if significance <= SIGNIFICANCE_LEVEL and PC5_MIN_HZ <= peak_frequency <= PC5_MAX_HZ:
            rows.append(
                {
                    "freq": peak_frequency,
                    "amp": peak_amplitude,
                    "phase": peak_phase,
                }
            )

        if fitted_parameters is not None:
            fitted_peak = hanning_peak_model(frequencies, *fitted_parameters)
        else:
            center = frequencies[peak_index]
            width = 2.0 / (signal_size * time_step_seconds)
            fitted_peak = peak_amplitude * np.sinc((frequencies - center) / width) ** 2
        amplitudes = np.maximum(amplitudes - fitted_peak, 0.0)

    return pd.DataFrame(rows, columns=["freq", "amp", "phase"])


def _peak_row(
    peaks: pd.DataFrame,
    abbreviation: str,
    *,
    url: str,
    satellite_date: str,
    date_value: pd.Timestamp,
    window: pd.DataFrame,
) -> pd.DataFrame | None:
    """Convert one component's sorted peaks to the checkpoint row schema."""

    if peaks.empty:
        return None
    ordered = peaks.sort_values("amp", ascending=False).reset_index(drop=True)
    columns: list[str] = []
    for number in range(1, len(ordered) + 1):
        columns.extend(
            [
                f"peak_freq_{abbreviation}_{number}",
                f"peak_pow_{abbreviation}_{number}",
                f"peak_phase_{abbreviation}_{number}",
            ]
        )
    row = pd.DataFrame(ordered.to_numpy().reshape(1, -1), columns=columns)
    row["data_file"] = url
    row["sat_date"] = satellite_date
    row["date"] = date_value
    # Despite their names, t1 and t2 contain the first and last MLT samples.
    row["t1"] = window["mlt"].iloc[0]
    row["t2"] = window["mlt"].iloc[-1]
    return row


def extract_daily_wave(
    frame: pd.DataFrame,
    url: str,
    satellite_date: str,
    *,
    monte_carlo_samples: int,
    day_seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract all three component tables from 47 overlapping daily windows."""

    daily = frame.copy().sort_values("time").reset_index(drop=True)
    daily["time"] = pd.to_datetime(daily["time"])
    reference_time = daily["time"].min().floor("h")
    daily["time_hours"] = (daily["time"] - reference_time) / pd.Timedelta(hours=1)
    date_value = daily["time"].iloc[0]
    for component in _COMPONENTS:
        daily[f"b_{component}_hp"] = high_pass_filter(daily[f"b_{component}"])

    rng = np.random.RandomState(day_seed)
    component_rows: dict[str, list[pd.DataFrame]] = {component: [] for component in _COMPONENTS}
    start_hour = 0.0
    end_hour = WINDOW_HOURS
    while start_hour <= 24.0 - WINDOW_HOURS:
        window = daily.loc[(daily["time_hours"] >= start_hour) & (daily["time_hours"] < end_hour)]
        high_pass_columns = [f"b_{component}_hp" for component in _COMPONENTS]
        if not window.empty and not window[high_pass_columns].isna().all().any():
            for component, abbreviation in (
                ("radial", "rad"),
                ("azimuthal", "az"),
                ("parallel", "par"),
            ):
                peaks = clean_ulf(
                    window,
                    component,
                    monte_carlo_samples=monte_carlo_samples,
                    rng=rng,
                )
                row = _peak_row(
                    peaks,
                    abbreviation,
                    url=url,
                    satellite_date=satellite_date,
                    date_value=date_value,
                    window=window,
                )
                if row is not None:
                    component_rows[component].append(row)
        start_hour += WINDOW_STEP_HOURS
        end_hour += WINDOW_STEP_HOURS

    outputs = []
    for component in _COMPONENTS:
        rows = component_rows[component]
        outputs.append(pd.concat(rows).reset_index(drop=True) if rows else pd.DataFrame())
    return outputs[0], outputs[1], outputs[2]


def _temp_paths(temp_dir: Path, index: int) -> dict[str, Path]:
    """Return the three per-day temporary filenames."""

    return {component: temp_dir / f"{component}_{index}.pkl" for component in _COMPONENTS}


def _metadata_path(temp_dir: Path, index: int) -> Path:
    """Return the identity sidecar for one set of temporary files."""

    return temp_dir / f"result_{index}.json"


def _expected_metadata(task: DetectionTask) -> dict[str, Any]:
    """Describe the inputs that make a temporary result reusable."""

    return {
        "algorithm_version": DETECTION_ALGORITHM_VERSION,
        "url": task.url,
        "satellite_date": task.satellite_date,
        "monte_carlo_samples": task.monte_carlo_samples,
        "day_seed": task.day_seed,
        "prepared_checkpoint_sha256": task.prepared_checkpoint_sha256,
    }


def _can_reuse(task: DetectionTask) -> bool:
    """Return true only when all temporary outputs match this exact task."""

    paths = _temp_paths(task.temp_dir, task.index)
    if not all(path.exists() for path in paths.values()):
        return False
    metadata_path = _metadata_path(task.temp_dir, task.index)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        try:
            output_sha256 = {path.name: sha256_file(path) for path in paths.values()}
        except OSError:
            return False
        return metadata == {
            **_expected_metadata(task),
            "output_sha256": output_sha256,
        }
    return False


def _execute_detection_task(task: DetectionTask) -> DetectionResult:
    """Compute and atomically persist one day's three component results."""

    try:
        radial, azimuthal, parallel = extract_daily_wave(
            task.frame,
            task.url,
            task.satellite_date,
            monte_carlo_samples=task.monte_carlo_samples,
            day_seed=task.day_seed,
        )
        values = {
            "radial": radial,
            "azimuthal": azimuthal,
            "parallel": parallel,
        }
        paths = _temp_paths(task.temp_dir, task.index)
        for component, value in values.items():
            _atomic_joblib_dump(value, paths[component])
        output_sha256 = {path.name: sha256_file(path) for path in paths.values()}
        _atomic_json_dump(
            {
                **_expected_metadata(task),
                "output_sha256": output_sha256,
            },
            _metadata_path(task.temp_dir, task.index),
        )
        return DetectionResult(task.index, reused=False)
    except Exception as exc:
        return DetectionResult(
            task.index,
            reused=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _bounded_process_results(
    executor: Any,
    tasks: Sequence[DetectionTask],
    *,
    max_in_flight: int,
) -> Iterator[DetectionResult]:
    """Submit tasks incrementally and yield results as workers finish.

    A :class:`DetectionTask` carries a daily DataFrame. Bounding the submitted
    futures therefore bounds the serialized DataFrames waiting in the process
    pool's work queue.
    """

    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1")

    task_iterator = iter(tasks)
    in_flight: set[Any] = set()

    def submit_next() -> bool:
        try:
            task = next(task_iterator)
        except StopIteration:
            return False
        in_flight.add(executor.submit(_execute_detection_task, task))
        return True

    for _ in range(max_in_flight):
        if not submit_next():
            break

    while in_flight:
        finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
        for future in finished:
            in_flight.remove(future)
            yield future.result()
        for _ in range(len(finished)):
            if not submit_next():
                break


def _derive_day_seed(random_seed: int | None, index: int) -> int | None:
    """Derive a deterministic independent 32-bit seed for one day."""

    if random_seed is None:
        return None
    sequence = np.random.SeedSequence([random_seed, index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _merge_temp_results(checkpoint_dir: Path, temp_dir: Path, task_count: int) -> dict[str, int]:
    """Merge and promote the three component checkpoints as one stage result."""

    merged: dict[str, list[pd.DataFrame]] = {component: [] for component in _COMPONENTS}
    for index in range(task_count):
        paths = _temp_paths(temp_dir, index)
        for component in _COMPONENTS:
            value = _joblib().load(paths[component])
            if not isinstance(value, pd.DataFrame):
                raise TypeError(f"Temporary result is not a DataFrame: {paths[component]}")
            if not value.empty:
                merged[component].append(value)

    staging_dir = Path(tempfile.mkdtemp(prefix=".detection-output-", dir=checkpoint_dir))
    try:
        for component, filename in _OUTPUT_FILES.items():
            _atomic_joblib_dump(merged[component], staging_dir / filename)
        for filename in _OUTPUT_FILES.values():
            os.replace(staging_dir / filename, checkpoint_dir / filename)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return {component: len(values) for component, values in merged.items()}


def _selected_product_digest(
    selected: list[tuple[ProductRecord, pd.DataFrame]],
) -> str:
    """Hash the ordered NOAA product identities used by one detection run."""

    identity = [
        {
            "date": record.observation_date.isoformat(),
            "satellite": record.satellite,
            "url": record.url,
            "version": list(record.version),
        }
        for record, _ in selected
    ]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_detection(
    checkpoint_dir: Path,
    temp_dir: Path,
    *,
    workers: int,
    force: bool = False,
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SAMPLES,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, int]:
    """Run or resume the full Pc5 detection stage.

    ``checkpoint_dir`` supplies the prepared daily lists and receives the
    intermediate and final detection checkpoints. ``temp_dir`` contains the
    reusable ``radial_N.pkl``, ``azimuthal_N.pkl``, and ``parallel_N.pkl``
    results.  Paths must be explicit :class:`pathlib.Path` objects.

    This is a resource-intensive full-interval operation at the paper defaults.
    Existing per-day files are reused only after their URL, date, Monte Carlo
    sample count, seed, algorithm version, and output hashes are validated.
    Successful days remain resumable if another day fails; final merged outputs
    are written only when every day has succeeded.
    """

    if not isinstance(checkpoint_dir, Path) or not isinstance(temp_dir, Path):
        raise TypeError("checkpoint_dir and temp_dir must be pathlib.Path objects")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if (
        isinstance(monte_carlo_samples, bool)
        or not isinstance(monte_carlo_samples, int)
        or monte_carlo_samples < 1
    ):
        raise ValueError("monte_carlo_samples must be a positive integer")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer or None")

    checkpoint_dir = checkpoint_dir.resolve()
    temp_dir = temp_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    _, selected = _validated_latest_products(checkpoint_dir)
    if not selected:
        raise RuntimeError("Pc5 detection requires at least one prepared daily product")
    prepared_paths = [checkpoint_dir / f"df_{satellite:02d}.pkl" for satellite in SATELLITE_NUMBERS]
    missing_prepared = [path for path in prepared_paths if not path.is_file()]
    if missing_prepared:
        raise FileNotFoundError(
            "Pc5 detection requires every satellite checkpoint; missing: "
            + ", ".join(path.name for path in missing_prepared)
        )
    LOGGER.info("Hashing prepared GOES checkpoints for input identity")
    prepared_hashes = {path.name: hash_file(path) for path in prepared_paths}
    completion_path = checkpoint_dir / "detection_complete.json"
    incomplete_path = checkpoint_dir / "detection_incomplete.json"
    processed_path = checkpoint_dir / "processed_data.pkl"
    run_identity = {
        "schema_version": 1,
        "algorithm_version": DETECTION_ALGORITHM_VERSION,
        "task_count": len(selected),
        "monte_carlo_samples": monte_carlo_samples,
        "random_seed": random_seed,
        "processed_data_sha256": hash_file(processed_path),
        "prepared_sha256": prepared_hashes,
        "selected_products_sha256": _selected_product_digest(selected),
    }
    acquisition_marker = checkpoint_dir / "acquisition_complete.json"
    if acquisition_marker.is_file():
        run_identity["acquisition_marker_sha256"] = hash_file(acquisition_marker)

    # The incomplete marker is published before the former completion claim is
    # removed, so every interruption state is distinguishable from a complete
    # run or an externally supplied checkpoint set.
    _atomic_json_dump({"status": "incomplete", **run_identity}, incomplete_path)
    completion_path.unlink(missing_ok=True)
    tasks = [
        DetectionTask(
            index=index,
            frame=frame,
            url=record.url,
            satellite_date=record.observation_date.strftime("%Y%m%d"),
            temp_dir=temp_dir,
            monte_carlo_samples=monte_carlo_samples,
            day_seed=_derive_day_seed(random_seed, index),
            prepared_checkpoint_sha256=prepared_hashes[f"df_{record.satellite:02d}.pkl"],
        )
        for index, (record, frame) in enumerate(selected)
    ]

    results: list[DetectionResult] = []
    pending: list[DetectionTask] = []
    for task in tasks:
        if not force and _can_reuse(task):
            results.append(DetectionResult(task.index, reused=True))
        else:
            # Mark a parameter-mismatched result incomplete before its
            # recomputation begins.
            _atomic_json_dump(
                {"status": "incomplete", **_expected_metadata(task)},
                _metadata_path(task.temp_dir, task.index),
            )
            pending.append(task)

    LOGGER.info(
        "Pc5 detection: %d daily products, %d reusable, %d pending",
        len(tasks),
        len(results),
        len(pending),
    )

    if workers == 1:
        for completed_count, task in enumerate(pending, start=1):
            results.append(_execute_detection_task(task))
            if completed_count % 100 == 0 or completed_count == len(pending):
                LOGGER.info(
                    "Pc5 detection progress: %d/%d pending products finished",
                    completed_count,
                    len(pending),
                )
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            process_results = _bounded_process_results(
                executor,
                pending,
                max_in_flight=2 * workers,
            )
            for completed_count, result in enumerate(process_results, start=1):
                results.append(result)
                if completed_count % 100 == 0 or completed_count == len(pending):
                    LOGGER.info(
                        "Pc5 detection progress: %d/%d pending products finished",
                        completed_count,
                        len(pending),
                    )

    failures = [result for result in results if result.error is not None]
    failure_payload = [
        {
            "index": result.index,
            "url": tasks[result.index].url,
            "error": result.error,
        }
        for result in sorted(failures, key=lambda item: item.index)
    ]
    _atomic_json_dump(failure_payload, checkpoint_dir / "detection_failures.json")
    if failures:
        raise RuntimeError(
            f"Pc5 detection failed for {len(failures)} daily products; "
            "successful temporary results were retained for resume"
        )

    merged_counts = _merge_temp_results(checkpoint_dir, temp_dir, len(tasks))
    output_hashes = {
        filename: hash_file(checkpoint_dir / filename) for filename in _OUTPUT_FILES.values()
    }
    _atomic_json_dump(
        {
            "status": "complete",
            **run_identity,
            "output_table_counts": merged_counts,
            "output_sha256": output_hashes,
        },
        completion_path,
    )
    incomplete_path.unlink(missing_ok=True)
    return {
        "days": len(tasks),
        "computed": sum(not result.reused for result in results),
        "reused": sum(result.reused for result in results),
        "radial_tables": merged_counts["radial"],
        "azimuthal_tables": merged_counts["azimuthal"],
        "parallel_tables": merged_counts["parallel"],
    }


__all__ = [
    "PC5_MAX_HZ",
    "PC5_MIN_HZ",
    "SIGNIFICANCE_LEVEL",
    "clean_ulf",
    "extract_daily_wave",
    "hanning_peak_model",
    "high_pass_filter",
    "run_detection",
]

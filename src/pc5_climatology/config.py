"""Executable constants and repository paths for the paper calculation.

The constants in this module are authoritative: the calculation imports them
directly, while ``configs/paper.toml`` is a human-readable summary checked
against :func:`paper_parameter_summary` by the test suite.  Keeping that
direction of dependency prevents documentation from silently changing a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

STUDY_START = date(1995, 7, 1)
STUDY_END = date(2025, 5, 10)
SATELLITE_NUMBERS = tuple(range(8, 19))
COMPONENTS = ("radial", "azimuthal", "parallel")
TIME_STANDARD = "UTC"

CADENCE_SECONDS = 60
MFA_BASELINE_MINUTES = 30
MAGNETIC_FIELD_ABSOLUTE_LIMIT_NT = 1024.0
ORBIT_IQR_MULTIPLIER = 1.5

DETECTION_WINDOW_MINUTES = 60
DETECTION_STEP_MINUTES = 30
WINDOWS_PER_COMPLETE_DAY = (24 * 60 - DETECTION_WINDOW_MINUTES) // DETECTION_STEP_MINUTES + 1
DETECTION_HIGHPASS_CUTOFF_MINUTES = 30
DETECTION_HIGHPASS_ORDER = 5
DETECTION_USES_HANNING_WINDOW = True
FFT_PADDED_SAMPLES = 1440
PC5_FREQUENCY_MIN_HZ = 0.0016
PC5_FREQUENCY_MAX_HZ = 0.0067
CANDIDATE_AMPLITUDE_MIN_NT = 1.0
MAXIMUM_CLEAN_ITERATIONS = 50
PEAK_FIT_HALF_WINDOW_MAX = 5
PEAK_FIT_HALF_WINDOW_MIN = 2
DEFAULT_MONTE_CARLO_SAMPLES = 5000
DEFAULT_RANDOM_SEED = 2025
MAXIMUM_FALSE_ALARM_PROBABILITY = 0.05
RETAINED_PEAKS_PER_WINDOW = 3
AMPLITUDE_FORMULA = "4 / N * abs(rfft)"
PUBLISHED_RANDOM_SEED_RECORDED = False

FREQUENCY_BIN_COUNT = 20
FREQUENCY_BIN_SUMMARY = "median amplitude"
POWER_LAW_FIT_FORMULA = "amplitude = coefficient * frequency ^ exponent"

MLT_SECTORS = {
    "dawn": (3.0, 9.0, False),
    "day": (9.0, 15.0, False),
    "dusk": (15.0, 21.0, False),
    "night": (21.0, 3.0, True),
}

SOLAR_WIND_LOWER_QUANTILE = 0.25
SOLAR_WIND_UPPER_QUANTILE = 0.75
SOLAR_WIND_VARIABLES = ("solar_wind_speed", "gsm_bz", "dynamic_pressure")
FIGURE_3_BZ_USES_SIGN = True

ANNUAL_BZ_USES_ABSOLUTE_VALUE = True
ANNUAL_OMNI_LAST_YEAR = STUDY_END.year
TEMPORAL_HIGHPASS_CUTOFF_FREQUENCY_PER_YEAR = 0.2
TEMPORAL_HIGHPASS_CUTOFF_PERIOD_YEARS = 1.0 / TEMPORAL_HIGHPASS_CUTOFF_FREQUENCY_PER_YEAR
TEMPORAL_HIGHPASS_ORDER = 5
SOLAR_CYCLE_23_START_YEAR = 1996
SOLAR_CYCLE_24_BOUNDARY_YEAR = 2009
SOLAR_CYCLE_24_END_YEAR_EXCLUSIVE = 2020

FIGURE_1_MLT_HISTOGRAM_BINS = 48
FIGURE_1_FREQUENCY_HISTOGRAM_BINS = 50

FIGURE_NAMES = ("Fig01.jpg", "Fig02.jpg", "Fig03.jpg", "Fig04.jpg")
CORRELATIONS_TABLE_NAME = "correlations_solar_cycles.csv"
FIGURE_DPI = 600

DETECTION_ALGORITHM_VERSION = 1

OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat"


def paper_parameter_summary() -> dict[str, Any]:
    """Return the complete expected content of ``configs/paper.toml``.

    This function serializes executable constants for a parity test.  Runtime
    stages do not read the TOML file.
    """

    return {
        "study": {
            "start_date": STUDY_START.isoformat(),
            "end_date": STUDY_END.isoformat(),
            "satellites": list(SATELLITE_NUMBERS),
            "components": list(COMPONENTS),
            "time_standard": TIME_STANDARD,
        },
        "preprocessing": {
            "cadence_seconds": CADENCE_SECONDS,
            "mfa_baseline_minutes": MFA_BASELINE_MINUTES,
            "magnetic_field_absolute_limit_nt": MAGNETIC_FIELD_ABSOLUTE_LIMIT_NT,
            "orbit_iqr_multiplier": ORBIT_IQR_MULTIPLIER,
        },
        "detection": {
            "window_minutes": DETECTION_WINDOW_MINUTES,
            "step_minutes": DETECTION_STEP_MINUTES,
            "windows_per_complete_day": WINDOWS_PER_COMPLETE_DAY,
            "highpass_cutoff_minutes": DETECTION_HIGHPASS_CUTOFF_MINUTES,
            "highpass_order": DETECTION_HIGHPASS_ORDER,
            "hanning_window": DETECTION_USES_HANNING_WINDOW,
            "fft_padded_samples": FFT_PADDED_SAMPLES,
            "frequency_min_hz": PC5_FREQUENCY_MIN_HZ,
            "frequency_max_hz": PC5_FREQUENCY_MAX_HZ,
            "candidate_amplitude_min_nt": CANDIDATE_AMPLITUDE_MIN_NT,
            "maximum_clean_iterations": MAXIMUM_CLEAN_ITERATIONS,
            "peak_fit_half_window_max": PEAK_FIT_HALF_WINDOW_MAX,
            "peak_fit_half_window_min": PEAK_FIT_HALF_WINDOW_MIN,
            "monte_carlo_samples": DEFAULT_MONTE_CARLO_SAMPLES,
            "repository_random_seed": DEFAULT_RANDOM_SEED,
            "maximum_false_alarm_probability": MAXIMUM_FALSE_ALARM_PROBABILITY,
            "retained_peaks_per_window": RETAINED_PEAKS_PER_WINDOW,
            "amplitude_formula": AMPLITUDE_FORMULA,
            "published_random_seed_recorded": PUBLISHED_RANDOM_SEED_RECORDED,
        },
        "frequency_binning": {
            "equal_width_bins": FREQUENCY_BIN_COUNT,
            "summary": FREQUENCY_BIN_SUMMARY,
            "fit": POWER_LAW_FIT_FORMULA,
        },
        "mlt": {
            name: {
                "start_hour": start,
                "end_hour": end,
                **({"wraps_midnight": True} if wraps_midnight else {}),
            }
            for name, (start, end, wraps_midnight) in MLT_SECTORS.items()
        },
        "solar_wind_conditions": {
            "lower_quantile": SOLAR_WIND_LOWER_QUANTILE,
            "upper_quantile": SOLAR_WIND_UPPER_QUANTILE,
            "variables": list(SOLAR_WIND_VARIABLES),
            "figure_3_bz_uses_sign": FIGURE_3_BZ_USES_SIGN,
        },
        "temporal": {
            "annual_bz_uses_absolute_value": ANNUAL_BZ_USES_ABSOLUTE_VALUE,
            "omni_last_year": ANNUAL_OMNI_LAST_YEAR,
            "highpass_cutoff_frequency_per_year": (TEMPORAL_HIGHPASS_CUTOFF_FREQUENCY_PER_YEAR),
            "highpass_order": TEMPORAL_HIGHPASS_ORDER,
            "solar_cycle_23_start_year": SOLAR_CYCLE_23_START_YEAR,
            "solar_cycle_24_boundary_year": SOLAR_CYCLE_24_BOUNDARY_YEAR,
            "solar_cycle_24_end_year_exclusive": SOLAR_CYCLE_24_END_YEAR_EXCLUSIVE,
        },
        "figure_1": {
            "mlt_histogram_bins": FIGURE_1_MLT_HISTOGRAM_BINS,
            "frequency_histogram_bins": FIGURE_1_FREQUENCY_HISTOGRAM_BINS,
        },
        "outputs": {
            "figure_names": list(FIGURE_NAMES),
            "correlations_table": CORRELATIONS_TABLE_NAME,
            "figure_dpi": FIGURE_DPI,
        },
    }


@dataclass(frozen=True)
class RepositoryPaths:
    """All mutable locations used by a reproduction run.

    Callers may point ``checkpoint_dir`` at any existing checkpoint directory.
    This avoids copying multi-gigabyte local files while keeping the repository
    itself free of research data.
    """

    root: Path
    checkpoint_dir: Path
    external_data_dir: Path
    temporary_dir: Path
    figures_dir: Path
    tables_dir: Path
    omni_file: Path

    @classmethod
    def from_root(
        cls,
        root: Path | str,
        *,
        checkpoint_dir: Path | str | None = None,
        omni_file: Path | str | None = None,
        figures_dir: Path | str | None = None,
        tables_dir: Path | str | None = None,
    ) -> "RepositoryPaths":
        root_path = Path(root).expanduser().resolve()

        def resolve(value: Path | str | None, default: Path) -> Path:
            if value is None:
                return default
            candidate = Path(value).expanduser()
            if candidate.is_absolute():
                return candidate.resolve()
            return (root_path / candidate).resolve()

        checkpoints = resolve(checkpoint_dir, root_path / "data" / "checkpoints")
        external = root_path / "data" / "external"
        return cls(
            root=root_path,
            checkpoint_dir=checkpoints,
            external_data_dir=external,
            temporary_dir=root_path / ".cache" / "detection",
            figures_dir=resolve(figures_dir, root_path / "outputs" / "figures"),
            tables_dir=resolve(tables_dir, root_path / "outputs" / "tables"),
            omni_file=resolve(omni_file, external / "omni2_all_years.dat"),
        )

    @property
    def observation_counts_file(self) -> Path:
        return self.checkpoint_dir / "observation_counts_by_year.csv"

    @property
    def acquisition_completion_file(self) -> Path:
        return self.checkpoint_dir / "acquisition_complete.json"

    @property
    def detection_completion_file(self) -> Path:
        return self.checkpoint_dir / "detection_complete.json"

    @property
    def detection_incomplete_file(self) -> Path:
        return self.checkpoint_dir / "detection_incomplete.json"

    @property
    def catalog_completion_file(self) -> Path:
        return self.checkpoint_dir / "catalog_complete.json"

    @property
    def catalog_incomplete_file(self) -> Path:
        return self.checkpoint_dir / "catalog_incomplete.json"

    def create_runtime_directories(self) -> None:
        for directory in (
            self.checkpoint_dir,
            self.external_data_dir,
            self.temporary_dir,
            self.figures_dir,
            self.tables_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

"""Small, explicit statistical operations for the four paper figures.

The calculations use these rules:

* Figure 3 expands every event against every hourly OMNI row on the same day.
* Figure fit annotations use Pearson correlation coefficients.
* Event checkpoints are consumed without a second date filter.
* Figure 4 limits annual OMNI values through calendar year 2025 before filtering.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Union

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from .config import (
    ANNUAL_OMNI_LAST_YEAR,
    FREQUENCY_BIN_COUNT,
    MLT_SECTORS,
    SATELLITE_NUMBERS,
    SOLAR_CYCLE_23_START_YEAR,
    SOLAR_CYCLE_24_BOUNDARY_YEAR,
    SOLAR_CYCLE_24_END_YEAR_EXCLUSIVE,
    SOLAR_WIND_LOWER_QUANTILE,
    SOLAR_WIND_UPPER_QUANTILE,
    STUDY_END,
    STUDY_START,
    TEMPORAL_HIGHPASS_CUTOFF_PERIOD_YEARS,
    TEMPORAL_HIGHPASS_ORDER,
)

CATALOG_FILENAMES: Final = {
    "radial": "radial_powers_freq_mlt_date.pkl",
    "azimuthal": "az_powers_freq_mlt_date.pkl",
    "parallel": "par_powers_freq_mlt_date.pkl",
}

OMNI_USECOLS: Final = (0, 1, 2, 9, 12, 15, 16, 24, 28)


@dataclass(frozen=True)
class PowerLawFit:
    """Parameters and plotting coordinates for ``y = coefficient * x**exponent``."""

    exponent: float
    coefficient: float
    fitted_x: np.ndarray
    fitted_y: np.ndarray


def load_event_catalogs(checkpoint_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all stored rows from the three component event catalogs.

    Date coverage is defined by the acquisition input, so this loader does not
    apply a second date filter.
    """

    checkpoint_dir = Path(checkpoint_dir)
    catalogs: dict[str, pd.DataFrame] = {}
    for component, filename in CATALOG_FILENAMES.items():
        path = checkpoint_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {component} event catalog: {path}")
        value = joblib.load(path)
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame in {path}")
        required = {"date", "t1", "t2", "freq", "power"}
        missing = sorted(required.difference(value.columns))
        if missing:
            raise KeyError(f"Event catalog {path} is missing columns: {missing}")
        catalogs[component] = value
    return catalogs


def power_law(
    x: Union[np.ndarray, pd.Series],
    exponent: float,
    coefficient: float,
) -> np.ndarray:
    """Evaluate the paper's amplitude-frequency power law."""

    return np.asarray(x, dtype=float) ** exponent * coefficient


def fit_power_law(x: Iterable[float], y: Iterable[float]) -> PowerLawFit:
    """Fit the unconstrained two-parameter power law."""

    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if x_values.size < 2:
        raise ValueError("At least two finite bins are required for a power-law fit")

    parameters, _ = curve_fit(power_law, x_values, y_values, maxfev=10_000)
    fitted_x = np.linspace(np.min(x_values), np.max(x_values), 100)
    fitted_y = power_law(fitted_x, *parameters)
    return PowerLawFit(
        exponent=float(parameters[0]),
        coefficient=float(parameters[1]),
        fitted_x=fitted_x,
        fitted_y=fitted_y,
    )


def pearson_r(
    x: Iterable[float],
    y: Iterable[float],
) -> float:
    """Return Pearson's correlation coefficient for paired finite values."""

    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    return float(np.corrcoef(x_values[valid], y_values[valid])[0, 1])


def pearson_r_and_p_value(
    x: Iterable[float],
    y: Iterable[float],
) -> tuple[float, float]:
    """Return the Pearson correlation coefficient and two-sided p-value."""

    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(valid) < 2:
        return float("nan"), float("nan")
    result = pearsonr(x_values[valid], y_values[valid])
    return float(result.statistic), float(result.pvalue)


def mlt_sector_binned_medians(
    data: pd.DataFrame, *, bins: int = FREQUENCY_BIN_COUNT
) -> dict[str, pd.DataFrame]:
    """Compute Figure-2 median amplitudes in its four MLT sectors."""

    minimum = float(data["freq"].min())
    maximum = float(data["freq"].max())
    edges = np.linspace(minimum, maximum, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    dawn_start, dawn_end, _ = MLT_SECTORS["dawn"]
    day_start, day_end, _ = MLT_SECTORS["day"]
    dusk_start, dusk_end, _ = MLT_SECTORS["dusk"]
    night_start, night_end, _ = MLT_SECTORS["night"]
    sectors = {
        "dawn": data.loc[(data["t1"] >= dawn_start) & (data["t1"] < dawn_end)],
        "day": data.loc[(data["t1"] >= day_start) & (data["t1"] < day_end)],
        "dusk": data.loc[(data["t1"] >= dusk_start) & (data["t1"] < dusk_end)],
        "night": pd.concat(
            [
                data.loc[(data["t1"] >= night_start) & (data["t1"] <= 24)],
                data.loc[(data["t1"] >= 0) & (data["t1"] < night_end)],
            ]
        ),
    }

    result: dict[str, pd.DataFrame] = {}
    for name, sector in sectors.items():
        working = sector.copy()
        working["freq_bin"] = pd.cut(
            working["freq"],
            bins=edges,
            labels=centers,
            include_lowest=True,
        )
        binned = working.groupby("freq_bin", observed=False)["power"].median().reset_index()
        binned["freq_bin"] = binned["freq_bin"].astype(float) * 1_000
        result[name] = binned.rename(columns={"freq_bin": "frq_centroid", "power": "median_power"})
    return result


def read_omni_hourly(omni_path: Path) -> pd.DataFrame:
    """Read the hourly OMNI2 columns used in Figures 3 and 4."""

    omni_path = Path(omni_path)
    if not omni_path.is_file():
        raise FileNotFoundError(f"Missing OMNI2 input: {omni_path}")
    try:
        raw = pd.read_csv(
            omni_path,
            header=None,
            sep=r"\s+",
            usecols=list(OMNI_USECOLS),
        )
    except ValueError as exc:
        if "Usecols do not match columns" in str(exc):
            raise ValueError("OMNI2 input must contain at least 29 columns") from exc
        raise
    if set(raw.columns) != set(OMNI_USECOLS):
        raise ValueError("OMNI2 input must contain at least 29 columns")

    year = pd.to_numeric(raw[0], errors="raise").astype(int)
    day_of_year = pd.to_numeric(raw[1], errors="raise").astype(int)
    hour = pd.to_numeric(raw[2], errors="raise").astype(int)
    dates = (
        pd.to_datetime(year.astype(str), format="%Y")
        + pd.to_timedelta(day_of_year - 1, unit="D")
        + pd.to_timedelta(hour, unit="h")
    )

    omni = pd.DataFrame(
        {
            "date": dates,
            "sw": raw[24].replace(9999, np.nan),
            "B_x": raw[12].replace(999.9, np.nan),
            "B_y": raw[15].replace(999.9, np.nan),
            "B_z": raw[16].replace(999.9, np.nan),
            "B_tot": raw[9].replace(999.9, np.nan),
            "dyn_pres": raw[28].replace(99.99, np.nan),
        }
    )
    # Return every source row. Figure 4 selects years from the first event year,
    # and Figure 3 selects OMNI rows through its event-date join.
    return omni


def validate_omni_study_grid(
    omni: pd.DataFrame,
    *,
    source: Path | None = None,
) -> None:
    """Require one unique OMNI timestamp per study hour."""

    label = str(source) if source is not None else "OMNI2 input"
    if omni.empty:
        raise ValueError(f"OMNI2 input contains no rows: {label}")
    if "date" not in omni:
        raise KeyError(f"OMNI2 input has no date column: {label}")

    dates = pd.DatetimeIndex(pd.to_datetime(omni["date"], errors="coerce"))
    if dates.hasnans:
        raise ValueError(f"OMNI2 input contains invalid timestamps: {label}")
    # Figure 4 calculates its first value from calendar year 1995, whereas
    # GOES event acquisition begins partway through that year.
    omni_start = pd.Timestamp(STUDY_START.year, 1, 1)
    study_end_exclusive = pd.Timestamp(STUDY_END) + pd.Timedelta(days=1)
    expected = pd.date_range(
        omni_start,
        study_end_exclusive,
        freq="h",
        inclusive="left",
    )
    observed = dates[(dates >= omni_start) & (dates < study_end_exclusive)]
    observed = observed.sort_values()
    if observed.has_duplicates:
        raise ValueError(f"OMNI2 input contains duplicate hourly timestamps: {label}")
    if not observed.equals(expected):
        missing = expected.difference(observed)
        unexpected = observed.difference(expected)
        raise ValueError(
            "OMNI2 input does not contain the complete hourly grid for "
            f"{omni_start.date().isoformat()}..{STUDY_END.isoformat()} "
            f"({len(missing)} missing, {len(unexpected)} unexpected): {label}"
        )


def prepare_condition_omni(omni: pd.DataFrame) -> pd.DataFrame:
    """Replace zero field values and calculate the Figure-3 clock angle."""

    result = omni.copy()
    for column in ("B_x", "B_y", "B_z", "B_tot"):
        result[column] = result[column].replace(0, 0.001)
    result["clock_angle"] = np.rad2deg(np.arctan(result["B_y"] / result["B_z"]))
    return result


def expand_events_with_daily_omni(
    events: pd.DataFrame,
    omni: pd.DataFrame,
) -> pd.DataFrame:
    """Expand events through a daily many-to-many event/OMNI merge.

    Every hourly OMNI timestamp is reduced to its calendar date before the
    merge.  Consequently each event is repeated once for every OMNI hour on the
    same date; this is intentionally *not* an hourly nearest-time join.
    """

    event_rows = events.copy()
    event_rows["date"] = pd.to_datetime(event_rows["date"])
    omni_rows = omni.copy()
    omni_rows["imf_date_time"] = pd.to_datetime(omni_rows["date"])
    omni_rows["date"] = omni_rows["imf_date_time"].dt.normalize()
    return event_rows.merge(omni_rows, how="left", on="date")


def split_solar_wind_conditions(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split an expanded component table using its own Q1/Q3 thresholds."""

    quantiles = [SOLAR_WIND_LOWER_QUANTILE, SOLAR_WIND_UPPER_QUANTILE]
    speed_q1, speed_q3 = data["sw"].quantile(quantiles)
    bz_q1, bz_q3 = data["B_z"].quantile(quantiles)
    pressure_q1, pressure_q3 = data["dyn_pres"].quantile(quantiles)

    strong = data.loc[
        (data["sw"] >= speed_q3) & (data["B_z"] <= bz_q1) & (data["dyn_pres"] >= pressure_q3)
    ]
    moderate = data.loc[
        (data["sw"] < speed_q3)
        & (data["sw"] >= speed_q1)
        & (data["B_z"] > bz_q1)
        & (data["B_z"] <= bz_q3)
        & (data["dyn_pres"] < pressure_q3)
        & (data["dyn_pres"] >= pressure_q1)
    ]
    weak = data.loc[
        (data["sw"] < speed_q1) & (data["B_z"] > bz_q3) & (data["dyn_pres"] < pressure_q1)
    ]

    # Drop rows missing any merged field, including fields not used directly in
    # the condition predicates.
    return {
        "strong": strong.dropna().reset_index(drop=True),
        "moderate": moderate.dropna().reset_index(drop=True),
        "weak": weak.dropna().reset_index(drop=True),
    }


def condition_binned_statistics(
    data: pd.DataFrame, *, bins: int = FREQUENCY_BIN_COUNT
) -> pd.DataFrame:
    """Compute Figure-3 within-condition frequency-bin statistics."""

    columns = [
        "freq",
        "amp",
        "amp_std",
        "sw",
        "sw_std",
        "bz",
        "bz_std",
        "pres",
        "pres_std",
    ]
    if data.empty:
        return pd.DataFrame(columns=columns)

    minimum = float(data["freq"].min())
    maximum = float(data["freq"].max())
    edges = np.linspace(minimum, maximum, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    working = data.copy()
    working["freq_bin"] = pd.cut(
        working["freq"],
        bins=edges,
        labels=centers,
        include_lowest=True,
    )

    rows: list[dict[str, float]] = []
    for frequency_bin, group in working.groupby("freq_bin", observed=True):
        if group.empty:
            continue
        rows.append(
            {
                "freq": float(frequency_bin) * 1_000,
                "amp": float(group["power"].median()),
                "amp_std": float(group["power"].std()),
                "sw": float(group["sw"].median()),
                "sw_std": float(group["sw"].std()),
                "bz": float(group["B_z"].median()),
                "bz_std": float(group["B_z"].std()),
                "pres": float(group["dyn_pres"].median()),
                "pres_std": float(group["dyn_pres"].std()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("freq").reset_index(drop=True)


def highpass_yearly(
    years: Iterable[float],
    values: Iterable[float],
    *,
    cutoff_period_years: float = TEMPORAL_HIGHPASS_CUTOFF_PERIOD_YEARS,
    order: int = TEMPORAL_HIGHPASS_ORDER,
) -> np.ndarray:
    """Apply the zero-phase Butterworth high-pass filter.

    The Figure-4 default is a five-year cutoff period, equivalent to
    1/5 year⁻¹.
    """

    year_values = np.asarray(list(years), dtype=float)
    data_values = np.asarray(list(values), dtype=float)
    if year_values.size != data_values.size:
        raise ValueError("years and values must have the same length")
    if year_values.size < 2:
        raise ValueError("At least two annual values are required")

    mean_step = float(np.mean(np.diff(year_values)))
    sampling_frequency = 1 / (mean_step * 365) * 86_400
    nyquist = 0.5 * sampling_frequency
    normalized_cutoff = (1 / (cutoff_period_years * mean_step * 365) * 86_400) / nyquist
    numerator, denominator = signal.butter(order, normalized_cutoff, "high")
    return signal.filtfilt(numerator, denominator, data_values)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=destination.suffix or ".csv",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Write a CSV atomically in the destination directory."""

    _atomic_csv(frame, Path(destination))


def _normalise_observation_counts(frame: pd.DataFrame) -> pd.DataFrame:
    year_column = next((name for name in ("year", "date") if name in frame.columns), None)
    count_column = next(
        (name for name in ("observation_count", "dat_count", "count") if name in frame.columns),
        None,
    )
    if year_column is None or count_column is None:
        raise KeyError(
            "Observation summary must contain year/observation_count "
            "(date/dat_count is also accepted)"
        )

    years = frame[year_column]
    if pd.api.types.is_datetime64_any_dtype(years) or not pd.api.types.is_numeric_dtype(years):
        years = pd.to_datetime(years).dt.year
    result = pd.DataFrame(
        {
            "date": pd.to_numeric(years, errors="raise").astype(int),
            "dat_count": pd.to_numeric(frame[count_column], errors="raise"),
        }
    )
    return result.groupby("date", as_index=False)["dat_count"].sum().sort_values("date")


def _derive_observation_counts(checkpoint_dir: Path) -> pd.DataFrame:
    dates: list[pd.Timestamp] = []
    for satellite in SATELLITE_NUMBERS:
        checkpoint = checkpoint_dir / f"df_{satellite:02d}.pkl"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing {checkpoint}; provide an observation-count summary "
                "to avoid loading the large daily checkpoints"
            )
        payload = joblib.load(checkpoint)
        if not isinstance(payload, (list, tuple)):
            raise TypeError(f"Expected a list-like daily checkpoint in {checkpoint}")
        for daily in payload:
            if not isinstance(daily, pd.DataFrame) or daily.empty or "time" not in daily:
                raise TypeError(f"Invalid daily observation object in {checkpoint}")
            dates.append(pd.Timestamp(daily["time"].iloc[0]))

    date_frame = pd.DataFrame({"date": dates})
    counts = date_frame.groupby(date_frame["date"].dt.year).size().reset_index(name="dat_count")
    counts.columns = ["date", "dat_count"]
    return counts


def load_or_build_observation_counts(
    checkpoint_dir: Path,
    observation_counts_path: Path,
) -> pd.DataFrame:
    """Load canonical counts, or derive them from GOES daily checkpoints."""

    checkpoint_dir = Path(checkpoint_dir)
    observation_counts_path = Path(observation_counts_path)
    if observation_counts_path.is_file():
        suffix = observation_counts_path.suffix.lower()
        if suffix == ".csv":
            loaded = pd.read_csv(observation_counts_path)
        elif suffix in {".pkl", ".joblib"}:
            loaded = joblib.load(observation_counts_path)
        elif suffix == ".parquet":
            loaded = pd.read_parquet(observation_counts_path)
        else:
            raise ValueError(f"Unsupported observation summary format: {observation_counts_path}")
        if not isinstance(loaded, pd.DataFrame):
            raise TypeError(f"Expected a table in {observation_counts_path}")
        return _normalise_observation_counts(loaded)

    counts = _derive_observation_counts(checkpoint_dir)
    public_summary = counts.rename(columns={"date": "year", "dat_count": "observation_count"})
    _atomic_csv(public_summary, observation_counts_path)
    return counts


def annual_occurrence_rates(
    catalogs: dict[str, pd.DataFrame],
    observation_counts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate annual peak-row counts per available satellite-day record."""

    rates: dict[str, pd.DataFrame] = {}
    for component, events in catalogs.items():
        annual = events.groupby(events["date"].dt.year)["power"].count().reset_index()
        annual = annual.merge(observation_counts, how="left", on="date")
        annual["oc_rate"] = annual["power"] / annual["dat_count"]
        annual["oc_rate_hp"] = highpass_yearly(
            annual["date"],
            annual["oc_rate"],
            cutoff_period_years=TEMPORAL_HIGHPASS_CUTOFF_PERIOD_YEARS,
            order=TEMPORAL_HIGHPASS_ORDER,
        )
        rates[component] = annual
    return rates


def annual_omni_parameters(
    omni: pd.DataFrame,
    *,
    first_year: int,
    last_year: int = ANNUAL_OMNI_LAST_YEAR,
) -> pd.DataFrame:
    """Calculate bounded annual OMNI means and five-year residuals."""

    if first_year > last_year:
        raise ValueError("first_year must not be later than last_year")
    dates = omni["date"]
    years = dates.dt.year
    study_end_exclusive = pd.Timestamp(STUDY_END) + pd.Timedelta(days=1)
    working = omni.loc[
        (years >= first_year) & (years <= last_year) & (dates < study_end_exclusive)
    ].copy()
    working["B_z"] = working["B_z"].abs()
    annual = (
        working.groupby(working["date"].dt.year)[["sw", "B_tot", "dyn_pres", "B_z"]]
        .mean()
        .reset_index()
    )
    for parameter in ("sw", "B_tot", "dyn_pres", "B_z"):
        annual[f"{parameter}_hp"] = highpass_yearly(
            annual["date"],
            annual[parameter],
            cutoff_period_years=TEMPORAL_HIGHPASS_CUTOFF_PERIOD_YEARS,
            order=TEMPORAL_HIGHPASS_ORDER,
        )
    return annual


def solar_cycle_correlations(
    occurrence_rates: dict[str, pd.DataFrame],
    annual_omni: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Pearson-r table for solar cycles 23 and 24."""

    cycles = {
        "Solar Cycle 23": (
            SOLAR_CYCLE_23_START_YEAR,
            SOLAR_CYCLE_24_BOUNDARY_YEAR,
        ),
        "Solar Cycle 24": (
            SOLAR_CYCLE_24_BOUNDARY_YEAR,
            SOLAR_CYCLE_24_END_YEAR_EXCLUSIVE,
        ),
    }
    parameter_sets = {
        "oc_rate": ("sw", "dyn_pres", "B_z"),
        "oc_rate_hp": ("sw_hp", "dyn_pres_hp", "B_z_hp"),
    }
    rows: list[dict[str, object]] = []

    for cycle, (start, end) in cycles.items():
        for component, rates in occurrence_rates.items():
            for rate_type, parameters in parameter_sets.items():
                for parameter in parameters:
                    aligned = rates[["date", rate_type]].merge(
                        annual_omni[["date", parameter]],
                        how="inner",
                        on="date",
                    )
                    aligned = aligned.loc[
                        (aligned["date"] >= start) & (aligned["date"] < end)
                    ].dropna()
                    if len(aligned) <= 1:
                        continue
                    correlation, p_value = pearson_r_and_p_value(
                        aligned[rate_type], aligned[parameter]
                    )
                    rows.append(
                        {
                            "Cycle": cycle,
                            "Component": component,
                            "Rate Type": rate_type,
                            "Parameter": parameter,
                            "Pearson's R": correlation,
                            "P-value": p_value,
                        }
                    )
    return pd.DataFrame(
        rows,
        columns=["Cycle", "Component", "Rate Type", "Parameter", "Pearson's R", "P-value"],
    )

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pc5_climatology.catalog import (
    _flatten_detection_batches,
    _top_three_catalog,
    build_event_catalogs,
    catalog_outputs_are_usable,
)
from pc5_climatology.statistics import (
    OMNI_USECOLS,
    annual_omni_parameters,
    expand_events_with_daily_omni,
    highpass_yearly,
    load_or_build_observation_counts,
    pearson_r,
    read_omni_hourly,
    solar_cycle_correlations,
    split_solar_wind_conditions,
)


def _detection_table(prefix: str) -> pd.DataFrame:
    dates = pd.to_datetime(["2001-02-03", "2004-05-06"])
    data: dict[str, object] = {
        "date": dates,
        "t1": [1.0, 2.0],
        "t2": [2.0, 3.0],
    }
    for rank in range(1, 4):
        data[f"peak_freq_{prefix}_{rank}"] = [rank / 1000, rank / 1000 + 0.0001]
        data[f"peak_pow_{prefix}_{rank}"] = [10.0 - rank, 20.0 - rank]
    return pd.DataFrame(data)


def test_catalog_keeps_three_ranked_peaks_and_all_input_rows() -> None:
    result = _top_three_catalog(
        _detection_table("rad"),
        frequency_prefix="peak_freq_rad_",
        amplitude_prefix="peak_pow_rad_",
    )

    assert len(result) == 6
    assert result["freq"].tolist() == [
        0.001,
        0.0011,
        0.002,
        0.0021,
        0.003,
        0.0031,
    ]
    assert result["date"].min() == pd.Timestamp("2001-02-03")
    assert result["date"].max() == pd.Timestamp("2004-05-06")


def test_catalog_accepts_fewer_than_three_peak_ranks() -> None:
    detections = _detection_table("az").drop(columns=["peak_freq_az_3", "peak_pow_az_3"])

    result = _top_three_catalog(
        detections,
        frequency_prefix="peak_freq_az_",
        amplitude_prefix="peak_pow_az_",
    )

    assert len(result) == 4
    assert result["freq"].max() == 0.0021


def test_catalog_accepts_a_detection_checkpoint_with_no_events(tmp_path: Path) -> None:
    source = tmp_path / "empty.pkl"

    detections = _flatten_detection_batches([[], []], source=source)
    result = _top_three_catalog(
        detections,
        frequency_prefix="peak_freq_par_",
        amplitude_prefix="peak_pow_par_",
    )

    assert result.empty
    assert list(result.columns) == ["date", "t1", "t2", "freq", "power"]


def test_catalog_rejects_event_rows_without_peak_columns() -> None:
    detections = pd.DataFrame({"date": [pd.Timestamp("2001-01-01")], "t1": [1.0], "t2": [2.0]})

    with pytest.raises(KeyError, match="no peak rank columns"):
        _top_three_catalog(
            detections,
            frequency_prefix="peak_freq_par_",
            amplitude_prefix="peak_pow_par_",
        )


def test_detection_placeholders_must_be_empty_lists(tmp_path: Path) -> None:
    source = tmp_path / "detection.pkl"
    table = _detection_table("rad")

    flattened = _flatten_detection_batches([[], table], source=source)
    pd.testing.assert_frame_equal(flattened, table.reset_index(drop=True))

    with pytest.raises(TypeError, match="Nonempty list"):
        _flatten_detection_batches([[{"unexpected": "data"}], table], source=source)


def test_catalog_stage_writes_all_checkpoint_filenames(tmp_path: Path) -> None:
    specifications = {
        "radial": ("rad", "Frequency_Power_radial_new_1h.pkl"),
        "azimuthal": ("az", "Frequency_Power_azimuthal_new_1h.pkl"),
        "parallel": ("par", "Frequency_Power_parallel_new_1h.pkl"),
    }
    for _, (prefix, filename) in specifications.items():
        joblib.dump([_detection_table(prefix)], tmp_path / filename)

    outputs = build_event_catalogs(tmp_path)

    assert set(outputs) == {"radial", "azimuthal", "parallel"}
    assert all(path.is_file() for path in outputs.values())
    assert all(len(joblib.load(path)) == 6 for path in outputs.values())
    assert (tmp_path / "catalog_complete.json").is_file()
    assert catalog_outputs_are_usable(tmp_path)

    (tmp_path / "catalog_incomplete.json").write_text("{}", encoding="utf-8")
    assert not catalog_outputs_are_usable(tmp_path)
    (tmp_path / "catalog_incomplete.json").unlink()
    first_detection = tmp_path / "Frequency_Power_radial_new_1h.pkl"
    first_detection.write_bytes(b"changed detection input")
    assert not catalog_outputs_are_usable(tmp_path)


def test_catalog_validates_present_inputs_without_requiring_pruned_ancestors(
    tmp_path: Path,
) -> None:
    specifications = {
        "rad": "Frequency_Power_radial_new_1h.pkl",
        "az": "Frequency_Power_azimuthal_new_1h.pkl",
        "par": "Frequency_Power_parallel_new_1h.pkl",
    }
    for prefix, filename in specifications.items():
        joblib.dump([_detection_table(prefix)], tmp_path / filename)
    build_event_catalogs(tmp_path)

    (tmp_path / specifications["az"]).unlink()
    (tmp_path / specifications["par"]).unlink()
    assert catalog_outputs_are_usable(tmp_path)

    (tmp_path / specifications["rad"]).write_bytes(b"changed remaining input")
    assert not catalog_outputs_are_usable(tmp_path)


def test_daily_omni_join_is_many_to_many() -> None:
    events = pd.DataFrame(
        {
            "event": ["a", "b"],
            "date": pd.to_datetime(["2001-01-01", "2001-01-01"]),
        }
    )
    omni = pd.DataFrame(
        {
            "date": pd.to_datetime(["2001-01-01 00:00", "2001-01-01 01:00"]),
            "sw": [400.0, 500.0],
        }
    )

    expanded = expand_events_with_daily_omni(events, omni)

    assert len(expanded) == 4
    assert expanded.groupby("event").size().to_dict() == {"a": 2, "b": 2}


def test_omni_reader_loads_only_the_columns_used_by_figures(tmp_path: Path, monkeypatch) -> None:
    omni_path = tmp_path / "omni.dat"
    values = list(range(29))
    values[0:3] = [2001, 32, 4]
    omni_path.write_text(" ".join(map(str, values)) + "\n", encoding="utf-8")
    observed_usecols: list[int] = []
    original_read_csv = pd.read_csv

    def recording_read_csv(*args, **kwargs):
        observed_usecols.extend(kwargs["usecols"])
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", recording_read_csv)

    result = read_omni_hourly(omni_path)

    assert observed_usecols == list(OMNI_USECOLS)
    assert result.columns.tolist() == [
        "date",
        "sw",
        "B_x",
        "B_y",
        "B_z",
        "B_tot",
        "dyn_pres",
    ]
    assert result.loc[0, "date"] == pd.Timestamp("2001-02-01 04:00:00")


def test_pearson_r_preserves_the_correlation_sign() -> None:
    value = pearson_r([1, 2, 3], [3, 2, 1])
    assert value == -1.0


def test_correlation_table_names_pearson_r_explicitly() -> None:
    result = solar_cycle_correlations({}, pd.DataFrame())

    assert result.columns.tolist() == [
        "Cycle",
        "Component",
        "Rate Type",
        "Parameter",
        "Pearson's R",
        "P-value",
    ]


def test_solar_wind_condition_boundaries_match_defined_inequalities() -> None:
    frame = pd.DataFrame(
        {
            "sw": [1.0, 2.0, 3.0, 4.0],
            "B_z": [4.0, 3.0, 2.0, 1.0],
            "dyn_pres": [1.0, 2.0, 3.0, 4.0],
            "required_non_null": [1.0] * 4,
        }
    )

    conditions = split_solar_wind_conditions(frame)

    assert conditions["strong"]["sw"].tolist() == [4.0]
    assert conditions["weak"]["sw"].tolist() == [1.0]
    assert conditions["moderate"]["sw"].tolist() == [2.0, 3.0]


def test_default_highpass_is_one_over_five_years() -> None:
    years = np.arange(1995, 2026, dtype=float)
    values = np.sin(np.arange(len(years)) / 2.0) + np.arange(len(years)) / 20.0

    implicit = highpass_yearly(years, values)
    explicit = highpass_yearly(years, values, cutoff_period_years=5, order=5)

    np.testing.assert_allclose(implicit, explicit, rtol=0, atol=0)


def test_annual_omni_ignores_rows_after_the_study_end() -> None:
    years = np.arange(1995, 2031)
    omni = pd.DataFrame(
        {
            "date": pd.to_datetime([f"{year}-01-01" for year in years]),
            "sw": 350 + years % 17,
            "B_tot": 5 + (years % 7) / 10,
            "dyn_pres": 1 + (years % 5) / 10,
            "B_z": -2 - (years % 3) / 10,
        }
    )
    omni = pd.concat(
        [
            omni,
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2025-06-01")],
                    "sw": [10_000.0],
                    "B_tot": [10_000.0],
                    "dyn_pres": [10_000.0],
                    "B_z": [-10_000.0],
                }
            ),
        ],
        ignore_index=True,
    )

    bounded = annual_omni_parameters(omni.loc[omni["date"] <= "2025-05-10"], first_year=1995)
    with_future_rows = annual_omni_parameters(omni, first_year=1995)

    pd.testing.assert_frame_equal(bounded, with_future_rows)
    assert with_future_rows["date"].max() == 2025


def test_canonical_observation_count_schema_is_normalised(tmp_path: Path) -> None:
    path = tmp_path / "observation_counts_by_year.csv"
    pd.DataFrame({"year": [2000, 2001], "observation_count": [10, 20]}).to_csv(path, index=False)

    result = load_or_build_observation_counts(tmp_path, path)

    assert result.columns.tolist() == ["date", "dat_count"]
    assert result.to_dict("records") == [
        {"date": 2000, "dat_count": 10},
        {"date": 2001, "dat_count": 20},
    ]

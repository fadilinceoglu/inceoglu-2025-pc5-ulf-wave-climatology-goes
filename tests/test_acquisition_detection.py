from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import replace
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import pytest

import pc5_climatology.detection as detection_module
from pc5_climatology.acquisition import (
    _aiohttp,
    _initial_acquisition_state,
    _recover_transaction,
    _validate_checkpoint_identity,
)
from pc5_climatology.detection import (
    DetectionTask,
    _bounded_process_results,
    _can_reuse,
    _expected_metadata,
    _temp_paths,
    _validated_latest_products,
    run_detection,
)
from pc5_climatology.io import sha256_file


def _url(satellite: int, date_token: str, version: str = "1-0-0") -> str:
    return (
        "https://example.test/archive/"
        f"dn_magn-l2-hires_g{satellite:02d}_d{date_token}_v{version}.nc"
    )


def _frame(date_token: str, satellite: int) -> pd.DataFrame:
    timestamp = pd.Timestamp(date_token)
    return pd.DataFrame(
        {
            "time": [timestamp, timestamp + pd.Timedelta(minutes=1)],
            "mlt": [1.0, 1.1],
            "b_radial": [1.0, 2.0],
            "b_azimuthal": [2.0, 3.0],
            "b_parallel": [3.0, 4.0],
            "satellite": [f"GOES{satellite:02d}"] * 2,
        }
    )


def test_detection_associates_inputs_by_identity(tmp_path: Path) -> None:
    urls = [
        _url(14, "20040615"),
        _url(9, "20020210"),
        _url(11, "20030920"),
    ]
    joblib.dump(urls, tmp_path / "processed_data.pkl")
    joblib.dump([_frame("2002-02-10", 9)], tmp_path / "df_09.pkl")
    joblib.dump([_frame("2003-09-20", 11)], tmp_path / "df_11.pkl")
    joblib.dump([_frame("2004-06-15", 14)], tmp_path / "df_14.pkl")

    _, selected = _validated_latest_products(tmp_path)

    # Stable product identity controls task order and keeps each URL attached
    # to its own daily frame.
    assert [record.satellite for record, _ in selected] == [9, 11, 14]
    assert [frame["time"].iloc[0].date() for _, frame in selected] == [
        date(2002, 2, 10),
        date(2003, 9, 20),
        date(2004, 6, 15),
    ]


def test_acquisition_rejects_duplicate_processed_urls() -> None:
    url = _url(8, "20000101")
    frames = {8: [_frame("2000-01-01", 8), _frame("2000-01-01", 8)]}

    with pytest.raises(RuntimeError, match="repeats exact processed URLs"):
        _validate_checkpoint_identity([url, url], frames)


def test_full_force_reset_does_not_read_discarded_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "processed_data.pkl").write_bytes(b"not a Joblib file")
    transaction = tmp_path / "acquisition_transaction.json"
    transaction.write_text("not JSON", encoding="utf-8")

    monkeypatch.setattr(
        "pc5_climatology.acquisition._load_list",
        lambda path: pytest.fail(f"discarded checkpoint was read: {path}"),
    )

    processed, failures, frames = _initial_acquisition_state(tmp_path, clear_all=True)

    assert processed == []
    assert failures == []
    assert all(value == [] for value in frames.values())
    assert not transaction.exists()


def test_aiohttp_rejects_a_preimported_c_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_aiohttp = types.ModuleType("aiohttp")
    fake_helpers = types.ModuleType("aiohttp.helpers")
    fake_helpers.NO_EXTENSIONS = False
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    monkeypatch.setitem(sys.modules, "aiohttp.helpers", fake_helpers)
    monkeypatch.delenv("AIOHTTP_NO_EXTENSIONS", raising=False)

    with pytest.raises(RuntimeError, match="imported before the acquisition stage"):
        _aiohttp()

    assert os.environ["AIOHTTP_NO_EXTENSIONS"] == "1"


@pytest.mark.parametrize("date_token", ["19941231", "20260101"])
def test_acquisition_rejects_matching_products_outside_paper_interval(
    date_token: str,
) -> None:
    url = _url(8, date_token)
    frame_date = pd.Timestamp(date_token).strftime("%Y-%m-%d")

    with pytest.raises(RuntimeError, match="outside the inclusive paper interval"):
        _validate_checkpoint_identity([url], {8: [_frame(frame_date, 8)]})


@pytest.mark.parametrize("date_token", ["19941231", "20260101"])
def test_detection_rejects_matching_products_outside_paper_interval(
    tmp_path: Path,
    date_token: str,
) -> None:
    url = _url(8, date_token)
    frame_date = pd.Timestamp(date_token).strftime("%Y-%m-%d")
    joblib.dump([url], tmp_path / "processed_data.pkl")
    joblib.dump([_frame(frame_date, 8)], tmp_path / "df_08.pkl")

    with pytest.raises(ValueError, match="outside the inclusive paper interval"):
        _validated_latest_products(tmp_path)


def test_detection_rejects_conflicting_frame_source_url(tmp_path: Path) -> None:
    url = _url(11, "20030304")
    frame = _frame("2003-03-04", 11)
    frame.attrs["source_url"] = _url(11, "20030304", "2-0-0")
    joblib.dump([url], tmp_path / "processed_data.pkl")
    joblib.dump([frame], tmp_path / "df_11.pkl")

    with pytest.raises(ValueError, match="source URL does not match"):
        _validated_latest_products(tmp_path)


def test_detection_rejects_duplicate_exact_product_urls(tmp_path: Path) -> None:
    url = _url(11, "20030304")
    joblib.dump([url, url], tmp_path / "processed_data.pkl")
    joblib.dump(
        [_frame("2003-03-04", 11), _frame("2003-03-04", 11)],
        tmp_path / "df_11.pkl",
    )

    with pytest.raises(ValueError, match="repeats exact product URLs"):
        _validated_latest_products(tmp_path)


def test_detection_refuses_an_empty_prepared_inventory(tmp_path: Path) -> None:
    joblib.dump([], tmp_path / "processed_data.pkl")

    with pytest.raises(RuntimeError, match="at least one prepared daily product"):
        run_detection(tmp_path, tmp_path / "temporary", workers=1)


def test_daily_detection_cache_binds_input_and_output_identity(
    tmp_path: Path,
) -> None:
    task = DetectionTask(
        index=0,
        frame=_frame("2003-03-04", 11),
        url=_url(11, "20030304"),
        satellite_date="20030304",
        temp_dir=tmp_path,
        monte_carlo_samples=5000,
        day_seed=123,
        prepared_checkpoint_sha256="a" * 64,
    )
    for component in ("radial", "azimuthal", "parallel"):
        joblib.dump(pd.DataFrame(), tmp_path / f"{component}_0.pkl")
    output_sha256 = {path.name: sha256_file(path) for path in _temp_paths(tmp_path, 0).values()}
    (tmp_path / "result_0.json").write_text(
        json.dumps(_expected_metadata(task), sort_keys=True), encoding="utf-8"
    )
    assert not _can_reuse(task)

    (tmp_path / "result_0.json").write_text(
        json.dumps(
            {**_expected_metadata(task), "output_sha256": output_sha256},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert _can_reuse(task)
    assert not _can_reuse(replace(task, prepared_checkpoint_sha256="b" * 64))

    (tmp_path / "radial_0.pkl").write_bytes(b"corrupted")
    assert not _can_reuse(task)


def test_parallel_submission_keeps_a_bounded_number_of_tasks_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_in_flight: list[int] = []

    class StubFuture:
        def __init__(self, value: int) -> None:
            self.value = value

        def result(self) -> int:
            return self.value

    class StubExecutor:
        def __init__(self) -> None:
            self.submitted: list[int] = []

        def submit(self, function: object, task: int) -> StubFuture:
            assert function is detection_module._execute_detection_task
            self.submitted.append(task)
            return StubFuture(task)

    def finish_one(futures: set[StubFuture], *, return_when: object):
        assert return_when is detection_module.FIRST_COMPLETED
        observed_in_flight.append(len(futures))
        completed = min(futures, key=lambda future: future.value)
        return {completed}, futures - {completed}

    monkeypatch.setattr(detection_module, "wait", finish_one)
    executor = StubExecutor()

    results = list(_bounded_process_results(executor, list(range(10)), max_in_flight=4))

    assert executor.submitted == list(range(10))
    assert sorted(results) == list(range(10))
    assert max(observed_in_flight) == 4


def test_product_journal_rolls_forward_orphan_frame(tmp_path: Path) -> None:
    first = _url(8, "20000101")
    second = _url(8, "20000102")
    processed = [first]
    frames = {8: [_frame("2000-01-01", 8), _frame("2000-01-02", 8)]}
    transaction = tmp_path / "acquisition_transaction.json"
    transaction.write_text(json.dumps({"operation": "product", "url": second}), encoding="utf-8")

    recovered, failures, recovered_frames = _recover_transaction(
        tmp_path, transaction, processed, [second], frames
    )

    assert recovered == [first, second]
    assert failures == []
    assert recovered_frames is frames
    assert not transaction.exists()
    assert joblib.load(tmp_path / "processed_data.pkl") == [first, second]


def test_product_journal_retries_when_frame_was_not_published(tmp_path: Path) -> None:
    first = _url(8, "20000101")
    pending = _url(8, "20000102")
    transaction = tmp_path / "acquisition_transaction.json"
    transaction.write_text(json.dumps({"operation": "product", "url": pending}), encoding="utf-8")

    recovered, _, _ = _recover_transaction(
        tmp_path,
        transaction,
        [first],
        [],
        {8: [_frame("2000-01-01", 8)]},
    )

    assert recovered == [first]
    assert not transaction.exists()

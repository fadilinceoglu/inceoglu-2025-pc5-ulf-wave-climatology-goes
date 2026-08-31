from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import pytest

import pc5_climatology.figures as figure_module
import pc5_climatology.pipeline as pipeline_module
from pc5_climatology.cli import build_parser, main, main_for_stage
from pc5_climatology.config import (
    DEFAULT_RANDOM_SEED,
    DETECTION_ALGORITHM_VERSION,
    OMNI2_URL,
    RepositoryPaths,
)
from pc5_climatology.io import artifact_record, sha256_file
from pc5_climatology.pipeline import PipelineOptions, ReproductionPipeline


def _pipeline(tmp_path: Path) -> ReproductionPipeline:
    return ReproductionPipeline(RepositoryPaths.from_root(tmp_path), PipelineOptions(workers=1))


def _touch(paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_runtime_fit_annotation_is_pearsons_r() -> None:
    figure, axis = figure_module.plt.subplots()
    frequencies = pd.Series([1.0, 2.0, 3.0, 4.0])
    binned = pd.DataFrame(
        {
            "freq": frequencies,
            "amp": 3.0 * frequencies ** (-0.5),
        }
    )
    try:
        figure_module._plot_power_law_panel(
            axis,
            binned,
            x_column="freq",
            y_column="amp",
            component="radial",
            color="tab:red",
            panel_label="a",
            correlation_y=0.55,
        )

        annotation = axis.texts[-1]
        assert annotation.get_text() == "Pearson's R=1.00"
        assert annotation.get_position() == (0.98, 0.55)
        assert annotation.get_horizontalalignment() == "right"
    finally:
        figure_module.plt.close(figure)


def _mark_acquisition_complete(pipeline: ReproductionPipeline) -> None:
    processed = pipeline.paths.checkpoint_dir / "processed_data.pkl"
    outputs = [
        processed,
        *pipeline.prepared_files,
        pipeline.paths.observation_counts_file,
    ]
    pipeline.paths.acquisition_completion_file.write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": 2,
                "start_date": "1995-07-01",
                "end_date": "2025-05-10",
                "processed_products": 1,
                "unresolved_failures": 0,
                "output_sha256": {path.name: sha256_file(path) for path in outputs},
            }
        ),
        encoding="utf-8",
    )


def test_default_all_refuses_before_downloading_when_heavy_inputs_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    downloaded = False

    def unexpected_download(*, force: bool) -> None:
        nonlocal downloaded
        downloaded = True

    monkeypatch.setattr(pipeline, "_download_omni_if_needed", unexpected_download)

    with pytest.raises(RuntimeError, match="resource-intensive full-interval"):
        pipeline.all(full=False)
    assert downloaded is False


def test_default_all_resumes_directly_from_catalogs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(pipeline.catalog_files + [pipeline.paths.omni_file])
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "_download_omni_if_needed",
        lambda *, force: calls.append("omni"),
    )
    monkeypatch.setattr(
        pipeline,
        "status",
        lambda: pytest.fail("stage orchestration must not eagerly validate all stages"),
    )
    monkeypatch.setattr(
        pipeline,
        "_detection_outputs_are_usable",
        lambda: pytest.fail("catalog resume must not validate obsolete detection files"),
    )
    monkeypatch.setattr(
        pipeline,
        "_acquire_goes_if_needed",
        lambda *, force: pytest.fail("acquisition must not run"),
    )
    monkeypatch.setattr(pipeline, "detect", lambda: pytest.fail("detection must not run"))
    monkeypatch.setattr(pipeline, "catalog", lambda: pytest.fail("catalog must not run"))
    monkeypatch.setattr(pipeline, "figures", lambda: calls.append("figures"))

    pipeline.all(full=False)

    assert calls == ["omni", "figures"]


def test_default_all_builds_catalog_from_detection_without_prepared_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(pipeline.detection_files + [pipeline.paths.omni_file])
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "_download_omni_if_needed", lambda *, force: None)

    def catalog() -> None:
        calls.append("catalog")
        _touch(pipeline.catalog_files)

    monkeypatch.setattr(pipeline, "catalog", catalog)
    monkeypatch.setattr(pipeline, "detect", lambda: pytest.fail("detection must not run"))
    monkeypatch.setattr(pipeline, "figures", lambda: calls.append("figures"))

    pipeline.all(full=False)

    assert calls == ["catalog", "figures"]


def test_individual_detection_does_not_recompute_complete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(pipeline.detection_files)

    import pc5_climatology.detection as detection

    monkeypatch.setattr(
        detection,
        "run_detection",
        lambda *args, **kwargs: pytest.fail("completed detection must be reused"),
    )
    pipeline.detect()


def test_forced_detection_invalidates_unmarked_catalogs_and_output_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RepositoryPaths.from_root(tmp_path)
    pipeline = ReproductionPipeline(paths, PipelineOptions(force=True))
    _touch(
        pipeline.prepared_files
        + pipeline.catalog_files
        + [paths.checkpoint_dir / "processed_data.pkl"]
    )
    manifest = pipeline._output_manifest_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    import pc5_climatology.detection as detection

    def completed_detection(*args, **kwargs) -> dict[str, int]:
        assert paths.catalog_incomplete_file.is_file()
        assert not manifest.exists()
        return {"tasks": 1, "completed": 1, "failed": 0}

    monkeypatch.setattr(detection, "run_detection", completed_detection)

    pipeline.detect()

    assert not pipeline._catalog_outputs_are_usable()


def test_interrupted_detection_leaves_old_catalogs_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RepositoryPaths.from_root(tmp_path)
    pipeline = ReproductionPipeline(paths, PipelineOptions(force=True))
    _touch(
        pipeline.prepared_files
        + pipeline.catalog_files
        + [paths.checkpoint_dir / "processed_data.pkl"]
    )

    import pc5_climatology.detection as detection

    def interrupted_detection(*args, **kwargs) -> dict[str, int]:
        raise RuntimeError("simulated detection interruption")

    monkeypatch.setattr(detection, "run_detection", interrupted_detection)

    with pytest.raises(RuntimeError, match="simulated detection interruption"):
        pipeline.detect()

    assert paths.catalog_incomplete_file.is_file()
    assert not pipeline._catalog_outputs_are_usable()


def test_individual_acquisition_does_not_recrawl_complete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(
        pipeline.prepared_files
        + [
            pipeline.paths.checkpoint_dir / "processed_data.pkl",
            pipeline.paths.omni_file,
            pipeline.paths.observation_counts_file,
        ]
    )
    _mark_acquisition_complete(pipeline)
    monkeypatch.setattr(pipeline, "_download_omni_if_needed", lambda *, force: None)
    monkeypatch.setattr(
        pipeline,
        "status",
        lambda: pytest.fail("acquire must validate only acquisition outputs"),
    )
    monkeypatch.setattr(
        pipeline,
        "_acquire_goes_if_needed",
        lambda *, force: pytest.fail("completed acquisition must be reused"),
    )

    pipeline.acquire()


def test_acquisition_marker_rejects_changed_output(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(
        pipeline.prepared_files
        + [
            pipeline.paths.checkpoint_dir / "processed_data.pkl",
            pipeline.paths.observation_counts_file,
        ]
    )
    _mark_acquisition_complete(pipeline)

    assert pipeline._acquisition_is_complete()
    pipeline.prepared_files[0].write_bytes(b"changed prepared input")
    assert not pipeline._acquisition_is_complete()


def test_nonforce_acquisition_never_reblesses_an_invalid_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(
        pipeline.prepared_files
        + [
            pipeline.paths.checkpoint_dir / "processed_data.pkl",
            pipeline.paths.observation_counts_file,
            pipeline.paths.omni_file,
        ]
    )
    _mark_acquisition_complete(pipeline)
    marker_before = pipeline.paths.acquisition_completion_file.read_bytes()
    pipeline.prepared_files[0].write_bytes(b"changed prepared content")
    monkeypatch.setattr(pipeline, "_download_omni_if_needed", lambda *, force: None)
    monkeypatch.setattr(
        pipeline,
        "_acquire_goes_if_needed",
        lambda *, force: pytest.fail("invalid completion must require --force"),
    )

    with pytest.raises(RuntimeError, match="does not match.*--force"):
        pipeline.acquire()

    assert pipeline.paths.acquisition_completion_file.read_bytes() == marker_before


def test_individual_forced_acquisition_invalidates_downstream_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RepositoryPaths.from_root(tmp_path)
    pipeline = ReproductionPipeline(paths, PipelineOptions(force=True))
    paths.detection_completion_file.parent.mkdir(parents=True, exist_ok=True)
    paths.detection_completion_file.write_text("{}", encoding="utf-8")
    paths.catalog_completion_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_download_omni_if_needed", lambda *, force: None)

    def acquire_after_invalidation(*, force: bool) -> dict[str, int]:
        assert force is True
        assert paths.detection_incomplete_file.is_file()
        assert paths.catalog_incomplete_file.is_file()
        assert not paths.detection_completion_file.exists()
        assert not paths.catalog_completion_file.exists()
        return {"attempted": 0, "completed": 0, "failed": 0}

    monkeypatch.setattr(pipeline, "_acquire_goes_if_needed", acquire_after_invalidation)

    pipeline.acquire()


def test_partial_acquisition_artifacts_do_not_bypass_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(
        pipeline.prepared_files
        + [
            pipeline.paths.checkpoint_dir / "processed_data.pkl",
            pipeline.paths.omni_file,
        ]
    )
    called: list[bool] = []
    monkeypatch.setattr(pipeline, "_download_omni_if_needed", lambda *, force: None)
    monkeypatch.setattr(
        pipeline,
        "_acquire_goes_if_needed",
        lambda *, force: called.append(force),
    )

    pipeline.acquire()

    assert called == [False]


def test_full_all_runs_missing_stages_in_causal_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "_download_omni_if_needed", lambda *, force: None)

    def acquire(*, force: bool) -> None:
        calls.append("acquire")
        _touch(pipeline.prepared_files + [pipeline.paths.checkpoint_dir / "processed_data.pkl"])

    def detect() -> None:
        calls.append("detect")
        _touch(pipeline.detection_files)

    def catalog() -> None:
        calls.append("catalog")
        _touch(pipeline.catalog_files)

    monkeypatch.setattr(pipeline, "_acquire_goes_if_needed", acquire)
    monkeypatch.setattr(pipeline, "detect", detect)
    monkeypatch.setattr(pipeline, "catalog", catalog)
    monkeypatch.setattr(pipeline, "figures", lambda: calls.append("figures"))

    pipeline.all(full=True)

    assert calls == ["acquire", "detect", "catalog", "figures"]


def test_forced_chain_invalidates_downstream_before_acquisition_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = RepositoryPaths.from_root(tmp_path)
    pipeline = ReproductionPipeline(paths, PipelineOptions(force=True))
    _touch(pipeline.detection_files + pipeline.catalog_files)
    paths.detection_completion_file.write_text("{}", encoding="utf-8")
    paths.catalog_completion_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_download_omni_if_needed", lambda *, force: None)

    def interrupted_acquisition(*, force: bool) -> None:
        assert force is True
        assert paths.detection_incomplete_file.is_file()
        assert paths.catalog_incomplete_file.is_file()
        raise RuntimeError("simulated acquisition interruption")

    monkeypatch.setattr(pipeline, "_acquire_goes_if_needed", interrupted_acquisition)

    with pytest.raises(RuntimeError, match="simulated acquisition interruption"):
        pipeline.all(full=True)

    assert not paths.detection_completion_file.exists()
    assert not paths.catalog_completion_file.exists()
    ordinary = ReproductionPipeline(paths, PipelineOptions())
    assert not ordinary._detection_outputs_are_usable()
    assert not ordinary._catalog_outputs_are_usable()


def test_cli_rejects_force_all_without_full(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires --full --force"):
        main(["all", "--force", "--root", str(tmp_path)])


def test_cli_renders_operational_error_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["all", "--root", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: Event catalogs")
    assert "resource-intensive full-interval" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("stage", "figure_number", "description"),
    [
        ("acquire", None, "Acquire and prepare the GOES magnetometer inputs."),
        ("detect", None, "Detect Pc5 events in the prepared GOES inputs."),
        ("catalog", None, "Build the up-to-three significant-peak event catalogs."),
        ("figure", 2, "Regenerate one paper figure."),
    ],
)
def test_fixed_stage_help_has_no_generic_stage_positionals(
    stage: str,
    figure_number: int | None,
    description: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main_for_stage(stage, ["--help"], figure_number=figure_number)

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert description in help_text
    assert "{all,acquire,detect,catalog,figures,figure,status}" not in help_text
    assert "figure_number" not in help_text
    assert "--full" not in help_text


@pytest.mark.parametrize("stage", ["acquire", "detect", "catalog", "figure"])
def test_stage_wrappers_accept_output_roots_for_manifest_invalidation(
    stage: str,
) -> None:
    arguments = ["--figures-dir", "/tmp/figures", "--tables-dir", "/tmp/tables"]
    if stage == "figure":
        arguments.insert(0, "1")
    parsed = build_parser(fixed_stage=stage).parse_args(arguments)

    assert parsed.figures_dir == Path("/tmp/figures")
    assert parsed.tables_dir == Path("/tmp/tables")


def test_status_is_read_only_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", "--root", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "catalog_artifacts": False,
        "cataloged": False,
        "detection_artifacts": False,
        "detected": False,
        "figures": False,
        "omni": False,
        "prepared": False,
        "prepared_artifacts": False,
    }
    assert not (tmp_path / "data").exists()


def test_artifact_record_accepts_custom_output_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    external = tmp_path / "external" / "result.txt"
    root.mkdir()
    external.parent.mkdir()
    external.write_text("result", encoding="utf-8")

    record = artifact_record(external, root)

    assert record["path"] == "result.txt"
    assert record["external"] is True
    assert record["bytes"] == 6
    assert len(record["sha256"]) == 64


def test_pipeline_reuses_file_digest_until_filesystem_revision_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    artifact = tmp_path / "large-checkpoint.pkl"
    artifact.write_bytes(b"first")
    calls: list[Path] = []
    original = pipeline_module.sha256_file

    def counting_hash(path: Path) -> str:
        calls.append(Path(path))
        return original(path)

    monkeypatch.setattr(pipeline_module, "sha256_file", counting_hash)

    first = pipeline._sha256_cached(artifact)
    assert pipeline._sha256_cached(artifact) == first
    assert len(calls) == 1

    original_mtime_ns = artifact.stat().st_mtime_ns
    artifact.write_bytes(b"other")
    os.utime(artifact, ns=(artifact.stat().st_atime_ns, original_mtime_ns))
    assert pipeline._sha256_cached(artifact) != first
    assert len(calls) == 2


def test_custom_outputs_keep_manifest_out_of_checkout(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    output_root = tmp_path / "custom-output"
    paths = RepositoryPaths.from_root(
        root,
        figures_dir=output_root / "figures",
        tables_dir=output_root / "tables",
    )
    pipeline = ReproductionPipeline(paths, PipelineOptions())
    _touch(pipeline.figure_files + [paths.tables_dir / "correlations_solar_cycles.csv"])
    paths.omni_file.parent.mkdir(parents=True, exist_ok=True)
    paths.omni_file.write_text("public-input", encoding="utf-8")
    parameter_file = root / "configs" / "paper.toml"
    parameter_file.parent.mkdir(parents=True, exist_ok=True)
    parameter_file.write_text("[study]\n", encoding="utf-8")
    _touch(pipeline.catalog_files + [paths.observation_counts_file])

    pipeline._write_output_manifest()

    assert (output_root / "manifest.json").is_file()
    assert not (root / "outputs" / "manifest.json").exists()
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["inputs"]] == ["data/external/omni2_all_years.dat"]
    assert manifest["inputs"][0]["external"] is True
    assert manifest["inputs"][0]["source_url"] == OMNI2_URL
    assert manifest["parameter_summary"]["path"] == "configs/paper.toml"


def test_single_canonical_figure_invalidates_complete_set_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(pipeline.catalog_files)
    manifest_path = pipeline._output_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"old": true}\n', encoding="utf-8")

    def render_figure_1(checkpoint_dir: Path, output_path: Path) -> Path:
        assert checkpoint_dir == pipeline.paths.checkpoint_dir
        assert not manifest_path.exists()
        output_path.write_bytes(b"new figure")
        return output_path

    monkeypatch.setattr(figure_module, "plot_figure_1", render_figure_1)

    pipeline.figure(1)

    assert pipeline.figure_files[0].read_bytes() == b"new figure"
    assert not manifest_path.exists()


def test_complete_figure_promotion_keeps_then_replaces_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    manifest_path = pipeline._output_manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"old": true}\n', encoding="utf-8")
    rendered: list[int] = []

    def render(number: int, *, output_dir: Path, table_dir: Path) -> None:
        assert manifest_path.exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"Fig{number:02d}.jpg").write_bytes(f"figure-{number}".encode("ascii"))
        if number == 4:
            (table_dir / "correlations_solar_cycles.csv").write_text(
                "cycle,value\n24,1\n", encoding="utf-8"
            )
        rendered.append(number)

    original_copyfile = pipeline_module.shutil.copyfile
    promotion_checks: list[bool] = []

    def copy_after_invalidation(source: Path, destination: Path) -> str:
        promotion_checks.append(not manifest_path.exists())
        return original_copyfile(source, destination)

    monkeypatch.setattr(pipeline, "figure", render)
    monkeypatch.setattr(pipeline_module.shutil, "copyfile", copy_after_invalidation)

    pipeline.figures()

    assert rendered == [1, 2, 3, 4]
    assert promotion_checks and all(promotion_checks)
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 5


def test_incomplete_detection_marker_blocks_checkpoint_reuse(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    _touch(pipeline.detection_files)
    pipeline.paths.detection_incomplete_file.write_text("{}", encoding="utf-8")

    assert pipeline._detection_outputs_are_usable() is False


def test_unmarked_detection_set_does_not_hide_nondefault_parameters(
    tmp_path: Path,
) -> None:
    paths = RepositoryPaths.from_root(tmp_path)
    default_pipeline = ReproductionPipeline(paths, PipelineOptions())
    _touch(default_pipeline.detection_files)

    assert default_pipeline._detection_outputs_are_usable()
    assert not ReproductionPipeline(
        paths, PipelineOptions(random_seed=7)
    )._detection_outputs_are_usable()
    assert not ReproductionPipeline(
        paths, PipelineOptions(monte_carlo_samples=1000)
    )._detection_outputs_are_usable()


def test_detection_marker_binds_parameters_inputs_and_outputs(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    processed = pipeline.paths.checkpoint_dir / "processed_data.pkl"
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_bytes(b"prepared-products")
    for index, path in enumerate(pipeline.prepared_files):
        path.write_bytes(f"prepared-{index}".encode("ascii"))
    for index, path in enumerate(pipeline.detection_files):
        path.write_bytes(f"component-{index}".encode("ascii"))
    output_hashes = {path.name: sha256_file(path) for path in pipeline.detection_files}
    pipeline.paths.detection_completion_file.write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": 1,
                "algorithm_version": DETECTION_ALGORITHM_VERSION,
                "task_count": 3,
                "monte_carlo_samples": 5000,
                "random_seed": DEFAULT_RANDOM_SEED,
                "processed_data_sha256": sha256_file(processed),
                "prepared_sha256": {
                    path.name: sha256_file(path) for path in pipeline.prepared_files
                },
                "selected_products_sha256": "a" * 64,
                "output_sha256": output_hashes,
            }
        ),
        encoding="utf-8",
    )

    assert pipeline._detection_is_complete()
    pipeline.detection_files[0].write_bytes(b"changed")
    assert not pipeline._detection_is_complete()


def test_detection_marker_changes_when_prepared_input_changes(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    processed = pipeline.paths.checkpoint_dir / "processed_data.pkl"
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_bytes(b"prepared-products")
    for index, path in enumerate(pipeline.prepared_files):
        path.write_bytes(f"prepared-{index}".encode("ascii"))
    for index, path in enumerate(pipeline.detection_files):
        path.write_bytes(f"component-{index}".encode("ascii"))
    pipeline.paths.detection_completion_file.write_text(
        json.dumps(
            {
                "status": "complete",
                "schema_version": 1,
                "algorithm_version": DETECTION_ALGORITHM_VERSION,
                "task_count": 3,
                "monte_carlo_samples": 5000,
                "random_seed": DEFAULT_RANDOM_SEED,
                "processed_data_sha256": sha256_file(processed),
                "prepared_sha256": {
                    path.name: sha256_file(path) for path in pipeline.prepared_files
                },
                "selected_products_sha256": "b" * 64,
                "output_sha256": {
                    path.name: sha256_file(path) for path in pipeline.detection_files
                },
            }
        ),
        encoding="utf-8",
    )

    assert pipeline._detection_is_complete()
    pipeline.prepared_files[0].write_bytes(b"changed-prepared-input")
    assert not pipeline._detection_is_complete()


def test_omni_validation_checks_structure(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.dat"
    invalid.write_text(" ".join(["0"] * 28) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 29"):
        ReproductionPipeline._validate_omni_file(invalid)


def test_changed_omni_download_invalidates_output_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = _pipeline(tmp_path)
    pipeline.paths.omni_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline.paths.omni_file.write_bytes(b"old omni")
    manifest = pipeline._output_manifest_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"complete": true}\n', encoding="utf-8")
    monkeypatch.setattr(pipeline, "_validate_omni_file", lambda path: None)
    monkeypatch.setattr(
        pipeline_module.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(b"new omni"),
    )

    pipeline._download_omni_if_needed(force=True)

    assert pipeline.paths.omni_file.read_bytes() == b"new omni"
    assert not manifest.exists()


def test_omni_validation_requires_every_interior_hour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    omni_path = tmp_path / "omni.dat"
    omni_path.touch()
    expected = pd.date_range(
        "1995-01-01",
        "2025-05-11",
        freq="h",
        inclusive="left",
    )
    complete = pd.DataFrame({"date": expected})
    frame = complete
    monkeypatch.setattr("pc5_climatology.statistics.read_omni_hourly", lambda path: frame)
    ReproductionPipeline._validate_omni_file(omni_path)

    frame = frame.drop(index=len(frame) // 2).reset_index(drop=True)
    with pytest.raises(ValueError, match="complete hourly grid"):
        ReproductionPipeline._validate_omni_file(omni_path)

    frame = pd.concat([complete, complete.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate hourly timestamps"):
        ReproductionPipeline._validate_omni_file(omni_path)

    frame = pd.concat(
        [complete, pd.DataFrame({"date": [pd.Timestamp("2025-05-11")] * 2})],
        ignore_index=True,
    )
    ReproductionPipeline._validate_omni_file(omni_path)


def test_figure_omni_reader_enforces_the_study_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    incomplete = pd.DataFrame({"date": pd.to_datetime(["1995-01-01 00:00", "2025-05-10 23:00"])})
    monkeypatch.setattr(figure_module, "read_omni_hourly", lambda path: incomplete)

    with pytest.raises(ValueError, match="complete hourly grid"):
        figure_module._read_validated_omni(tmp_path / "omni.dat")


def test_tracked_manifest_matches_in_repository_files() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "outputs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for record in manifest["artifacts"]:
        path = manifest_path.parent / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    for record in manifest["inputs"]:
        if record.get("external"):
            assert record["source_url"] == OMNI2_URL
            assert len(record["sha256"]) == 64
            continue
        path = root / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert sha256_file(path) == record["sha256"]
    parameter_summary = manifest["parameter_summary"]
    parameter_path = root / parameter_summary["path"]
    assert parameter_path.stat().st_size == parameter_summary["bytes"]
    assert sha256_file(parameter_path) == parameter_summary["sha256"]

"""Causal stage orchestration for full and checkpoint-based reproduction."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import (
    COMPONENTS,
    CORRELATIONS_TABLE_NAME,
    DEFAULT_MONTE_CARLO_SAMPLES,
    DEFAULT_RANDOM_SEED,
    DETECTION_ALGORITHM_VERSION,
    FIGURE_NAMES,
    OMNI2_URL,
    SATELLITE_NUMBERS,
    STUDY_END,
    STUDY_START,
    RepositoryPaths,
)
from .io import (
    artifact_record,
    atomic_destination,
    atomic_write_json,
    require_files,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    workers: int = 1
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SAMPLES
    random_seed: int | None = DEFAULT_RANDOM_SEED
    force: bool = False


class ReproductionPipeline:
    """Run the paper stages without hiding their data dependencies."""

    def __init__(self, paths: RepositoryPaths, options: PipelineOptions) -> None:
        self.paths = paths
        self.options = options
        self._file_hash_cache: dict[
            Path,
            tuple[tuple[int, int, int | None, int | None, int | None], str],
        ] = {}
        self._acquisition_validation_cache = None
        self._detection_validation_cache = None
        self._catalog_validation_cache = None

    @property
    def prepared_files(self) -> list[Path]:
        return [self.paths.checkpoint_dir / f"df_{number:02d}.pkl" for number in SATELLITE_NUMBERS]

    @property
    def detection_files(self) -> list[Path]:
        return [
            self.paths.checkpoint_dir / f"Frequency_Power_{component}_new_1h.pkl"
            for component in COMPONENTS
        ]

    @property
    def catalog_files(self) -> list[Path]:
        return [
            self.paths.checkpoint_dir / name
            for name in (
                "radial_powers_freq_mlt_date.pkl",
                "az_powers_freq_mlt_date.pkl",
                "par_powers_freq_mlt_date.pkl",
            )
        ]

    @property
    def figure_files(self) -> list[Path]:
        return [self.paths.figures_dir / name for name in FIGURE_NAMES]

    def status(self) -> dict[str, bool]:
        prepared_artifacts = (
            all(path.is_file() for path in self.prepared_files)
            and (self.paths.checkpoint_dir / "processed_data.pkl").is_file()
        )
        detection_artifacts = all(path.is_file() for path in self.detection_files)
        catalog_artifacts = all(path.is_file() for path in self.catalog_files)
        return {
            "omni": self.paths.omni_file.is_file(),
            "prepared_artifacts": prepared_artifacts,
            "prepared": prepared_artifacts and self._acquisition_is_complete(),
            "detection_artifacts": detection_artifacts,
            "detected": detection_artifacts and self._detection_is_complete(),
            "catalog_artifacts": catalog_artifacts,
            "cataloged": catalog_artifacts and self._catalog_outputs_are_usable(),
            "figures": all(path.is_file() for path in self.figure_files),
        }

    def _acquisition_is_complete(self) -> bool:
        """Validate the acquisition marker against every downstream artifact."""

        processed = self.paths.checkpoint_dir / "processed_data.pkl"
        state = self._path_state(
            [
                *self.prepared_files,
                processed,
                self.paths.observation_counts_file,
                self.paths.acquisition_completion_file,
            ]
        )
        if (
            self._acquisition_validation_cache is not None
            and self._acquisition_validation_cache[0] == state
        ):
            return self._acquisition_validation_cache[1]
        result = self._validate_acquisition_completion()
        self._acquisition_validation_cache = (state, result)
        return result

    def _validate_acquisition_completion(self) -> bool:
        """Perform the uncached acquisition-marker validation."""

        marker = self.paths.acquisition_completion_file
        processed = self.paths.checkpoint_dir / "processed_data.pkl"
        outputs = [
            processed,
            *self.prepared_files,
            self.paths.observation_counts_file,
        ]
        if not marker.is_file() or not all(path.is_file() for path in outputs):
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not (
            payload.get("status") == "complete"
            and payload.get("schema_version") == 2
            and payload.get("start_date") == STUDY_START.isoformat()
            and payload.get("end_date") == STUDY_END.isoformat()
            and isinstance(payload.get("processed_products"), int)
            and payload.get("processed_products") > 0
            and payload.get("unresolved_failures") == 0
        ):
            return False
        try:
            expected_hashes = {path.name: self._sha256_cached(path) for path in outputs}
        except OSError:
            return False
        return payload.get("output_sha256") == expected_hashes

    def _require_consistent_acquisition_marker(self) -> bool:
        """Return completion, but never re-bless a present invalid marker."""

        complete = self._acquisition_is_complete()
        marker = self.paths.acquisition_completion_file
        if marker.exists() and not complete:
            raise RuntimeError(
                "The existing acquisition completion marker does not match "
                "its prepared outputs. Preserve the files for diagnosis or "
                "restart acquisition explicitly with --force."
            )
        return complete

    def _detection_is_complete(self) -> bool:
        """Validate the stage marker against inputs and current parameters."""

        state = self._path_state(
            [
                *self.detection_files,
                *self.prepared_files,
                self.paths.detection_completion_file,
                self.paths.detection_incomplete_file,
                self.paths.checkpoint_dir / "processed_data.pkl",
                self.paths.acquisition_completion_file,
            ]
        )
        if (
            self._detection_validation_cache is not None
            and self._detection_validation_cache[0] == state
        ):
            return self._detection_validation_cache[1]
        result = self._validate_detection_completion()
        self._detection_validation_cache = (state, result)
        return result

    def _validate_detection_completion(self) -> bool:
        """Perform the uncached detection-marker validation."""

        marker = self.paths.detection_completion_file
        processed = self.paths.checkpoint_dir / "processed_data.pkl"
        if (
            self.paths.detection_incomplete_file.exists()
            or not marker.is_file()
            or not processed.is_file()
            or not all(path.is_file() for path in self.detection_files)
        ):
            return False
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        selected_digest = payload.get("selected_products_sha256")
        if not (
            payload.get("status") == "complete"
            and payload.get("schema_version") == 1
            and payload.get("algorithm_version") == DETECTION_ALGORITHM_VERSION
            and payload.get("monte_carlo_samples") == self.options.monte_carlo_samples
            and payload.get("random_seed") == self.options.random_seed
            and isinstance(payload.get("task_count"), int)
            and payload.get("task_count") > 0
            and isinstance(selected_digest, str)
            and len(selected_digest) == 64
        ):
            return False
        try:
            if payload.get("processed_data_sha256") != self._sha256_cached(processed):
                return False
            expected_prepared_hashes = {
                path.name: self._sha256_cached(path) for path in self.prepared_files
            }
            if payload.get("prepared_sha256") != expected_prepared_hashes:
                return False
            expected_output_hashes = {
                path.name: self._sha256_cached(path) for path in self.detection_files
            }
            if payload.get("output_sha256") != expected_output_hashes:
                return False
            acquisition_digest = payload.get("acquisition_marker_sha256")
            if acquisition_digest is not None:
                acquisition_marker = self.paths.acquisition_completion_file
                if not acquisition_marker.is_file() or acquisition_digest != self._sha256_cached(
                    acquisition_marker
                ):
                    return False
        except OSError:
            return False
        return True

    def _detection_outputs_are_usable(self) -> bool:
        """Accept a complete run or an unmarked externally supplied set."""

        if not all(path.is_file() for path in self.detection_files):
            return False
        if self.paths.detection_incomplete_file.exists():
            return False
        if self.paths.detection_completion_file.exists():
            return self._detection_is_complete()
        return (
            self.options.monte_carlo_samples == DEFAULT_MONTE_CARLO_SAMPLES
            and self.options.random_seed == DEFAULT_RANDOM_SEED
        )

    def _catalog_outputs_are_usable(self) -> bool:
        from .catalog import catalog_outputs_are_usable

        state = self._path_state(
            [
                *self.catalog_files,
                *self.detection_files,
                self.paths.catalog_completion_file,
                self.paths.catalog_incomplete_file,
            ]
        )
        if (
            self._catalog_validation_cache is not None
            and self._catalog_validation_cache[0] == state
        ):
            return self._catalog_validation_cache[1]
        result = catalog_outputs_are_usable(
            self.paths.checkpoint_dir,
            hash_file=self._sha256_cached,
        )
        self._catalog_validation_cache = (state, result)
        return result

    @staticmethod
    def _path_state(
        paths: list[Path],
    ) -> tuple[
        tuple[
            str,
            int | None,
            int | None,
            int | None,
            int | None,
            int | None,
        ],
        ...,
    ]:
        """Return a cheap cache key that changes when any stage file changes."""

        state = []
        for path in paths:
            try:
                metadata = path.stat()
            except OSError:
                state.append((str(path), None, None, None, None, None))
            else:
                state.append((str(path), *ReproductionPipeline._stat_identity(metadata)))
        return tuple(state)

    @staticmethod
    def _stat_identity(
        metadata: object,
    ) -> tuple[int, int, int | None, int | None, int | None]:
        """Return metadata that identifies a concrete filesystem revision."""

        return (
            int(getattr(metadata, "st_size")),
            int(getattr(metadata, "st_mtime_ns")),
            getattr(metadata, "st_ctime_ns", None),
            getattr(metadata, "st_ino", None),
            getattr(metadata, "st_dev", None),
        )

    def _sha256_cached(self, path: Path) -> str:
        """Hash one stable filesystem revision once in this pipeline."""

        resolved = Path(path).resolve()
        before = resolved.stat()
        state = self._stat_identity(before)
        cached = self._file_hash_cache.get(resolved)
        if cached is not None and cached[0] == state:
            return cached[1]

        digest = sha256_file(resolved)
        after = resolved.stat()
        if self._stat_identity(after) != state:
            raise OSError(f"File changed while its checksum was calculated: {resolved}")
        self._file_hash_cache[resolved] = (state, digest)
        return digest

    def acquire(self) -> None:
        LOGGER.info("Starting acquisition stage")
        self.paths.create_runtime_directories()
        self._download_omni_if_needed(force=self.options.force)
        if not self.options.force:
            if self._require_consistent_acquisition_marker():
                LOGGER.info("Acquisition stage is already complete")
                return
        self._invalidate_downstream_for_acquisition_reset()
        summary = self._acquire_goes_if_needed(force=self.options.force)
        LOGGER.info("Acquisition stage complete: %s", summary)

    def _acquire_goes_if_needed(self, *, force: bool) -> dict[str, int]:
        from .acquisition import run_acquisition

        return run_acquisition(
            self.paths.checkpoint_dir,
            STUDY_START,
            STUDY_END,
            force=force,
            hash_file=self._sha256_cached,
        )

    def _invalidate_downstream_for_acquisition_reset(self) -> None:
        """Make old detection and catalog outputs unusable before a force reset."""

        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._invalidate_catalog_for_detection_reset(reason="upstream acquisition reset")
        marker = {
            "schema_version": 1,
            "status": "incomplete",
            "reason": "upstream acquisition reset",
        }
        atomic_write_json(marker, self.paths.detection_incomplete_file)
        self.paths.detection_completion_file.unlink(missing_ok=True)

    def _invalidate_catalog_for_detection_reset(self, *, reason: str) -> None:
        """Block catalog and figure reuse before their detection input changes."""

        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            {
                "schema_version": 1,
                "status": "incomplete",
                "reason": reason,
            },
            self.paths.catalog_incomplete_file,
        )
        self.paths.catalog_completion_file.unlink(missing_ok=True)
        self._output_manifest_path().unlink(missing_ok=True)

    def detect(self) -> None:
        if not self.options.force and self._detection_outputs_are_usable():
            LOGGER.info("Detection checkpoint set is already usable")
            return
        from .detection import run_detection

        require_files(
            [self.paths.checkpoint_dir / "processed_data.pkl", *self.prepared_files],
            purpose="Pc5 detection",
        )
        self._invalidate_catalog_for_detection_reset(reason="upstream detection reset")
        LOGGER.info("Starting Pc5 detection stage")
        summary = run_detection(
            self.paths.checkpoint_dir,
            self.paths.temporary_dir,
            workers=self.options.workers,
            force=self.options.force,
            monte_carlo_samples=self.options.monte_carlo_samples,
            random_seed=self.options.random_seed,
            hash_file=self._sha256_cached,
        )
        LOGGER.info("Pc5 detection stage complete: %s", summary)

    def catalog(self) -> None:
        from .catalog import build_event_catalogs

        if not self.options.force and self._catalog_outputs_are_usable():
            LOGGER.info("Event-catalog checkpoint set is already usable")
            return
        if not self._detection_outputs_are_usable():
            raise RuntimeError(
                "Detection outputs are absent, incomplete, or do not match the "
                "requested detection parameters. Resume the detection stage first."
            )
        require_files(
            self.detection_files,
            purpose="up-to-three-peak event catalog construction",
        )
        self._output_manifest_path().unlink(missing_ok=True)
        LOGGER.info("Building up-to-three-peak event catalogs")
        build_event_catalogs(
            self.paths.checkpoint_dir,
            force=self.options.force,
            hash_file=self._sha256_cached,
        )
        LOGGER.info("Event-catalog stage complete")

    def figure(
        self,
        number: int,
        *,
        output_dir: Path | None = None,
        table_dir: Path | None = None,
    ) -> None:
        from .figures import (
            plot_figure_1,
            plot_figure_2,
            plot_figure_3,
            plot_figure_4,
        )

        if number not in range(1, 5):
            raise ValueError("Figure number must be 1, 2, 3, or 4")
        if not self._catalog_outputs_are_usable():
            raise RuntimeError(
                "Event catalogs are absent or an interrupted catalog stage must "
                "be resumed before plotting."
            )
        require_files(self.catalog_files, purpose=f"Figure {number}")
        canonical_outputs = output_dir is None and table_dir is None
        figures = output_dir or self.paths.figures_dir
        tables = table_dir or self.paths.tables_dir
        figures.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        destination = figures / FIGURE_NAMES[number - 1]
        calls: dict[int, Callable[[], object]] = {
            1: lambda: plot_figure_1(self.paths.checkpoint_dir, destination),
            2: lambda: plot_figure_2(self.paths.checkpoint_dir, destination),
            3: lambda: plot_figure_3(self.paths.checkpoint_dir, self.paths.omni_file, destination),
            4: lambda: plot_figure_4(
                self.paths.checkpoint_dir,
                self.paths.omni_file,
                self.paths.observation_counts_file,
                destination,
                tables / CORRELATIONS_TABLE_NAME,
            ),
        }
        if number in (3, 4):
            require_files([self.paths.omni_file], purpose=f"Figure {number}")
        if canonical_outputs:
            # A complete-set manifest cannot describe a partly regenerated set.
            # Invalidate it immediately before the canonical overwrite begins.
            self._output_manifest_path().unlink(missing_ok=True)
        calls[number]()

    def figures(self) -> None:
        """Render all figures, then promote the complete set together."""

        LOGGER.info("Rendering Figures 1-4")
        self.paths.figures_dir.parent.mkdir(parents=True, exist_ok=True)
        self.paths.tables_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(prefix="pc5-paper-output-"))
        staging_figures = staging_root / "figures"
        staging_tables = staging_root / "tables"
        try:
            for number in range(1, 5):
                self.figure(number, output_dir=staging_figures, table_dir=staging_tables)
            self.paths.figures_dir.mkdir(parents=True, exist_ok=True)
            self.paths.tables_dir.mkdir(parents=True, exist_ok=True)
            # Staging leaves the prior outputs valid, so their manifest remains
            # valid throughout rendering. Promotion starts a new output set.
            self._output_manifest_path().unlink(missing_ok=True)
            for source in sorted(staging_figures.iterdir()):
                destination = self.paths.figures_dir / source.name
                with atomic_destination(destination, suffix=".promote.part") as temporary:
                    shutil.copyfile(source, temporary)
            for source in sorted(staging_tables.iterdir()):
                destination = self.paths.tables_dir / source.name
                with atomic_destination(destination, suffix=".promote.part") as temporary:
                    shutil.copyfile(source, temporary)
            self._write_output_manifest()
            LOGGER.info("Figure stage complete")
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def all(self, *, full: bool) -> None:
        """Resume the complete chain and always regenerate the paper outputs.

        ``full=False`` requires an existing catalog or detection checkpoint
        before proceeding. ``full=True`` authorizes the resource-intensive
        full-interval stages while still resuming completed work unless
        ``force`` is set.
        """

        if self.options.force:
            if not full:
                raise ValueError("A forced full-chain run requires full=True")
            # A full forced run discards upstream state, so do not hash or
            # deserialize the checkpoints that it is explicitly replacing.
            self.paths.create_runtime_directories()
            self._download_omni_if_needed(force=True)
            self._invalidate_downstream_for_acquisition_reset()
            self._acquire_goes_if_needed(force=True)
            self.detect()
            self.catalog()
            self.figures()
            return

        catalog_is_usable = self._catalog_outputs_are_usable()
        detection_is_usable = False if catalog_is_usable else self._detection_outputs_are_usable()
        if not full and not catalog_is_usable and not detection_is_usable:
            # Validate the controlling GOES prerequisite before creating
            # directories or downloading the smaller auxiliary OMNI input.
            raise RuntimeError(
                "Event catalogs and detection checkpoints are absent. "
                "Supply either checkpoint set with --checkpoint-dir, "
                "or allow the resource-intensive full-interval stages with --full."
            )

        self.paths.create_runtime_directories()

        # OMNI is the public auxiliary input for Figures 3 and 4. Acquisition
        # remains controlled by the full-interval GOES prerequisite above.
        self._download_omni_if_needed(force=False)

        # Resume from the deepest complete causal checkpoint.  A catalog is a
        # sufficient input to every figure; its absent ancestors need not be
        # recomputed merely to prove that they once existed.
        if not catalog_is_usable:
            if not detection_is_usable:
                if not self._require_consistent_acquisition_marker():
                    self._acquire_goes_if_needed(force=False)
                self.detect()
            self.catalog()
        self.figures()

    def _download_omni_if_needed(self, *, force: bool) -> None:
        if self.paths.omni_file.exists() and not force:
            self._validate_omni_file(self.paths.omni_file)
            return
        previous_digest = (
            self._sha256_cached(self.paths.omni_file) if self.paths.omni_file.is_file() else None
        )
        self.paths.omni_file.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            OMNI2_URL,
            headers={"User-Agent": "pc5-climatology-reproduction/1.0"},
        )
        with atomic_destination(self.paths.omni_file, suffix=".download.part") as temporary:
            with (
                urllib.request.urlopen(request, timeout=300) as response,
                temporary.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            self._validate_omni_file(temporary)
            replacement_digest = sha256_file(temporary)
            if replacement_digest != previous_digest:
                # Remove the complete-output claim before the changed input is
                # promoted, so interruption cannot leave a stale manifest.
                self._output_manifest_path().unlink(missing_ok=True)
        resolved = self.paths.omni_file.resolve()
        self._file_hash_cache[resolved] = (
            self._stat_identity(resolved.stat()),
            replacement_digest,
        )

    @staticmethod
    def _validate_omni_file(path: Path) -> None:
        """Reject truncated or structurally invalid OMNI2 inputs."""

        from .statistics import read_omni_hourly, validate_omni_study_grid

        omni = read_omni_hourly(path)
        validate_omni_study_grid(omni, source=path)

    def _output_manifest_path(self) -> Path:
        """Return the manifest location for the configured output roots."""

        if self.paths.figures_dir.parent == self.paths.tables_dir.parent:
            return self.paths.figures_dir.parent / "manifest.json"
        # With unrelated custom roots, the table root owns the shared manifest.
        return self.paths.tables_dir / "manifest.json"

    def _write_output_manifest(self) -> None:
        artifacts = [*self.figure_files]
        correlations = self.paths.tables_dir / CORRELATIONS_TABLE_NAME
        if correlations.exists():
            artifacts.append(correlations)
        destination = self._output_manifest_path()
        inputs = []
        if self.paths.omni_file.is_file():
            omni_record = artifact_record(self.paths.omni_file, self.paths.root)
            # OMNI is an external, ignored runtime input even when downloaded
            # beneath the checkout's default data directory.
            omni_record["external"] = True
            inputs.append(
                {
                    **omni_record,
                    "source_url": OMNI2_URL,
                }
            )
        payload = {
            "study_interval": {"start": STUDY_START.isoformat(), "end": STUDY_END.isoformat()},
            "inputs": inputs,
            "artifacts": [artifact_record(path, destination.parent) for path in artifacts],
        }
        parameter_file = self.paths.root / "configs" / "paper.toml"
        if parameter_file.is_file():
            payload["parameter_summary"] = artifact_record(parameter_file, self.paths.root)
        atomic_write_json(payload, destination)

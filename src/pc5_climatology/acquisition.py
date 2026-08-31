"""Download and preprocess the GOES high-resolution magnetometer archive.

Calling :func:`run_acquisition` is the execution boundary for network work and
optional-dependency imports.

The stage writes the documented Joblib checkpoint names and daily DataFrame
schema. Existing satellite lists and the failure list are loaded before new
observations are appended, and every local checkpoint replacement is atomic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

from .config import (
    CADENCE_SECONDS,
    MAGNETIC_FIELD_ABSOLUTE_LIMIT_NT,
    ORBIT_IQR_MULTIPLIER,
    STUDY_END,
    STUDY_START,
)
from .coordinates import (
    j2000_seconds_to_datetime,
    llr_to_mlt,
    mean_field_aligned_components,
)
from .io import sha256_file

LOGGER = logging.getLogger(__name__)

MAGNETIC_MIN_NT = -MAGNETIC_FIELD_ABSOLUTE_LIMIT_NT
MAGNETIC_MAX_NT = MAGNETIC_FIELD_ABSOLUTE_LIMIT_NT
RESAMPLE_INTERVAL = f"{CADENCE_SECONDS // 60}min"

_DATE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")


@dataclass(frozen=True)
class SatelliteSource:
    """One NOAA archive root and its output checkpoint identifier."""

    number: int
    base_url: str
    group: str

    @property
    def code(self) -> str:
        """Return the zero-padded two-digit satellite code."""

        return f"{self.number:02d}"


SATELLITE_SOURCES = (
    *(
        SatelliteSource(
            number,
            "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/"
            f"access/science/mag/goes{number:02d}/magn-l2-hires/",
            "group1",
        )
        for number in range(8, 16)
    ),
    *(
        SatelliteSource(
            number,
            "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/"
            f"goes/goes{number:02d}/l2/data/magn-l2-hires/",
            "group2",
        )
        for number in range(16, 19)
    ),
)


DOWNLOAD_CONFIG: dict[str, dict[str, int]] = {
    "group1": {
        "max_concurrent_downloads": 5,
        "retry_attempts": 3,
        "timeout_seconds": 300,
        "chunk_size": 1024 * 1024,
        "tcp_limit": 10,
    },
    "group2": {
        "max_concurrent_downloads": 3,
        "retry_attempts": 5,
        "timeout_seconds": 500,
        "chunk_size": 2 * 1024 * 1024,
        "tcp_limit": 5,
    },
}


def _aiohttp() -> Any:
    """Import aiohttp with its Python HTTP parser selected for Python 3.9."""

    # Select the pure-Python HTTP parser required by the pinned acquisition
    # environment before aiohttp initializes its parser implementation.
    os.environ["AIOHTTP_NO_EXTENSIONS"] = "1"
    try:
        import aiohttp
        from aiohttp.helpers import NO_EXTENSIONS
    except ImportError as exc:  # pragma: no cover - optional full-stage extra
        raise RuntimeError(
            "Full GOES acquisition requires the optional 'aiohttp' dependency"
        ) from exc
    if not NO_EXTENSIONS:
        raise RuntimeError(
            "aiohttp was imported before the acquisition stage selected its "
            "Python HTTP parser. Start acquisition in a fresh Python process."
        )
    return aiohttp


def _joblib() -> Any:
    """Import Joblib only when a checkpoint is read or written."""

    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - installation error
        raise RuntimeError("GOES acquisition requires the 'joblib' package") from exc
    return joblib


def _atomic_joblib_dump(value: Any, path: Path, *, compress: int = 3) -> None:
    """Atomically replace a local Joblib checkpoint."""

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


def _atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace a CSV file in ``path.parent``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(value: Any, path: Path) -> None:
    """Atomically replace a small acquisition transaction journal."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_list(path: Path) -> list[Any]:
    """Load a list checkpoint, returning an empty list if absent."""

    if not path.exists():
        return []
    value = _joblib().load(path)
    if not isinstance(value, list):
        raise TypeError(f"Expected a list checkpoint at {path}")
    return value


def _validate_study_dates(start_date: date, end_date: date) -> None:
    """Validate the inclusive paper interval requested by the caller."""

    if isinstance(start_date, datetime) or isinstance(end_date, datetime):
        raise TypeError("start_date and end_date must be datetime.date values")
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise TypeError("start_date and end_date must be datetime.date values")
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    if start_date < STUDY_START or end_date > STUDY_END:
        raise ValueError(
            "Requested dates must lie within the inclusive paper interval "
            f"{STUDY_START.isoformat()}..{STUDY_END.isoformat()}"
        )


def _date_from_url(url: str) -> date:
    """Extract the first isolated YYYYMMDD token from a NOAA URL."""

    match = _DATE_PATTERN.search(url)
    if match is None:
        raise ValueError(f"No YYYYMMDD date found in NOAA URL: {url}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def _satellite_number_from_url(url: str) -> int:
    """Extract a GOES number from a product filename or URL path."""

    lowered = url.lower()
    filename = Path(urlparse(lowered).path).name
    filename_match = re.search(r"(?:^|_)g0?(\d{1,2})(?:_|\.)", filename)
    if filename_match:
        return int(filename_match.group(1))

    for segment in urlparse(lowered).path.split("/"):
        match = re.fullmatch(r"goes0?(\d{1,2})", segment)
        if match:
            return int(match.group(1))
    raise ValueError(f"No GOES satellite identifier found in URL: {url}")


def _dataframe_date(frame: pd.DataFrame) -> date:
    """Return the observation date encoded by a nonempty daily DataFrame."""

    if not isinstance(frame, pd.DataFrame) or frame.empty or "time" not in frame:
        raise ValueError("Daily GOES checkpoints must be nonempty DataFrames with time")
    timestamps = pd.to_datetime(frame["time"], errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError("Daily GOES checkpoint contains no valid timestamps")
    dates = {stamp.date() for stamp in timestamps}
    if len(dates) != 1:
        raise ValueError("A daily GOES checkpoint spans more than one UTC date")
    return dates.pop()


def _observation_counts(satellite_frames: dict[int, list[pd.DataFrame]]) -> pd.DataFrame:
    """Count one observation per daily satellite DataFrame, as in Figure 4."""

    years: list[int] = []
    for frames in satellite_frames.values():
        for frame in frames:
            observation_date = _dataframe_date(frame)
            if STUDY_START <= observation_date <= STUDY_END:
                years.append(observation_date.year)

    if not years:
        return pd.DataFrame(
            {
                "year": pd.Series(dtype="int64"),
                "observation_count": pd.Series(dtype="int64"),
            }
        )
    counts = pd.Series(years, name="year").value_counts().sort_index()
    return counts.rename("observation_count").rename_axis("year").reset_index()


def _html_links(payload: bytes, pattern: re.Pattern[str]) -> list[str]:
    """Return naturally sortable href values that fully match ``pattern``."""

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - optional full-stage extra
        raise RuntimeError(
            "Full GOES acquisition requires the optional 'beautifulsoup4' dependency"
        ) from exc
    soup = BeautifulSoup(payload, "html.parser")
    links = {
        href
        for node in soup.find_all("a")
        if (href := node.get("href")) is not None and pattern.fullmatch(href)
    }
    return sorted(
        links,
        key=lambda value: [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)
        ],
    )


async def _fetch_bytes(session: Any, url: str, settings: dict[str, int]) -> bytes:
    """Fetch one URL with the configured retry, timeout, and chunk sizes."""

    aiohttp = _aiohttp()

    last_error: BaseException | None = None
    for attempt in range(settings["retry_attempts"]):
        try:
            timeout = aiohttp.ClientTimeout(total=settings["timeout_seconds"])
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    payload = bytearray()
                    async for chunk in response.content.iter_chunked(settings["chunk_size"]):
                        payload.extend(chunk)
                    return bytes(payload)
                last_error = RuntimeError(f"HTTP {response.status} for {url}")
                if response.status == 429:
                    await asyncio.sleep(min(60, 10 * 2**attempt))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < settings["retry_attempts"]:
                await asyncio.sleep(5 * (attempt + 1))

    raise RuntimeError(
        f"Failed to fetch {url} after {settings['retry_attempts']} attempts"
    ) from last_error


async def _discover_satellite_urls(
    session: Any,
    source: SatelliteSource,
    start_date: date,
    end_date: date,
    settings: dict[str, int],
) -> list[str]:
    """Discover product URLs for one satellite within the inclusive interval."""

    root = await _fetch_bytes(session, source.base_url, settings)
    year_links = _html_links(root, re.compile(r"\d{4}/"))
    urls: list[str] = []
    for year_link in year_links:
        year = int(year_link.rstrip("/"))
        if year < start_date.year or year > end_date.year:
            continue
        year_url = urljoin(source.base_url, year_link)
        year_payload = await _fetch_bytes(session, year_url, settings)
        month_links = _html_links(year_payload, re.compile(r"\d{2}/"))
        for month_link in month_links:
            month = int(month_link.rstrip("/"))
            if date(year, month, 1) > end_date:
                continue
            if year == start_date.year and month < start_date.month:
                continue
            month_url = urljoin(year_url, month_link)
            month_payload = await _fetch_bytes(session, month_url, settings)
            # Match only the science product identity consumed by this study.
            # Other NetCDF files in an archive directory are not silently
            # treated as failed magnetometer products.
            file_links = _html_links(
                month_payload,
                re.compile(
                    r"dn_magn-l2-hires_g0?\d{1,2}_d\d{8}_v"
                    r"\d+[_-]\d+[_-]\d+\.nc"
                ),
            )
            for file_link in file_links:
                product_url = urljoin(month_url, file_link)
                if _satellite_number_from_url(product_url) != source.number:
                    raise RuntimeError(
                        "Archive product satellite does not match its source "
                        f"directory: {product_url}"
                    )
                product_date = _date_from_url(product_url)
                if start_date <= product_date <= end_date:
                    urls.append(product_url)
    return urls


def _replace_fill_values(values: np.ndarray, fill_value: Any) -> np.ndarray:
    """Replace an exact NetCDF fill value with NaN in a float copy."""

    result = np.asarray(values, dtype=float).copy()
    if fill_value is not None:
        result[result == float(fill_value)] = np.nan
    return result


def _decode_daily_netcdf(payload: bytes, product_url: str) -> pd.DataFrame:
    """Decode and transform one daily NOAA NetCDF payload.

    The operation applies the E-component sign flip, 1.5-IQR orbit
    rejection, strict outside-±1024 nT rejection, independent one-minute
    resampling, positional MLT assignment, and documented checkpoint column order.
    Independent resampling can yield unequal magnetic/orbit lengths; that is
    treated as a failed product rather than silently time-aligning a result that
    would differ from the specified calculation.
    """

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - optional full-stage extra
        raise RuntimeError(
            "Full GOES acquisition requires the optional 'netCDF4' dependency"
        ) from exc

    product_date = _date_from_url(product_url)
    with Dataset("in-memory-goes.nc", memory=payload) as dataset:
        magnetic_variable = dataset["b_epn"]
        orbit_variable = dataset["orbit_llr_geo"]
        magnetic_values = np.asarray(magnetic_variable[:].data)
        orbit_values = np.asarray(orbit_variable[:].data)

        magnetic = pd.DataFrame(
            {
                "time": j2000_seconds_to_datetime(dataset["time"][:].data),
                "E": _replace_fill_values(
                    magnetic_values[:, 0], getattr(magnetic_variable, "_FillValue", None)
                ),
                "P": _replace_fill_values(
                    magnetic_values[:, 1], getattr(magnetic_variable, "_FillValue", None)
                ),
                "N": _replace_fill_values(
                    magnetic_values[:, 2], getattr(magnetic_variable, "_FillValue", None)
                ),
            }
        )
        orbit = pd.DataFrame(
            {
                "time": j2000_seconds_to_datetime(dataset["time_orbit"][:].data),
                "lat": _replace_fill_values(
                    orbit_values[:, 0], getattr(orbit_variable, "_FillValue", None)
                ),
                "long": _replace_fill_values(
                    orbit_values[:, 1], getattr(orbit_variable, "_FillValue", None)
                ),
                "radius": _replace_fill_values(
                    orbit_values[:, 2], getattr(orbit_variable, "_FillValue", None)
                ),
            }
        )

    magnetic["E"] *= -1.0

    orbit_columns = ["lat", "long", "radius"]
    first_quartile = orbit[orbit_columns].quantile(0.25)
    third_quartile = orbit[orbit_columns].quantile(0.75)
    iqr = third_quartile - first_quartile
    inlier = ~(
        (orbit[orbit_columns] < first_quartile - ORBIT_IQR_MULTIPLIER * iqr)
        | (orbit[orbit_columns] > third_quartile + ORBIT_IQR_MULTIPLIER * iqr)
    ).any(axis=1)
    orbit = orbit.loc[inlier]

    magnetic["time"] = pd.to_datetime(magnetic["time"])
    orbit["time"] = pd.to_datetime(orbit["time"])
    magnetic = magnetic.loc[magnetic["time"].dt.date == product_date]
    orbit = orbit.loc[orbit["time"].dt.date == product_date]
    magnetic = magnetic.sort_values("time").reset_index(drop=True)
    orbit = orbit.sort_values("time").reset_index(drop=True)
    if magnetic.empty or orbit.empty:
        raise ValueError(f"No same-day magnetic/orbit data in {product_url}")

    field_columns = ["E", "P", "N"]
    magnetic[field_columns] = magnetic[field_columns].mask(
        magnetic[field_columns] < MAGNETIC_MIN_NT
    )
    magnetic[field_columns] = magnetic[field_columns].mask(
        magnetic[field_columns] > MAGNETIC_MAX_NT
    )

    magnetic = magnetic.set_index("time").resample(RESAMPLE_INTERVAL).mean().reset_index()
    orbit = orbit.set_index("time").resample(RESAMPLE_INTERVAL).mean().reset_index()

    mfac = mean_field_aligned_components(magnetic[["time", "E", "N", "P"]])
    orbit_times = pd.to_datetime(orbit["time"]).dt.to_pydatetime()
    mlt = llr_to_mlt(orbit[orbit_columns].to_numpy(), orbit_times)
    if len(mfac) != len(mlt):
        raise ValueError(
            "Independent minute resampling produced unequal magnetic and orbit "
            f"lengths for {product_url}: {len(mfac)} != {len(mlt)}"
        )
    mfac["mlt"] = mlt
    mfac = mfac[["time", "mlt", "b_parallel", "b_azimuthal", "b_radial"]]
    prepared = mfac.merge(magnetic, how="left", on="time")
    prepared.attrs["source_url"] = product_url
    return prepared


async def _download_and_decode(
    session: Any,
    url: str,
    settings: dict[str, int],
    semaphore: asyncio.Semaphore,
) -> tuple[str, pd.DataFrame]:
    """Download and synchronously decode one product under a group semaphore."""

    async with semaphore:
        payload = await _fetch_bytes(session, url, settings)
        return url, _decode_daily_netcdf(payload, url)


async def _capture_product_result(
    session: Any,
    url: str,
    settings: dict[str, int],
    semaphore: asyncio.Semaphore,
) -> tuple[str, pd.DataFrame | None, BaseException | None]:
    """Keep a product URL attached to either its result or its exception."""

    try:
        _, frame = await _download_and_decode(session, url, settings, semaphore)
        return url, frame, None
    except Exception as exc:  # returned to the serial checkpoint writer
        return url, None, exc


def _remove_force_interval(
    processed_urls: list[str],
    satellite_frames: dict[int, list[pd.DataFrame]],
    start_date: date,
    end_date: date,
) -> tuple[list[str], dict[int, list[pd.DataFrame]]]:
    """Remove an explicitly forced interval before it is recomputed."""

    retained_urls: list[str] = []
    for url in processed_urls:
        try:
            product_date = _date_from_url(url)
        except ValueError:
            retained_urls.append(url)
            continue
        if not start_date <= product_date <= end_date:
            retained_urls.append(url)

    retained_frames: dict[int, list[pd.DataFrame]] = {}
    for satellite, frames in satellite_frames.items():
        retained_frames[satellite] = [
            frame for frame in frames if not start_date <= _dataframe_date(frame) <= end_date
        ]
    return retained_urls, retained_frames


def _identity_multiplicities(
    processed_urls: list[str],
    satellite_frames: dict[int, list[pd.DataFrame]],
) -> tuple[Counter[tuple[int, date]], Counter[tuple[int, date]]]:
    """Count URL and frame identities without assuming a shared list order."""

    url_counts = Counter(
        (_satellite_number_from_url(url), _date_from_url(url)) for url in processed_urls
    )
    frame_counts = Counter(
        (satellite, _dataframe_date(frame))
        for satellite, frames in satellite_frames.items()
        for frame in frames
    )
    return url_counts, frame_counts


def _validate_checkpoint_identity(
    processed_urls: list[str],
    satellite_frames: dict[int, list[pd.DataFrame]],
) -> None:
    """Validate the global study interval and one-to-one stored identities."""

    duplicate_urls = sorted(url for url, count in Counter(processed_urls).items() if count > 1)
    if duplicate_urls:
        raise RuntimeError(
            f"Acquisition checkpoint repeats exact processed URLs: {duplicate_urls[:10]}"
        )
    url_counts, frame_counts = _identity_multiplicities(processed_urls, satellite_frames)
    outside_study = sorted(
        key for key in set(url_counts) | set(frame_counts) if not STUDY_START <= key[1] <= STUDY_END
    )
    if outside_study:
        preview = [
            {"satellite": satellite, "date": observation_date.isoformat()}
            for satellite, observation_date in outside_study[:10]
        ]
        raise RuntimeError(
            "Acquisition checkpoints contain product dates outside the inclusive "
            f"paper interval {STUDY_START.isoformat()}..{STUDY_END.isoformat()}: "
            f"{preview}"
        )
    if url_counts != frame_counts:
        differing = sorted(
            key
            for key in set(url_counts) | set(frame_counts)
            if url_counts[key] != frame_counts[key]
        )
        preview = [
            {
                "satellite": key[0],
                "date": key[1].isoformat(),
                "urls": url_counts[key],
                "frames": frame_counts[key],
            }
            for key in differing[:10]
        ]
        raise RuntimeError(
            f"Acquisition checkpoints have unequal URL/DataFrame identity multiplicities: {preview}"
        )


def _persist_force_reset(
    checkpoint_dir: Path,
    processed_urls: list[str],
    failed_urls: list[str],
    satellite_frames: dict[int, list[pd.DataFrame]],
) -> None:
    """Publish every checkpoint file belonging to an idempotent forced reset."""

    for source in SATELLITE_SOURCES:
        _atomic_joblib_dump(
            satellite_frames[source.number],
            checkpoint_dir / f"df_{source.code}.pkl",
        )
    _atomic_joblib_dump(processed_urls, checkpoint_dir / "processed_data.pkl")
    _atomic_joblib_dump(failed_urls, checkpoint_dir / "faulty_data.pkl")


def _apply_force_reset(
    processed_urls: list[str],
    failed_urls: list[str],
    satellite_frames: dict[int, list[pd.DataFrame]],
    *,
    start_date: date,
    end_date: date,
    clear_all: bool,
) -> tuple[list[str], list[str], dict[int, list[pd.DataFrame]]]:
    """Return the deterministic state requested by a forced recomputation."""

    if clear_all:
        return (
            [],
            [],
            {source.number: [] for source in SATELLITE_SOURCES},
        )
    retained_urls, retained_frames = _remove_force_interval(
        processed_urls, satellite_frames, start_date, end_date
    )
    retained_failures = [
        url
        for url in failed_urls
        if not (_DATE_PATTERN.search(url) and start_date <= _date_from_url(url) <= end_date)
    ]
    return retained_urls, retained_failures, retained_frames


def _recover_transaction(
    checkpoint_dir: Path,
    transaction_path: Path,
    processed_urls: list[str],
    failed_urls: list[str],
    satellite_frames: dict[int, list[pd.DataFrame]],
) -> tuple[list[str], list[str], dict[int, list[pd.DataFrame]]]:
    """Finish or roll forward the one interrupted cross-file transaction."""

    if not transaction_path.exists():
        return processed_urls, failed_urls, satellite_frames
    try:
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Unreadable acquisition transaction journal: {transaction_path}"
        ) from exc

    operation = transaction.get("operation")
    if operation == "force_reset":
        start_date = date.fromisoformat(str(transaction["start_date"]))
        end_date = date.fromisoformat(str(transaction["end_date"]))
        processed_urls, failed_urls, satellite_frames = _apply_force_reset(
            processed_urls,
            failed_urls,
            satellite_frames,
            start_date=start_date,
            end_date=end_date,
            clear_all=bool(transaction.get("clear_all", False)),
        )
        _persist_force_reset(checkpoint_dir, processed_urls, failed_urls, satellite_frames)
        transaction_path.unlink(missing_ok=True)
        return processed_urls, failed_urls, satellite_frames

    if operation != "product" or not isinstance(transaction.get("url"), str):
        raise RuntimeError(f"Unsupported acquisition transaction journal: {transaction_path}")

    url = str(transaction["url"])
    url_counts, frame_counts = _identity_multiplicities(processed_urls, satellite_frames)
    key = (_satellite_number_from_url(url), _date_from_url(url))
    if url in processed_urls:
        if url_counts != frame_counts:
            raise RuntimeError(
                "Interrupted acquisition marked a URL complete, but its "
                "URL/DataFrame multiplicities do not agree"
            )
    else:
        differing = {
            candidate
            for candidate in set(url_counts) | set(frame_counts)
            if url_counts[candidate] != frame_counts[candidate]
        }
        if not differing:
            # The interruption preceded publication of the daily DataFrame.
            transaction_path.unlink(missing_ok=True)
            return processed_urls, failed_urls, satellite_frames
        if differing != {key} or frame_counts[key] != url_counts[key] + 1:
            raise RuntimeError(
                "Interrupted acquisition cannot identify one unambiguous "
                f"orphan DataFrame for {url}"
            )
        # The DataFrame replacement completed and only its processed marker was
        # interrupted.  Roll the marker forward rather than downloading and
        # appending the same day a second time.
        processed_urls.append(url)
        _atomic_joblib_dump(processed_urls, checkpoint_dir / "processed_data.pkl")

    if url in failed_urls:
        failed_urls = [failed_url for failed_url in failed_urls if failed_url != url]
        _atomic_joblib_dump(failed_urls, checkpoint_dir / "faulty_data.pkl")
    transaction_path.unlink(missing_ok=True)
    return processed_urls, failed_urls, satellite_frames


def _initial_acquisition_state(
    checkpoint_dir: Path,
    *,
    clear_all: bool,
) -> tuple[list[str], list[str], dict[int, list[pd.DataFrame]]]:
    """Load resumable state, or start clean without reading discarded files."""

    transaction_path = checkpoint_dir / "acquisition_transaction.json"
    if clear_all:
        transaction_path.unlink(missing_ok=True)
        return [], [], {source.number: [] for source in SATELLITE_SOURCES}

    processed_urls = [str(value) for value in _load_list(checkpoint_dir / "processed_data.pkl")]
    failed_urls = [str(value) for value in _load_list(checkpoint_dir / "faulty_data.pkl")]
    satellite_frames: dict[int, list[pd.DataFrame]] = {
        source.number: _load_list(checkpoint_dir / f"df_{source.code}.pkl")
        for source in SATELLITE_SOURCES
    }
    return _recover_transaction(
        checkpoint_dir,
        transaction_path,
        processed_urls,
        failed_urls,
        satellite_frames,
    )


async def _run_acquisition_async(
    checkpoint_dir: Path,
    start_date: date,
    end_date: date,
    force: bool,
    hash_file: Callable[[Path], str],
) -> dict[str, int]:
    """Asynchronous implementation behind the public synchronous entry point."""

    aiohttp = _aiohttp()

    processed_path = checkpoint_dir / "processed_data.pkl"
    failure_path = checkpoint_dir / "faulty_data.pkl"
    transaction_path = checkpoint_dir / "acquisition_transaction.json"
    completion_path = checkpoint_dir / "acquisition_complete.json"
    clear_all = force and start_date == STUDY_START and end_date == STUDY_END

    # Invalidate the claim before touching stage state. A full forced run is a
    # recovery boundary: corrupted checkpoints being discarded are never read.
    completion_path.unlink(missing_ok=True)
    processed_urls, failed_urls, satellite_frames = _initial_acquisition_state(
        checkpoint_dir,
        clear_all=clear_all,
    )
    if not clear_all:
        _validate_checkpoint_identity(processed_urls, satellite_frames)

    if force:
        _atomic_json_dump(
            {
                "operation": "force_reset",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "clear_all": clear_all,
            },
            transaction_path,
        )
        processed_urls, failed_urls, satellite_frames = _apply_force_reset(
            processed_urls,
            failed_urls,
            satellite_frames,
            start_date=start_date,
            end_date=end_date,
            clear_all=clear_all,
        )
        _persist_force_reset(checkpoint_dir, processed_urls, failed_urls, satellite_frames)
        transaction_path.unlink(missing_ok=True)
        _validate_checkpoint_identity(processed_urls, satellite_frames)

    processed_set = set(processed_urls)
    attempted = 0
    completed = 0
    failed = 0

    for group in ("group1", "group2"):
        settings = DOWNLOAD_CONFIG[group]
        sources = [source for source in SATELLITE_SOURCES if source.group == group]
        connector = aiohttp.TCPConnector(limit=settings["tcp_limit"])
        headers = {"User-Agent": "pc5-climatology-reproduction/1.0"}
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            discoveries = await asyncio.gather(
                *(
                    _discover_satellite_urls(session, source, start_date, end_date, settings)
                    for source in sources
                ),
                return_exceptions=True,
            )
            discovered_urls: list[str] = []
            discovery_errors: list[str] = []
            for source, discovery in zip(sources, discoveries):
                if isinstance(discovery, BaseException):
                    discovery_errors.append(
                        f"GOES {source.code}: {type(discovery).__name__}: {discovery}"
                    )
                    continue
                if start_date == STUDY_START and end_date == STUDY_END and not discovery:
                    discovery_errors.append(
                        f"GOES {source.code}: no matching products were discovered"
                    )
                    continue
                discovered_urls.extend(discovery)
            if discovery_errors:
                raise RuntimeError(
                    "Archive discovery was incomplete; no complete acquisition "
                    "checkpoint can be declared:\n  - " + "\n  - ".join(discovery_errors)
                )
            pending_urls = [url for url in discovered_urls if url not in processed_set]
            LOGGER.info(
                "%s acquisition: %d products discovered, %d pending",
                group,
                len(discovered_urls),
                len(pending_urls),
            )
            attempted += len(pending_urls)
            semaphore = asyncio.Semaphore(settings["max_concurrent_downloads"])
            tasks = [
                asyncio.create_task(_capture_product_result(session, url, settings, semaphore))
                for url in pending_urls
            ]

            for task in asyncio.as_completed(tasks):
                url, daily_frame, product_error = await task
                if product_error is None and daily_frame is not None:
                    satellite_number = _satellite_number_from_url(url)
                    if satellite_number not in satellite_frames:
                        raise ValueError(f"Unsupported GOES satellite in {url}")

                    # The journal makes the two checkpoint replacements one
                    # recoverable transaction.  A restart can distinguish an
                    # unpublished frame from an already committed product.
                    _atomic_json_dump(
                        {"operation": "product", "url": url},
                        transaction_path,
                    )
                    satellite_frames[satellite_number].append(daily_frame)
                    source_code = f"{satellite_number:02d}"
                    _atomic_joblib_dump(
                        satellite_frames[satellite_number],
                        checkpoint_dir / f"df_{source_code}.pkl",
                    )
                    processed_urls.append(url)
                    processed_set.add(url)
                    _atomic_joblib_dump(processed_urls, processed_path)
                    if url in failed_urls:
                        failed_urls = [
                            failed_url for failed_url in failed_urls if failed_url != url
                        ]
                        _atomic_joblib_dump(failed_urls, failure_path)
                    transaction_path.unlink(missing_ok=True)
                    completed += 1
                    if (completed + failed) % 100 == 0 or completed + failed == attempted:
                        LOGGER.info(
                            "GOES preparation progress: %d completed, %d failed, %d attempted",
                            completed,
                            failed,
                            attempted,
                        )
                    continue

                if product_error is not None:
                    LOGGER.error(
                        "Failed to process %s: %s",
                        url,
                        product_error,
                        exc_info=(
                            type(product_error),
                            product_error,
                            product_error.__traceback__,
                        ),
                    )
                    if url not in failed_urls:
                        failed_urls.append(url)
                        _atomic_joblib_dump(failed_urls, failure_path)
                    failed += 1
                    if (completed + failed) % 100 == 0 or completed + failed == attempted:
                        LOGGER.info(
                            "GOES preparation progress: %d completed, %d failed, %d attempted",
                            completed,
                            failed,
                            attempted,
                        )

    counts = _observation_counts(satellite_frames)
    unresolved_failures: list[str] = []
    for url in failed_urls:
        try:
            inside_request = start_date <= _date_from_url(url) <= end_date
        except ValueError:
            inside_request = True
        if inside_request and url not in processed_set:
            unresolved_failures.append(url)
    if failed or unresolved_failures:
        raise RuntimeError(
            "GOES acquisition has "
            f"{len(unresolved_failures)} unresolved product failure(s). "
            f"See {failure_path}; successful products remain resumable."
        )
    if not processed_urls:
        raise RuntimeError("GOES acquisition produced no prepared products")
    if start_date == STUDY_START and end_date == STUDY_END:
        empty_satellites = [
            source.code for source in SATELLITE_SOURCES if not satellite_frames[source.number]
        ]
        if empty_satellites:
            raise RuntimeError(
                "GOES acquisition has no prepared products for satellite(s): "
                + ", ".join(empty_satellites)
            )
    _validate_checkpoint_identity(processed_urls, satellite_frames)
    prepared_paths = [checkpoint_dir / f"df_{source.code}.pkl" for source in SATELLITE_SOURCES]
    for source, prepared_path in zip(SATELLITE_SOURCES, prepared_paths):
        if not prepared_path.is_file():
            _atomic_joblib_dump(satellite_frames[source.number], prepared_path)
    counts_path = checkpoint_dir / "observation_counts_by_year.csv"
    _atomic_csv_dump(counts, counts_path)
    output_paths = [
        processed_path,
        *prepared_paths,
        counts_path,
    ]
    _atomic_json_dump(
        {
            "status": "complete",
            "schema_version": 2,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "processed_products": len(processed_urls),
            "unresolved_failures": 0,
            "output_sha256": {path.name: hash_file(path) for path in output_paths},
        },
        completion_path,
    )
    return {
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "already_processed": len(processed_urls) - completed,
    }


def run_acquisition(
    checkpoint_dir: Path,
    start_date: date,
    end_date: date,
    *,
    force: bool = False,
    hash_file: Callable[[Path], str] = sha256_file,
) -> dict[str, int]:
    """Run the inclusive GOES acquisition/preprocessing stage.

    Parameters
    ----------
    checkpoint_dir:
        Directory containing or receiving the ``processed_data.pkl``,
        ``faulty_data.pkl``, and ``df_08.pkl`` through ``df_18.pkl`` files.
        The caller supplies this path explicitly, and all persistent stage state
        is written beneath it.
    start_date, end_date:
        Inclusive dates within 1995-07-01 through 2025-05-10.
    force:
        If true, remove and recompute the requested interval.  Otherwise,
        URLs already recorded in ``processed_data.pkl`` are resumed/skipped.
    hash_file:
        Stable file-digest function used to bind completion metadata. The
        pipeline supplies its revision-aware cache for study-scale files.

    Returns
    -------
    dict
        Counts for attempted, completed, failed, and already-processed files.

    Notes
    -----
    This is a resource-intensive full-interval upstream stage. The synchronous
    function call gives command-line wrappers one explicit execution boundary.
    """

    if not isinstance(checkpoint_dir, Path):
        raise TypeError("checkpoint_dir must be a pathlib.Path")
    _validate_study_dates(start_date, end_date)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _run_acquisition_async(checkpoint_dir.resolve(), start_date, end_date, force, hash_file)
        )
    raise RuntimeError(
        "run_acquisition cannot run inside an active asyncio event loop; "
        "invoke it from the command-line stage wrapper"
    )


__all__ = [
    "DOWNLOAD_CONFIG",
    "SATELLITE_SOURCES",
    "STUDY_END",
    "STUDY_START",
    "run_acquisition",
]

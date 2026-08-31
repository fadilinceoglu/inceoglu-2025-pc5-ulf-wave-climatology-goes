"""Magnetic-coordinate transformations used by the GOES preprocessing stage.

Only the three transformations used by GOES preparation live here.  SpacePy is
imported inside :func:`llr_to_mlt`, so importing this module neither requires
the optional acquisition dependencies nor performs coordinate conversion.

The calculation uses these numerical conventions:

* J2000 seconds are measured from 2000-01-01 12:00:00 UTC;
* geographic radius is divided by 6,371 km before the SpacePy conversion;
* MLT is obtained from the arctangent-plus-quadrant formula; and
* the mean field is a clock-aligned 30-minute block mean interpolated back to
  one-minute cadence.  It is *not* a rolling 30-minute mean.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .config import MFA_BASELINE_MINUTES

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


J2000_EPOCH = datetime(2000, 1, 1, 12)
"""Naive UTC epoch used by the NOAA GOES ``time`` variables."""

EARTH_RADIUS_M = 6_371_000.0
"""Earth radius used by the LLR-to-MLT conversion, in metres."""

MEAN_FIELD_INTERVAL = f"{MFA_BASELINE_MINUTES}min"
"""Clock-aligned averaging interval used to construct the MFAC basis."""

SPACEPY_VERSION = "0.7.0"
"""SpacePy release used for the source-data coordinate conversion."""

SPACEPY_IGRF_FILENAME = "igrf13coeffs.txt"
"""Coefficient table bundled with the pinned SpacePy release."""

SPACEPY_TRANSFORM_TOLERANCE_SECONDS = 30
"""CTrans tolerance used to reuse a transformation at a nearby timestamp."""

WGS84_SEMI_MAJOR_AXIS_KM = 6378.137
"""WGS84 equatorial radius used by SpacePy CTrans, in kilometres."""

WGS84_INVERSE_FLATTENING = 298.257223563
"""WGS84 inverse flattening used by SpacePy CTrans."""

_SPACEPY_IMPORT_LOCK = threading.RLock()
_SPACEPY_STATE_DIRECTORY = None
_SPACEPY_CTRANS_PARTS = None


def _spacepy_was_imported() -> bool:
    """Return whether this process has loaded SpacePy or one of its modules."""

    return any(name == "spacepy" or name.startswith("spacepy.") for name in sys.modules)


def _controlled_spacepy_parent() -> str:
    """Return a process-local parent for SpacePy's ``.spacepy`` directory."""

    global _SPACEPY_STATE_DIRECTORY
    if _SPACEPY_STATE_DIRECTORY is None:
        _SPACEPY_STATE_DIRECTORY = tempfile.TemporaryDirectory(prefix="pc5-spacepy-")
    return _SPACEPY_STATE_DIRECTORY.name


def _bundled_igrf_path(spacepy_module: object) -> Path:
    """Resolve the IGRF13 table distributed inside the SpacePy package."""

    package_file = getattr(spacepy_module, "__file__", None)
    if package_file is None:
        raise RuntimeError("Cannot locate SpacePy's bundled IGRF13 coefficients")
    path = Path(package_file).resolve().parent / "data" / SPACEPY_IGRF_FILENAME
    if not path.is_file():
        raise RuntimeError(
            f"SpacePy {SPACEPY_VERSION} is missing its bundled "
            f"{SPACEPY_IGRF_FILENAME} coefficient table"
        )
    return path


def _coefficients_match(left: object, right: object) -> bool:
    """Compare the populated values in two SpacePy IGRF coefficient sets."""

    if not np.array_equal(left.epochs, right.epochs):
        return False

    left_coefficients = left.coeffs
    right_coefficients = right.coeffs
    if set(left_coefficients) != set(right_coefficients):
        return False

    # SpacePy allocates unused degree-zero and h(n, 0) entries with np.empty.
    # Compare only the values populated from the coefficient table.
    for degree in range(1, len(right_coefficients["g"])):
        if not np.array_equal(left_coefficients["g"][degree], right_coefficients["g"][degree]):
            return False
        if not np.array_equal(
            left_coefficients["g_SV"][degree],
            right_coefficients["g_SV"][degree],
        ):
            return False
        if not np.array_equal(
            left_coefficients["h"][degree][1:],
            right_coefficients["h"][degree][1:],
        ):
            return False
        if not np.array_equal(
            left_coefficients["h_SV"][degree][1:],
            right_coefficients["h_SV"][degree][1:],
        ):
            return False
    return True


def _ellipsoids_match(left: object, right: object) -> bool:
    """Compare every numerical field in two SpacePy ellipsoid mappings."""

    if set(left) != set(right):
        return False
    return all(np.array_equal(left[key], right[key]) for key in right)


def _validate_spacepy_contract(spacepy_module: object, coordinates_module: object) -> object:
    """Require SpacePy 0.7.0 CTrans with its bundled IGRF13 coefficients."""

    version = getattr(spacepy_module, "__version__", None)
    if version != SPACEPY_VERSION:
        raise RuntimeError(
            f"GOES coordinate conversion requires SpacePy {SPACEPY_VERSION}; "
            f"found {version or 'an unknown version'}"
        )

    bundled_path = _bundled_igrf_path(spacepy_module)

    ctrans_module = getattr(coordinates_module, "ctrans", None)
    igrf_module = getattr(ctrans_module, "igrf", None)
    active_coefficients = getattr(igrf_module, "igrfcoeffs", None)
    coefficient_type = getattr(igrf_module, "IGRFCoefficients", None)
    if active_coefficients is None or coefficient_type is None:
        raise RuntimeError("Cannot verify the IGRF coefficients used by SpacePy CTrans")

    bundled_coefficients = coefficient_type(fname=str(bundled_path))
    if not _coefficients_match(active_coefficients, bundled_coefficients):
        raise RuntimeError(
            "SpacePy CTrans is not using the bundled IGRF13 coefficients. "
            "Start a fresh Python process before running GOES acquisition."
        )
    return bundled_coefficients


def _spacepy_ctrans_parts() -> tuple[object, object, object, object]:
    """Load pinned SpacePy CTrans parts under a deterministic data state."""

    global _SPACEPY_CTRANS_PARTS
    with _SPACEPY_IMPORT_LOCK:
        if _SPACEPY_CTRANS_PARTS is not None:
            return _SPACEPY_CTRANS_PARTS

        if _spacepy_was_imported():
            raise RuntimeError(
                "GOES coordinate conversion must initialize SpacePy itself in "
                "isolated state. Start acquisition in a fresh Python process."
            )
        previous_spacepy_parent = os.environ.get("SPACEPY")
        os.environ["SPACEPY"] = _controlled_spacepy_parent()
        try:
            spacepy_module = importlib.import_module("spacepy")
            coordinates_module = importlib.import_module("spacepy.coordinates")
            time_module = importlib.import_module("spacepy.time")
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Full GOES acquisition requires the optional 'spacepy' dependency"
            ) from exc
        finally:
            if previous_spacepy_parent is None:
                os.environ.pop("SPACEPY", None)
            else:
                os.environ["SPACEPY"] = previous_spacepy_parent

        bundled_coefficients = _validate_spacepy_contract(spacepy_module, coordinates_module)
        _SPACEPY_CTRANS_PARTS = (
            coordinates_module.ctrans.convert_multitime,
            time_module.Ticktock,
            coordinates_module,
            bundled_coefficients,
        )
        return _SPACEPY_CTRANS_PARTS


def j2000_seconds_to_datetime(seconds: Iterable[float]) -> "NDArray[np.object_]":
    """Convert seconds since J2000 noon to naive UTC ``datetime`` objects.

    The NetCDF product supplies J2000 seconds.  The calculation uses a NumPy
    object array of naive UTC datetimes rather than timezone-aware timestamps.
    """

    values = np.asarray(seconds)
    return np.asarray(
        [J2000_EPOCH + timedelta(seconds=float(value)) for value in values],
        dtype=object,
    )


def calculate_mlt(
    solar_magnetic_x: "ArrayLike", solar_magnetic_y: "ArrayLike"
) -> tuple["NDArray[np.float64]", "NDArray[np.float64]"]:
    """Calculate magnetic local time from Solar Magnetic Cartesian x and y.

    The calculation uses ``arctan(y / x)`` with explicit quadrant corrections
    instead of replacing it with
    ``arctan2``.  The result is MLT in hours on the interval [0, 24], together
    with the intermediate angle in degrees.
    """

    sm_x = np.asarray(solar_magnetic_x, dtype=float)
    sm_y = np.asarray(solar_magnetic_y, dtype=float)
    if sm_x.shape != sm_y.shape:
        raise ValueError("Solar Magnetic x and y arrays must have the same shape")

    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.rad2deg(np.arctan(sm_y / sm_x))
    theta = theta + 180.0 * ((sm_x < 0) & (sm_y > 0)) - 180.0 * ((sm_x < 0) & (sm_y < 0))
    mlt = 12.0 + theta * 12.0 / 180.0
    negative_x_axis = (sm_x < 0) & (sm_y == 0)
    mlt[negative_x_axis] = 0.0
    theta[negative_x_axis] = -180.0
    return mlt, theta


def llr_to_mlt(orbit_llr_geo: "ArrayLike", timestamps: Iterable[object]) -> "NDArray[np.float64]":
    """Convert geographic latitude/longitude/radius samples to MLT hours.

    Parameters
    ----------
    orbit_llr_geo:
        N-by-3 array ordered as latitude [deg], longitude [deg], radius [m].
    timestamps:
        N UTC-like timestamps accepted by SpacePy's ``Ticktock``.

    Notes
    -----
    The input is copied before its radius column is converted to Earth radii.
    SpacePy is a lazy optional dependency and ``use_irbem=False`` selects the
    CTrans route used by this calculation.
    """

    llr = np.asarray(orbit_llr_geo, dtype=float).copy()
    times = np.asarray(list(timestamps), dtype=object)
    if llr.shape != (times.size, 3):
        raise ValueError("orbit_llr_geo must have shape (number of timestamps, 3)")

    with _SPACEPY_IMPORT_LOCK:
        (
            convert_multitime,
            Ticktock,
            coordinates_module,
            bundled_coefficients,
        ) = _spacepy_ctrans_parts()
        active_coefficients = coordinates_module.ctrans.igrf.igrfcoeffs
        if not _coefficients_match(active_coefficients, bundled_coefficients):
            raise RuntimeError(
                "SpacePy CTrans IGRF coefficients changed after initialization. "
                "Start a fresh Python process before running GOES acquisition."
            )

        ctrans_module = coordinates_module.ctrans
        exact_wgs84 = ctrans_module.Ellipsoid(
            name="WGS84",
            A=WGS84_SEMI_MAJOR_AXIS_KM,
            iFlat=WGS84_INVERSE_FLATTENING,
        )
        if not _ellipsoids_match(ctrans_module.WGS84, exact_wgs84):
            raise RuntimeError(
                "SpacePy CTrans WGS84 constants differ from the pinned values. "
                "Start a fresh Python process before running GOES acquisition."
            )

        llr[:, 2] /= EARTH_RADIUS_M
        ticks = Ticktock(times, "UTC")
        ctrans_defaults = SimpleNamespace(
            ellipsoid=exact_wgs84,
            itol=SPACEPY_TRANSFORM_TOLERANCE_SECONDS,
        )
        solar_magnetic = np.atleast_2d(
            convert_multitime(
                llr[:, [2, 0, 1]],
                ticks,
                "RLL",
                "SM",
                defaults=ctrans_defaults,
            )
        )
    mlt, _ = calculate_mlt(solar_magnetic[:, 0], solar_magnetic[:, 1])
    return mlt


def mean_field_aligned_components(epn: pd.DataFrame) -> pd.DataFrame:
    """Rotate E/N/P magnetic components into the paper's MFAC basis.

    ``epn`` must contain ``time`` followed by exactly three component columns.
    The acquisition stage passes them in E, N, P order after negating E to make the source
    system right-handed.  The first basis vector is radial/anti-Earthward, the
    second azimuthal/eastward, and the third parallel to the interpolated mean
    field.

    The mean field is constructed by clock-aligned 30-minute resampling and
    linear time interpolation.  Degenerate or missing mean-field vectors
    naturally produce NaNs.
    """

    if "time" not in epn.columns or len(epn.columns) != 4:
        raise ValueError("epn must contain 'time' and exactly three components")

    indexed = epn.copy().set_index("time")
    mean_field = indexed.resample(MEAN_FIELD_INTERVAL).mean()
    full_mean_field = (
        mean_field.reindex(mean_field.index.union(indexed.index))
        .interpolate(method="index", limit_direction="both")
        .loc[indexed.index]
    )

    output = np.full((len(indexed), 3), np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for index in range(len(indexed)):
            field = full_mean_field.iloc[index].to_numpy(dtype=float)
            field_norm = np.sqrt(np.sum(field**2))
            field_xz_norm = np.sqrt(field[0] ** 2 + field[2] ** 2)

            parallel = field / field_norm
            radial_reference = np.asarray([field[0] / field_xz_norm, 0.0, field[2] / field_xz_norm])
            cross = np.cross(parallel, radial_reference)
            azimuthal = cross / np.sqrt(np.sum(cross**2))
            radial = np.cross(azimuthal, parallel)
            transform = np.vstack((radial, azimuthal, parallel))
            output[index] = transform @ indexed.iloc[index].to_numpy(dtype=float)

    result = pd.DataFrame(
        output,
        columns=["b_radial", "b_azimuthal", "b_parallel"],
        index=indexed.index,
    )
    return result.reset_index()


__all__ = [
    "EARTH_RADIUS_M",
    "J2000_EPOCH",
    "MEAN_FIELD_INTERVAL",
    "SPACEPY_IGRF_FILENAME",
    "SPACEPY_TRANSFORM_TOLERANCE_SECONDS",
    "SPACEPY_VERSION",
    "WGS84_INVERSE_FLATTENING",
    "WGS84_SEMI_MAJOR_AXIS_KM",
    "calculate_mlt",
    "j2000_seconds_to_datetime",
    "llr_to_mlt",
    "mean_field_aligned_components",
]

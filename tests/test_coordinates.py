from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pc5_climatology.coordinates import llr_to_mlt

EXPECTED_MLT = [
    0.22143363291274198,
    17.50308791819903,
    23.860394515134836,
]


def _require_spacepy() -> Path:
    specification = importlib.util.find_spec("spacepy")
    if specification is None or specification.origin is None:
        pytest.skip("SpacePy is installed by the optional full-stage dependency group")
    return Path(specification.origin).resolve().parent


def _subprocess_environment(spacepy_parent: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["SPACEPY"] = str(spacepy_parent)
    source_directory = Path(__file__).resolve().parents[1] / "src"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_directory)
    if existing_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + existing_pythonpath
    return environment


def _coordinate_subprocess_code(prefix: str = "") -> str:
    return f"""
{prefix}
import json
from datetime import datetime
import numpy as np
from pc5_climatology.coordinates import llr_to_mlt

samples = np.asarray([
    [0.0, 0.0, 42_164_000.0],
    [10.0, 90.0, 42_164_000.0],
    [-10.0, -90.0, 42_164_000.0],
])
timestamps = [
    datetime(1995, 7, 1, 0),
    datetime(2012, 6, 15, 12),
    datetime(2024, 1, 1, 6),
]
result = llr_to_mlt(samples, timestamps)
import spacepy
print(json.dumps({{"mlt": result.tolist(), "dot_fln": spacepy.DOT_FLN}}))
"""


def test_spacepy_rll_to_mlt_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_spacepy()
    monkeypatch.setenv("SPACEPY", str(tmp_path))
    samples = np.asarray(
        [
            [0.0, 0.0, 42_164_000.0],
            [10.0, 90.0, 42_164_000.0],
            [-10.0, -90.0, 42_164_000.0],
        ]
    )
    timestamps = [
        datetime(1995, 7, 1, 0),
        datetime(2012, 6, 15, 12),
        datetime(2024, 1, 1, 6),
    ]

    result = llr_to_mlt(samples, timestamps)

    np.testing.assert_allclose(
        result,
        EXPECTED_MLT,
        rtol=0,
        atol=1e-12,
    )


def test_fresh_import_ignores_user_igrf_override(tmp_path: Path) -> None:
    _require_spacepy()
    fake_home = tmp_path / "home"
    override = fake_home / ".spacepy" / "data" / "igrfcoeffs.txt"
    override.parent.mkdir(parents=True)
    override.write_text("this must never be read\n", encoding="utf-8")
    environment = _subprocess_environment(fake_home)
    environment.pop("SPACEPY")
    environment["HOME"] = str(fake_home)

    completed = subprocess.run(
        [sys.executable, "-c", _coordinate_subprocess_code()],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    np.testing.assert_allclose(payload["mlt"], EXPECTED_MLT, rtol=0, atol=1e-12)
    assert Path(payload["dot_fln"]) != fake_home / ".spacepy"


def test_preimported_spacepy_rejects_user_igrf_override(tmp_path: Path) -> None:
    _require_spacepy()
    user_parent = tmp_path / "user-spacepy"
    override = user_parent / ".spacepy" / "data" / "igrfcoeffs.txt"
    override.parent.mkdir(parents=True)
    override.write_text("this must be rejected before CTrans reads it\n", encoding="utf-8")
    code = _coordinate_subprocess_code(prefix="import spacepy")

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        env=_subprocess_environment(user_parent),
        text=True,
    )

    assert completed.returncode != 0
    assert "must initialize SpacePy itself" in completed.stderr


def test_preimported_ctrans_rejects_removed_custom_coefficients(tmp_path: Path) -> None:
    package_directory = _require_spacepy()
    bundled = package_directory / "data" / "igrf13coeffs.txt"
    custom_text = bundled.read_text(encoding="utf-8").replace("-31543", "-31542", 1)
    user_parent = tmp_path / "user-spacepy"
    override = user_parent / ".spacepy" / "data" / "igrfcoeffs.txt"
    override.parent.mkdir(parents=True)
    override.write_text(custom_text, encoding="utf-8")
    prefix = f"""
import os
import spacepy.coordinates
os.unlink({str(override)!r})
"""

    completed = subprocess.run(
        [sys.executable, "-c", _coordinate_subprocess_code(prefix=prefix)],
        capture_output=True,
        env=_subprocess_environment(user_parent),
        text=True,
    )

    assert completed.returncode != 0
    assert "must initialize SpacePy itself" in completed.stderr


def test_preimported_spacepy_is_rejected_before_conversion(
    tmp_path: Path,
) -> None:
    _require_spacepy()
    user_parent = tmp_path / "user-spacepy"
    prefix = """
import spacepy.coordinates
spacepy.coordinates.DEFAULTS.set_values(itol=1e12)
"""

    completed = subprocess.run(
        [sys.executable, "-c", _coordinate_subprocess_code(prefix=prefix)],
        capture_output=True,
        env=_subprocess_environment(user_parent),
        text=True,
    )

    assert completed.returncode != 0
    assert "must initialize SpacePy itself" in completed.stderr


def test_conversion_rejects_igrf_mutation_after_initialization(
    tmp_path: Path,
) -> None:
    package_directory = _require_spacepy()
    bundled = package_directory / "data" / "igrf13coeffs.txt"
    custom = tmp_path / "custom-igrf.txt"
    custom.write_text(
        bundled.read_text(encoding="utf-8").replace("-31543", "-31542", 1),
        encoding="utf-8",
    )
    prefix = f"""
from datetime import datetime
import numpy as np
from pc5_climatology.coordinates import llr_to_mlt
llr_to_mlt(np.asarray([[0.0, 0.0, 42_164_000.0]]), [datetime(2000, 1, 1)])
import spacepy.coordinates
spacepy.coordinates.ctrans.igrf.igrfcoeffs = (
    spacepy.coordinates.ctrans.igrf.IGRFCoefficients(fname={str(custom)!r})
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", _coordinate_subprocess_code(prefix=prefix)],
        capture_output=True,
        env=_subprocess_environment(tmp_path / "user-spacepy"),
        text=True,
    )

    assert completed.returncode != 0
    assert "coefficients changed after initialization" in completed.stderr


def test_conversion_rejects_wgs84_mutation_after_initialization(
    tmp_path: Path,
) -> None:
    _require_spacepy()
    prefix = """
from datetime import datetime
import numpy as np
from pc5_climatology.coordinates import llr_to_mlt
llr_to_mlt(np.asarray([[0.0, 0.0, 42_164_000.0]]), [datetime(2000, 1, 1)])
import spacepy.coordinates
spacepy.coordinates.ctrans.WGS84["A"] = 1.0
"""

    completed = subprocess.run(
        [sys.executable, "-c", _coordinate_subprocess_code(prefix=prefix)],
        capture_output=True,
        env=_subprocess_environment(tmp_path / "user-spacepy"),
        text=True,
    )

    assert completed.returncode != 0
    assert "WGS84 constants differ" in completed.stderr

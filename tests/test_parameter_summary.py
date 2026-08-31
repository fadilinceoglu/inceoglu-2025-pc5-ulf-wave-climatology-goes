from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on the supported Python 3.9
    import tomli as tomllib

from pc5_climatology.config import paper_parameter_summary


def test_paper_toml_matches_every_executable_parameter() -> None:
    """The documentation summary must contain exactly the executable fields."""

    parameter_file = Path(__file__).resolve().parents[1] / "configs" / "paper.toml"
    with parameter_file.open("rb") as stream:
        documented = tomllib.load(stream)

    assert documented == paper_parameter_summary()

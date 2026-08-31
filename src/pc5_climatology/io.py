"""Small, explicit I/O primitives shared by pipeline stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


@contextmanager
def atomic_destination(destination: Path, suffix: str = ".part") -> Iterator[Path]:
    """Yield a sibling temporary path and publish it with ``os.replace``.

    A failed calculation leaves the last complete artifact untouched.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix, dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Any, destination: Path) -> None:
    with atomic_destination(destination, suffix=".json.part") as temporary:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(paths: Sequence[Path], *, purpose: str) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing files required for {purpose}:\n  - {formatted}")


def artifact_record(path: Path, root: Path) -> dict[str, Any]:
    """Return the stable path, size, and checksum identity for an artifact.

    Repository artifacts use paths relative to ``root``. External artifacts use
    their basename plus ``external=true``; their SHA-256 digest supplies the
    content identity.
    """

    path = Path(path).resolve()
    root = Path(root).resolve()
    external = False
    try:
        display_path = path.relative_to(root)
    except ValueError:
        display_path = Path(path.name)
        external = True
    record = {
        "path": str(display_path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if external:
        record["external"] = True
    return record

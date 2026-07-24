"""Dataset snapshots: freeze data/raw/ + data/processed/ into an immutable,
content-hashed directory under data/snapshots/{id}/ so a backtest or a live
model can be pinned to a known-good dataset state instead of always
reflecting whatever's currently on disk (which tennis-data.co.uk/TML can
silently revise — see
docs/superpowers/specs/2026-07-24-snapshot-persistence-design.md).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_ROOT = _ROOT / "data" / "snapshots"
_RAW_DIR = _ROOT / "data" / "raw"
_PROCESSED_DIR = _ROOT / "data" / "processed"

_RAW_TOUR_DIRS = {
    "atp": "tennis_atp_tml",
    "wta": "tennis_wta_tduk",
}

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MB, streamed so multi-MB CSVs don't need
                                 # to be fully loaded into memory to hash.


class SnapshotExistsError(Exception):
    """Raised by create_snapshot() when snapshot_id already exists —
    snapshots are immutable, callers wanting to redo one must delete the
    directory first."""


class SnapshotNotFoundError(Exception):
    """Raised when a snapshot_id doesn't exist under SNAPSHOT_ROOT."""


def _sha256_file(path: Path) -> str:
    """Stream-hash a file in chunks so multi-MB CSVs don't need to be fully
    loaded into memory just to compute a digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path: Path) -> dict:
    return {"sha256": _sha256_file(path), "bytes": path.stat().st_size}

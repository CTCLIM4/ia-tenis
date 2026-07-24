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


def _current_git_commit() -> str | None:
    """Best-effort short git commit hash, None if git is unavailable or
    this isn't a git checkout — never blocks snapshot creation on this
    being unavailable."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def create_snapshot(snapshot_id: str | None = None, tours: tuple[str, ...] = ("atp", "wta")) -> Path:
    """Create data/snapshots/{snapshot_id}/, copying each tour's processed
    features CSV and raw match files, then writing metadata.json with
    per-tour last_match_date, row count, and per-file sha256/bytes.

    Raises SnapshotExistsError if the target directory already exists —
    snapshots are immutable, delete the directory first to recreate one.
    Raises FileNotFoundError if a tour's processed features CSV is missing
    (run `python -m src.pipeline {tour}` first); the partial snapshot
    directory is cleaned up before the error propagates, so a failed
    creation never leaves a half-written snapshot behind.
    """
    import json
    import shutil
    from datetime import date, datetime

    import pandas as pd

    snapshot_id = snapshot_id or date.today().isoformat()
    snapshot_dir = SNAPSHOT_ROOT / snapshot_id
    if snapshot_dir.exists():
        raise SnapshotExistsError(
            f"Snapshot '{snapshot_id}' already exists at {snapshot_dir}. "
            "Snapshots are immutable — delete the directory first to recreate it."
        )

    processed_dir = snapshot_dir / "processed"
    raw_dir = snapshot_dir / "raw"

    try:
        processed_dir.mkdir(parents=True)
        raw_dir.mkdir(parents=True)

        tours_meta: dict = {}
        for tour in tours:
            features_src = _PROCESSED_DIR / f"{tour}_features.csv"
            if not features_src.exists():
                raise FileNotFoundError(
                    f"No se encontro {features_src}. Corre primero: python -m src.pipeline {tour}"
                )
            features_dst = processed_dir / features_src.name
            shutil.copy2(features_src, features_dst)

            df = pd.read_csv(features_src)
            last_match_date = df["match_date"].max()

            raw_tour_src = _RAW_DIR / _RAW_TOUR_DIRS[tour]
            raw_tour_dst = raw_dir / _RAW_TOUR_DIRS[tour]
            raw_tour_dst.mkdir(parents=True)
            files_meta = {
                f"processed/{features_dst.name}": _file_record(features_dst),
            }
            if raw_tour_src.exists():
                for raw_file in sorted(raw_tour_src.iterdir()):
                    if not raw_file.is_file():
                        continue
                    raw_dst = raw_tour_dst / raw_file.name
                    shutil.copy2(raw_file, raw_dst)
                    files_meta[f"raw/{_RAW_TOUR_DIRS[tour]}/{raw_file.name}"] = _file_record(raw_dst)

            tours_meta[tour] = {
                "last_match_date": str(last_match_date),
                "n_rows": int(len(df)),
                "files": files_meta,
            }
    except Exception:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise

    metadata = {
        "snapshot_id": snapshot_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _current_git_commit(),
        "tours": tours_meta,
    }
    with open(snapshot_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return snapshot_dir

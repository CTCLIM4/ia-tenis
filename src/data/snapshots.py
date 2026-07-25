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

from src.git_utils import current_git_commit

_ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_ROOT = _ROOT / "data" / "snapshots"
_RAW_DIR = _ROOT / "data" / "raw"
_PROCESSED_DIR = _ROOT / "data" / "processed"

_RAW_TOUR_DIRS = {
    "atp": "tennis_atp_tml",
    "wta": "tennis_wta_tduk",
}

# Only these extensions are ever read by src/data/loader.py's load_atp_matches/
# load_wta_matches — anything else in a raw tour directory (stray .gitignore
# files left over from the pre-API-migration git-clone ATP download, OS
# artifacts like .DS_Store, editor scratch files, etc.) is not real match
# data and must not be copied into a snapshot.
_ALLOWED_RAW_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def _is_real_raw_data_file(path: Path) -> bool:
    """True if path should be included in a snapshot's raw/ copy: a
    non-hidden file with an extension load_atp_matches/load_wta_matches
    actually reads."""
    return not path.name.startswith(".") and path.suffix.lower() in _ALLOWED_RAW_EXTENSIONS

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
    Raises ValueError if snapshot_id isn't a plain directory-name-safe
    string — it flows straight into a path that's later shutil.rmtree'd on
    failure, so path separators and '.'/'..' are rejected outright rather
    than trusting the caller not to pass something like '../elsewhere'.
    """
    import json
    import re
    import shutil
    from datetime import date, datetime

    import pandas as pd

    snapshot_id = snapshot_id or date.today().isoformat()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", snapshot_id) or snapshot_id in (".", ".."):
        raise ValueError(
            f"snapshot_id invalido: {snapshot_id!r}. Solo se permiten letras, "
            "numeros, '-', '_' y '.' (sin separadores de ruta ni '..')."
        )
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
                    if not raw_file.is_file() or not _is_real_raw_data_file(raw_file):
                        continue
                    raw_dst = raw_tour_dst / raw_file.name
                    shutil.copy2(raw_file, raw_dst)
                    files_meta[f"raw/{_RAW_TOUR_DIRS[tour]}/{raw_file.name}"] = _file_record(raw_dst)

            tours_meta[tour] = {
                "last_match_date": str(last_match_date),
                "n_rows": int(len(df)),
                "files": files_meta,
            }

        metadata = {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": current_git_commit(),
            "tours": tours_meta,
        }
        with open(snapshot_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise

    return snapshot_dir


def list_snapshots() -> list[str]:
    """Sorted list of snapshot ids found under SNAPSHOT_ROOT (ISO date ids
    sort chronologically as plain strings). A directory only counts as a
    snapshot if it has a metadata.json — an incomplete/manually-created
    directory without one is silently skipped rather than raising."""
    if not SNAPSHOT_ROOT.exists():
        return []
    return sorted(
        p.name for p in SNAPSHOT_ROOT.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )


def load_snapshot_metadata(snapshot_id: str) -> dict:
    """Load and parse SNAPSHOT_ROOT/{snapshot_id}/metadata.json.

    Raises SnapshotNotFoundError specifically when metadata.json is missing
    — not when the snapshot directory itself is missing, which produces the
    same error since a snapshot without metadata.json isn't a valid
    snapshot (see list_snapshots()'s matching skip logic).
    """
    import json

    path = SNAPSHOT_ROOT / snapshot_id / "metadata.json"
    if not path.exists():
        raise SnapshotNotFoundError(f"No metadata.json for snapshot '{snapshot_id}' at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_snapshot_path(snapshot_id: str, tour: str, kind: Literal["processed", "raw"]) -> Path:
    """Path to a snapshot's data for one tour.

    kind="processed" -> the {tour}_features.csv copy (for backtest reuse,
    src/pipeline.py's --snapshot).
    kind="raw" -> the tour's raw match-file directory copy (for
    _build_elo_fb-style live Elo/FeatureBuilder replay,
    src/value_analysis.py's load_model(snapshot=...)).
    """
    snapshot_dir = SNAPSHOT_ROOT / snapshot_id
    if not snapshot_dir.exists():
        raise SnapshotNotFoundError(f"Snapshot '{snapshot_id}' not found at {snapshot_dir}")
    if kind == "processed":
        return snapshot_dir / "processed" / f"{tour}_features.csv"
    if kind == "raw":
        return snapshot_dir / "raw" / _RAW_TOUR_DIRS[tour]
    raise ValueError(f"Unknown kind: {kind!r} (expected 'processed' or 'raw')")


def verify_snapshot_integrity(snapshot_id: str) -> bool:
    """Re-hash every file recorded in metadata.json and compare. True only
    if every file exists and matches its recorded sha256/bytes exactly —
    this is what makes the hashes in metadata.json useful for detecting
    local corruption or manual tampering, not just decorative."""
    metadata = load_snapshot_metadata(snapshot_id)
    snapshot_dir = SNAPSHOT_ROOT / snapshot_id
    for tour_meta in metadata["tours"].values():
        for rel_path, recorded in tour_meta["files"].items():
            actual_path = snapshot_dir / rel_path
            if not actual_path.exists():
                return False
            if actual_path.stat().st_size != recorded["bytes"]:
                return False
            if _sha256_file(actual_path) != recorded["sha256"]:
                return False
    return True

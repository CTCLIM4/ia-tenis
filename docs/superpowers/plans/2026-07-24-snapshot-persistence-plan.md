# Dataset Snapshots & Reproducible Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `src/pipeline.py`'s backtest and `src/value_analysis.py`'s `load_model()` be pinned to an immutable, content-hashed snapshot of the datasets (`data/snapshots/{id}/`) instead of always reading live `data/raw/`/`data/processed/`, which `tennis-data.co.uk`/TML can silently revise.

**Architecture:** A new `src/data/snapshots.py` module owns snapshot creation (`create_snapshot`), lookup (`list_snapshots`, `load_snapshot_metadata`, `resolve_snapshot_path`), and integrity checking (`verify_snapshot_integrity`) — pure functions operating on `Path`s, no coupling to `pipeline.py`/`value_analysis.py` internals. `src/data/loader.py`'s two loaders gain an optional raw-directory override so `_build_elo_fb` can replay a snapshot's frozen raw files instead of live ones. `src/backtest/walkforward.py` gains a small `load_features_with_mirror()` helper (extracted from `value_analysis.py`'s `_train_lr`, removing duplicated mirroring logic) that both the pipeline's `--snapshot` path and the live LR-training path use. `value_analysis.py`'s cache gains a snapshot dimension (`{tour}__snapshot-{id}.pkl`, never expires) so a pinned load can't collide with or be evicted by live/`--retrain` loads.

**Tech Stack:** Python 3.14, pytest, hashlib (stdlib), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-24-snapshot-persistence-design.md` (all design questions confirmed 2026-07-24).

---

## File Map

```
D:\ia-tenis\
├── src/
│   ├── backtest/
│   │   └── walkforward.py       # MODIFIED — new load_features_with_mirror()
│   ├── data/
│   │   ├── loader.py            # MODIFIED — raw_dir_override param on both loaders
│   │   └── snapshots.py         # NEW — create/list/load/resolve/verify
│   ├── pipeline.py               # MODIFIED — --snapshot flag
│   └── value_analysis.py         # MODIFIED — snapshot-aware load_model()
├── scripts/
│   └── create_snapshot.py        # NEW — CLI wrapper around create_snapshot()
├── tests/
│   ├── test_backtest.py          # MODIFIED — load_features_with_mirror tests
│   ├── test_loader.py            # MODIFIED — raw_dir_override tests
│   ├── test_value_analysis.py    # MODIFIED — _train_lr features_path, load_model snapshot tests
│   ├── test_snapshots.py         # NEW
│   ├── test_create_snapshot.py   # NEW
│   └── test_pipeline.py          # NEW
├── docs/
│   └── runbooks/
│       └── data-refresh-and-staleness.md   # MODIFIED — new "Dataset snapshots" section
└── .gitignore                    # MODIFIED — ignore snapshot data files, keep metadata.json trackable
```

---

## Task 1: Extract `load_features_with_mirror()`, refactor `_train_lr` to use it

**Files:**
- Modify: `src/backtest/walkforward.py:1-30` (imports + insert function after `_MIRROR_FLIP_COLS`)
- Modify: `src/value_analysis.py:439-477` (`_train_lr`)
- Modify: `tests/test_backtest.py` (new test class)
- Modify: `tests/test_value_analysis.py` (new test method)

- [ ] **Step 1: Write failing tests for `load_features_with_mirror`**

In `tests/test_backtest.py`, add after the existing `_synthetic_features` helper (before `test_returns_metrics_dict_with_one_entry_per_test_year`):

```python
class TestLoadFeaturesWithMirror:
    def test_returns_equal_original_and_mirror_counts(self, tmp_path):
        path = tmp_path / "atp_features.csv"
        full = _synthetic_features(n=50)
        full[~full["is_mirror"]].to_csv(path, index=False)

        result = load_features_with_mirror(path)

        n_original = int((~result["is_mirror"]).sum())
        n_mirror = int(result["is_mirror"].sum())
        assert n_original == n_mirror == 50
        assert len(result) == 100

    def test_mirror_rows_have_flipped_diff_columns(self, tmp_path):
        path = tmp_path / "atp_features.csv"
        full = _synthetic_features(n=10)
        full[~full["is_mirror"]].to_csv(path, index=False)

        result = load_features_with_mirror(path)
        original = result[~result["is_mirror"]].reset_index(drop=True)
        mirror = result[result["is_mirror"]].reset_index(drop=True)

        for col in _MIRROR_FLIP_COLS:
            assert (mirror[col] == -original[col]).all()

    def test_mirror_rows_have_outcome_zero_original_have_one(self, tmp_path):
        path = tmp_path / "atp_features.csv"
        full = _synthetic_features(n=10)
        full[~full["is_mirror"]].to_csv(path, index=False)

        result = load_features_with_mirror(path)

        assert (result[result["is_mirror"]]["outcome"] == 0).all()
        assert (result[~result["is_mirror"]]["outcome"] == 1).all()
```

Change the import line at the top of `tests/test_backtest.py`:

```python
import src.backtest.walkforward as walkforward_module
from src.backtest.walkforward import _FEATURE_COLS, walk_forward_backtest
```

to:

```python
import src.backtest.walkforward as walkforward_module
from src.backtest.walkforward import _FEATURE_COLS, _MIRROR_FLIP_COLS, load_features_with_mirror, walk_forward_backtest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest.py::TestLoadFeaturesWithMirror -v`
Expected: `ImportError: cannot import name 'load_features_with_mirror'` (or `_MIRROR_FLIP_COLS` if that's not exported yet — it already is, so the error will name `load_features_with_mirror`).

- [ ] **Step 3: Add `load_features_with_mirror` to `src/backtest/walkforward.py`**

Change:

```python
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from src.features.decay import EloHistoryTracker, calculate_decay_features
from src.features.engineering import FeatureBuilder
from src.models.elo import EloSystem

_FEATURE_COLS = [
    "elo_diff",
    "elo_prob",
    "rank_diff",
    "form_diff",
    "surface_form_diff",
    "h2h_rate",
    "rest_diff",
    "rolling_elo_diff",
    "age_multiplier_diff",
    "rust_factor_diff",
    "adjusted_elo_diff",
]

_MIRROR_FLIP_COLS = (
    "elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff",
)


def _age_or_none(value) -> Optional[float]:
```

to:

```python
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from src.features.decay import EloHistoryTracker, calculate_decay_features
from src.features.engineering import FeatureBuilder
from src.models.elo import EloSystem

_FEATURE_COLS = [
    "elo_diff",
    "elo_prob",
    "rank_diff",
    "form_diff",
    "surface_form_diff",
    "h2h_rate",
    "rest_diff",
    "rolling_elo_diff",
    "age_multiplier_diff",
    "rust_factor_diff",
    "adjusted_elo_diff",
]

_MIRROR_FLIP_COLS = (
    "elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff",
)


def load_features_with_mirror(path: Path) -> pd.DataFrame:
    """Load a {tour}_features.csv (as saved by src/pipeline.py's
    run_pipeline — original rows only, is_mirror=False for every row) and
    reconstruct the mirrored rows (loser's perspective, outcome=0) exactly
    as build_match_features() does internally.

    The saved CSV only persists 'original' rows to avoid doubling file size
    on disk — every consumer that needs the full mirrored training set
    (LR fitting in src/value_analysis.py's _train_lr, walk_forward_backtest
    when reading a pinned snapshot in src/pipeline.py) reconstructs it via
    this single function instead of duplicating the mirroring logic.
    """
    df = pd.read_csv(path)
    mirror = df.copy()
    for col in _MIRROR_FLIP_COLS:
        mirror[col] = -mirror[col]
    mirror["elo_prob"] = 1 - mirror["elo_prob"]
    mirror["h2h_rate"] = 1 - mirror["h2h_rate"]
    mirror["outcome"] = 0
    mirror["is_mirror"] = True
    return pd.concat([df, mirror], ignore_index=True)


def _age_or_none(value) -> Optional[float]:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest.py::TestLoadFeaturesWithMirror -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Write failing test for `_train_lr`'s `features_path` override**

In `tests/test_value_analysis.py`, add this method to the end of the `TestTrainLRPipeline` class (after `test_scaling_actually_applied_matches_manual_pipeline`):

```python
    def test_features_path_override_takes_precedence_over_default_location(self, tmp_path, monkeypatch):
        """Proves the (currently unused by any caller) features_path param
        actually overrides the default data/processed/{tour}_features.csv
        lookup — the seam Task 10 of the snapshot plan wires up later."""
        import src.value_analysis as va
        monkeypatch.setattr(va, "_DATA_DIR", tmp_path / "does_not_exist")
        custom_path = self._write_synthetic_features_csv(tmp_path, tour="atp")

        clf = va._train_lr("atp", features_path=custom_path)

        assert isinstance(clf, Pipeline)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `./tenis-env/Scripts/python.exe -m pytest "tests/test_value_analysis.py::TestTrainLRPipeline::test_features_path_override_takes_precedence_over_default_location" -v`
Expected: `TypeError: _train_lr() got an unexpected keyword argument 'features_path'`.

- [ ] **Step 7: Refactor `_train_lr` to accept `features_path` and use `load_features_with_mirror`**

Change:

```python
from src.backtest.walkforward import _MIRROR_FLIP_COLS
```

to:

```python
from src.backtest.walkforward import _MIRROR_FLIP_COLS, load_features_with_mirror
```

Change:

```python
def _train_lr(tour: str) -> Pipeline:
    """Load pre-computed features CSV and train a scaled LR on the full history.

    StandardScaler + LogisticRegression, matching walk_forward_backtest's
    per-year scaling (src/backtest/walkforward.py) — before this, the
    backtest reported metrics for a scaled-feature LR while this function
    (which trains the model actually serving live predictions) fit on raw,
    unscaled features. Fitting the scaler here uses the exact same full
    training set the LR itself sees, same as the backtest's per-year
    fit-on-train-only scaler.
    """
    path = _DATA_DIR / "processed" / f"{tour}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. "
            f"Corre primero: python -m src.pipeline {tour}"
        )
    df = pd.read_csv(path)

    mirror = df.copy()
    for col in _MIRROR_FLIP_COLS:
        mirror[col] = -mirror[col]
    mirror["elo_prob"]  = 1 - mirror["elo_prob"]
    mirror["h2h_rate"]  = 1 - mirror["h2h_rate"]
    mirror["outcome"]   = 0
    mirror["is_mirror"] = True
    full = pd.concat([df, mirror], ignore_index=True)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )
    # .values strips DataFrame column names before fit, matching the plain
    # ndarray predict_match() passes to predict_proba() later — fitting on a
    # named DataFrame and predicting on an unnamed array triggers a sklearn
    # UserWarning otherwise.
    clf.fit(full[FEATURE_COLS].fillna(0).values, full["outcome"].values)
    print(f"  LR entrenado sobre {len(df):,} partidos ({tour.upper()}).")
    return clf
```

to:

```python
def _train_lr(tour: str, features_path: Optional[Path] = None) -> Pipeline:
    """Load a features CSV and train a scaled LR on the full history.

    StandardScaler + LogisticRegression, matching walk_forward_backtest's
    per-year scaling (src/backtest/walkforward.py) — before this, the
    backtest reported metrics for a scaled-feature LR while this function
    (which trains the model actually serving live predictions) fit on raw,
    unscaled features. Fitting the scaler here uses the exact same full
    training set the LR itself sees, same as the backtest's per-year
    fit-on-train-only scaler.

    features_path: read from here instead of the default
    data/processed/{tour}_features.csv — used for snapshot-pinned training
    (src/data/snapshots.py's resolve_snapshot_path).
    """
    path = features_path or (_DATA_DIR / "processed" / f"{tour}_features.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. "
            f"Corre primero: python -m src.pipeline {tour}"
        )
    full = load_features_with_mirror(path)
    n_original = int((~full["is_mirror"]).sum())

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )
    # .values strips DataFrame column names before fit, matching the plain
    # ndarray predict_match() passes to predict_proba() later — fitting on a
    # named DataFrame and predicting on an unnamed array triggers a sklearn
    # UserWarning otherwise.
    clf.fit(full[FEATURE_COLS].fillna(0).values, full["outcome"].values)
    print(f"  LR entrenado sobre {n_original:,} partidos ({tour.upper()}).")
    return clf
```

- [ ] **Step 8: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass (213 previous + 4 new = 217), no failures, no warnings.

- [ ] **Step 9: Commit**

```bash
git add src/backtest/walkforward.py src/value_analysis.py tests/test_backtest.py tests/test_value_analysis.py
git commit -m "refactor: extract load_features_with_mirror, add _train_lr features_path override"
```

---

## Task 2: `src/data/snapshots.py` — file hashing helpers

**Files:**
- Create: `src/data/snapshots.py`
- Create: `tests/test_snapshots.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_snapshots.py`:

```python
"""Tests for src/data/snapshots.py — dataset snapshot creation, listing,
metadata, and integrity verification. No test touches the real
data/raw/, data/processed/, or data/snapshots/ — everything works against
tmp_path fixtures via monkeypatched module constants."""
from __future__ import annotations

import hashlib

import src.data.snapshots as snapshots


class TestSha256File:
    def test_matches_hashlib_reference(self, tmp_path):
        path = tmp_path / "sample.txt"
        path.write_bytes(b"hello world" * 1000)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert snapshots._sha256_file(path) == expected

    def test_different_content_gives_different_hash(self, tmp_path):
        path_a = tmp_path / "a.txt"
        path_b = tmp_path / "b.txt"
        path_a.write_bytes(b"content A")
        path_b.write_bytes(b"content B")
        assert snapshots._sha256_file(path_a) != snapshots._sha256_file(path_b)

    def test_handles_file_larger_than_chunk_size(self, tmp_path):
        path = tmp_path / "large.bin"
        path.write_bytes(b"x" * (snapshots._HASH_CHUNK_SIZE * 3 + 12345))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert snapshots._sha256_file(path) == expected


class TestFileRecord:
    def test_returns_sha256_and_bytes(self, tmp_path):
        path = tmp_path / "sample.txt"
        content = b"some content"
        path.write_bytes(content)
        record = snapshots._file_record(path)
        assert record == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_snapshots.py -v`
Expected: `ModuleNotFoundError: No module named 'src.data.snapshots'`.

- [ ] **Step 3: Create `src/data/snapshots.py` with module skeleton and hashing helpers**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_snapshots.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/snapshots.py tests/test_snapshots.py
git commit -m "feat: add snapshot file-hashing helpers (src/data/snapshots.py)"
```

---

## Task 3: `create_snapshot()`

**Files:**
- Modify: `src/data/snapshots.py` (append)
- Modify: `tests/test_snapshots.py` (append)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_snapshots.py`, after the imports (`import hashlib` block), add:

```python
import json

import pandas as pd
import pytest
```

so the top of the file reads:

```python
"""Tests for src/data/snapshots.py — dataset snapshot creation, listing,
metadata, and integrity verification. No test touches the real
data/raw/, data/processed/, or data/snapshots/ — everything works against
tmp_path fixtures via monkeypatched module constants."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import src.data.snapshots as snapshots
```

Append at the end of the file:

```python
@pytest.fixture()
def fake_data_dirs(tmp_path, monkeypatch):
    """Redirect snapshots.py's data-source and snapshot-root constants to an
    isolated tmp_path tree, and populate minimal ATP+WTA processed/raw
    fixtures."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    snapshot_root = tmp_path / "snapshots"
    monkeypatch.setattr(snapshots, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(snapshots, "_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(snapshots, "SNAPSHOT_ROOT", snapshot_root)

    processed_dir.mkdir(parents=True)
    for tour in ("atp", "wta"):
        df = pd.DataFrame({
            "match_date": ["2026-07-10", "2026-07-15", "2026-07-20"],
            "winner": ["A", "B", "A"],
            "loser":  ["B", "A", "C"],
        })
        df.to_csv(processed_dir / f"{tour}_features.csv", index=False)

    (raw_dir / "tennis_atp_tml").mkdir(parents=True)
    (raw_dir / "tennis_atp_tml" / "2026.csv").write_text("winner_name,loser_name\nA,B\n")
    (raw_dir / "tennis_wta_tduk").mkdir(parents=True)
    (raw_dir / "tennis_wta_tduk" / "2026w.csv").write_text("Winner,Loser\nA,B\n")

    return {"raw_dir": raw_dir, "processed_dir": processed_dir, "snapshot_root": snapshot_root}


class TestCreateSnapshot:
    def test_creates_expected_directory_structure(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        assert snapshot_dir == fake_data_dirs["snapshot_root"] / "2026-07-24"
        assert (snapshot_dir / "metadata.json").exists()
        assert (snapshot_dir / "processed" / "atp_features.csv").exists()
        assert (snapshot_dir / "processed" / "wta_features.csv").exists()
        assert (snapshot_dir / "raw" / "tennis_atp_tml" / "2026.csv").exists()
        assert (snapshot_dir / "raw" / "tennis_wta_tduk" / "2026w.csv").exists()

    def test_defaults_snapshot_id_to_today(self, fake_data_dirs):
        from datetime import date
        snapshot_dir = snapshots.create_snapshot()
        assert snapshot_dir.name == date.today().isoformat()

    def test_metadata_has_correct_row_counts_and_last_match_date(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        with open(snapshot_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["snapshot_id"] == "2026-07-24"
        assert meta["tours"]["atp"]["n_rows"] == 3
        assert meta["tours"]["atp"]["last_match_date"] == "2026-07-20"
        assert meta["tours"]["wta"]["n_rows"] == 3

    def test_metadata_has_file_hashes(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        with open(snapshot_dir / "metadata.json") as f:
            meta = json.load(f)
        atp_files = meta["tours"]["atp"]["files"]
        assert "processed/atp_features.csv" in atp_files
        entry = atp_files["processed/atp_features.csv"]
        assert "sha256" in entry and len(entry["sha256"]) == 64
        assert entry["bytes"] == (snapshot_dir / "processed" / "atp_features.csv").stat().st_size

    def test_metadata_has_created_at_and_git_commit_keys(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        with open(snapshot_dir / "metadata.json") as f:
            meta = json.load(f)
        assert "created_at" in meta
        assert "git_commit" in meta  # may be None outside a git checkout, key must exist regardless

    def test_raises_on_duplicate_snapshot_id(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        with pytest.raises(snapshots.SnapshotExistsError):
            snapshots.create_snapshot(snapshot_id="2026-07-24")

    def test_raises_clean_error_when_processed_csv_missing(self, fake_data_dirs):
        (fake_data_dirs["processed_dir"] / "wta_features.csv").unlink()
        with pytest.raises(FileNotFoundError):
            snapshots.create_snapshot(snapshot_id="2026-07-24")
        assert not (fake_data_dirs["snapshot_root"] / "2026-07-24").exists()

    def test_can_snapshot_a_single_tour(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24", tours=("atp",))
        assert (snapshot_dir / "processed" / "atp_features.csv").exists()
        assert not (snapshot_dir / "processed" / "wta_features.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_snapshots.py::TestCreateSnapshot -v`
Expected: `AttributeError: module 'src.data.snapshots' has no attribute 'create_snapshot'`.

- [ ] **Step 3: Add `create_snapshot()` to `src/data/snapshots.py`**

Append to `src/data/snapshots.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_snapshots.py::TestCreateSnapshot -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures.

- [ ] **Step 6: Commit**

```bash
git add src/data/snapshots.py tests/test_snapshots.py
git commit -m "feat: add create_snapshot() to src/data/snapshots.py"
```

---

## Task 4: `list_snapshots()` and `load_snapshot_metadata()`

**Files:**
- Modify: `src/data/snapshots.py` (append)
- Modify: `tests/test_snapshots.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_snapshots.py`:

```python
class TestListSnapshots:
    def test_empty_when_no_snapshots_dir(self, fake_data_dirs):
        assert snapshots.list_snapshots() == []

    def test_lists_created_snapshots_sorted(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-20")
        snapshots.create_snapshot(snapshot_id="2026-07-10")
        assert snapshots.list_snapshots() == ["2026-07-10", "2026-07-20"]

    def test_ignores_directories_without_metadata_json(self, fake_data_dirs):
        fake_data_dirs["snapshot_root"].mkdir(parents=True)
        (fake_data_dirs["snapshot_root"] / "not-a-snapshot").mkdir()
        assert snapshots.list_snapshots() == []


class TestLoadSnapshotMetadata:
    def test_loads_written_metadata(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        meta = snapshots.load_snapshot_metadata("2026-07-24")
        assert meta["snapshot_id"] == "2026-07-24"

    def test_raises_for_unknown_snapshot(self, fake_data_dirs):
        with pytest.raises(snapshots.SnapshotNotFoundError):
            snapshots.load_snapshot_metadata("does-not-exist")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_snapshots.py::TestListSnapshots tests/test_snapshots.py::TestLoadSnapshotMetadata -v`
Expected: `AttributeError: module 'src.data.snapshots' has no attribute 'list_snapshots'`.

- [ ] **Step 3: Add both functions to `src/data/snapshots.py`**

Append:

```python
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
    import json

    path = SNAPSHOT_ROOT / snapshot_id / "metadata.json"
    if not path.exists():
        raise SnapshotNotFoundError(f"No metadata.json for snapshot '{snapshot_id}' at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_snapshots.py::TestListSnapshots tests/test_snapshots.py::TestLoadSnapshotMetadata -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/snapshots.py tests/test_snapshots.py
git commit -m "feat: add list_snapshots() and load_snapshot_metadata()"
```

---

## Task 5: `resolve_snapshot_path()` and `verify_snapshot_integrity()`

**Files:**
- Modify: `src/data/snapshots.py` (append)
- Modify: `tests/test_snapshots.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_snapshots.py`:

```python
class TestResolveSnapshotPath:
    def test_processed_path(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        path = snapshots.resolve_snapshot_path("2026-07-24", "atp", "processed")
        assert path == fake_data_dirs["snapshot_root"] / "2026-07-24" / "processed" / "atp_features.csv"
        assert path.exists()

    def test_raw_path(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        path = snapshots.resolve_snapshot_path("2026-07-24", "wta", "raw")
        assert path == fake_data_dirs["snapshot_root"] / "2026-07-24" / "raw" / "tennis_wta_tduk"
        assert path.is_dir()

    def test_raises_for_unknown_snapshot(self, fake_data_dirs):
        with pytest.raises(snapshots.SnapshotNotFoundError):
            snapshots.resolve_snapshot_path("does-not-exist", "atp", "processed")

    def test_raises_for_unknown_kind(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        with pytest.raises(ValueError):
            snapshots.resolve_snapshot_path("2026-07-24", "atp", "bogus")


class TestVerifySnapshotIntegrity:
    def test_true_for_untouched_snapshot(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        assert snapshots.verify_snapshot_integrity("2026-07-24") is True

    def test_false_when_file_content_tampered(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        target = snapshot_dir / "processed" / "atp_features.csv"
        target.write_text(target.read_text() + "\nTAMPERED,ROW,HERE")
        assert snapshots.verify_snapshot_integrity("2026-07-24") is False

    def test_false_when_file_missing(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        (snapshot_dir / "processed" / "wta_features.csv").unlink()
        assert snapshots.verify_snapshot_integrity("2026-07-24") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_snapshots.py::TestResolveSnapshotPath tests/test_snapshots.py::TestVerifySnapshotIntegrity -v`
Expected: `AttributeError: module 'src.data.snapshots' has no attribute 'resolve_snapshot_path'`.

- [ ] **Step 3: Add both functions to `src/data/snapshots.py`**

Append:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_snapshots.py -v`
Expected: all tests in the file PASS (19 total across all `TestSha256File`/`TestFileRecord`/`TestCreateSnapshot`/`TestListSnapshots`/`TestLoadSnapshotMetadata`/`TestResolveSnapshotPath`/`TestVerifySnapshotIntegrity` classes).

- [ ] **Step 5: Commit**

```bash
git add src/data/snapshots.py tests/test_snapshots.py
git commit -m "feat: add resolve_snapshot_path() and verify_snapshot_integrity()"
```

---

## Task 6: `.gitignore` — commit `metadata.json`, ignore snapshot data files

**Files:**
- Modify: `.gitignore`

No automated test — `.gitignore` correctness is verified with `git check-ignore`, consistent with there being no test coverage for `.gitignore` elsewhere in this repo.

- [ ] **Step 1: Update `.gitignore`**

Change:

```
tenis-env/
data/raw/
data/processed/
data/model_cache/
data/odds_cache/
data/value_bets_log.csv
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
.pytest_cache/
.env
*.env
.env.*
.idea/
.vscode/
```

to:

```
tenis-env/
data/raw/
data/processed/
data/model_cache/
data/odds_cache/
data/value_bets_log.csv
data/snapshots/*/processed/
data/snapshots/*/raw/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
.pytest_cache/
.env
*.env
.env.*
.idea/
.vscode/
```

Note: this deliberately does **not** ignore `data/snapshots/` itself or `data/snapshots/*/metadata.json` — only the two data-bearing subdirectories per snapshot. Ignoring the whole `data/snapshots/*/` directory and trying to negate `metadata.json` back in with a `!` pattern would **not** work: git does not descend into an already-ignored directory to evaluate negation patterns for files inside it. Ignoring only `processed/` and `raw/` sidesteps that pitfall entirely — `metadata.json` was never ignored in the first place.

- [ ] **Step 2: Verify with `git check-ignore`**

Run:
```bash
git check-ignore -v data/snapshots/2026-07-24/processed/atp_features.csv
git check-ignore -v data/snapshots/2026-07-24/raw/tennis_atp_tml/2026.csv
git check-ignore -v data/snapshots/2026-07-24/metadata.json
```

Expected: the first two print a match (`.gitignore:15:data/snapshots/*/processed/  ...` and similar for `raw/`); the third prints nothing and the command exits with status 1 (not ignored). This works without any of these paths existing on disk — `git check-ignore` is a pure pattern match.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore snapshot data files, keep metadata.json trackable"
```

---

## Task 7: `scripts/create_snapshot.py`

**Files:**
- Create: `scripts/create_snapshot.py`
- Create: `tests/test_create_snapshot.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_create_snapshot.py`:

```python
"""Tests for scripts/create_snapshot.py — thin CLI wrapper around
src.data.snapshots.create_snapshot()."""
from __future__ import annotations

import pandas as pd
import pytest

import src.data.snapshots as snapshots
from scripts import create_snapshot as cs


@pytest.fixture()
def fake_data_dirs(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    snapshot_root = tmp_path / "snapshots"
    monkeypatch.setattr(snapshots, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(snapshots, "_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(snapshots, "SNAPSHOT_ROOT", snapshot_root)

    processed_dir.mkdir(parents=True)
    for tour in ("atp", "wta"):
        pd.DataFrame({"match_date": ["2026-07-20"], "winner": ["A"], "loser": ["B"]}).to_csv(
            processed_dir / f"{tour}_features.csv", index=False
        )
    (raw_dir / "tennis_atp_tml").mkdir(parents=True)
    (raw_dir / "tennis_atp_tml" / "2026.csv").write_text("winner_name,loser_name\nA,B\n")
    (raw_dir / "tennis_wta_tduk").mkdir(parents=True)
    (raw_dir / "tennis_wta_tduk" / "2026w.csv").write_text("Winner,Loser\nA,B\n")
    return snapshot_root


class TestCreateSnapshotScript:
    def test_run_creates_snapshot_and_returns_its_path(self, fake_data_dirs):
        result = cs.run(snapshot_id="2026-07-24")
        assert result == fake_data_dirs / "2026-07-24"
        assert (result / "metadata.json").exists()

    def test_run_prints_summary(self, fake_data_dirs, capsys):
        cs.run(snapshot_id="2026-07-24")
        out = capsys.readouterr().out
        assert "2026-07-24" in out
        assert "ATP" in out
        assert "WTA" in out

    def test_run_respects_tours_subset(self, fake_data_dirs):
        result = cs.run(snapshot_id="2026-07-24", tours=("atp",))
        assert (result / "processed" / "atp_features.csv").exists()
        assert not (result / "processed" / "wta_features.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_create_snapshot.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.create_snapshot'`.

- [ ] **Step 3: Create `scripts/create_snapshot.py`**

```python
"""
Create an immutable dataset snapshot under data/snapshots/{id}/.

Copies each tour's current data/raw/{...} files and
data/processed/{tour}_features.csv into data/snapshots/{id}/, then writes
metadata.json (last_match_date, row counts, per-file sha256 hashes) — see
docs/superpowers/specs/2026-07-24-snapshot-persistence-design.md.

Uso:
  python scripts/create_snapshot.py                   # id = hoy (YYYY-MM-DD)
  python scripts/create_snapshot.py --id 2026-07-24   # id explicito
  python scripts/create_snapshot.py --tours atp        # solo un tour

Requiere que data/processed/{tour}_features.csv ya exista para cada tour
solicitado (correr primero: python -m src.pipeline {tour}).
"""
import argparse
from pathlib import Path

from src.data.snapshots import create_snapshot, load_snapshot_metadata


def run(snapshot_id: str | None = None, tours: tuple[str, ...] = ("atp", "wta")) -> Path:
    snapshot_dir = create_snapshot(snapshot_id=snapshot_id, tours=tours)
    meta = load_snapshot_metadata(snapshot_dir.name)
    print(f"Snapshot creado: {snapshot_dir}")
    for tour, tour_meta in meta["tours"].items():
        print(f"  {tour.upper()}: {tour_meta['n_rows']:,} filas, "
              f"ultimo partido {tour_meta['last_match_date']}")
    return snapshot_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crear un snapshot inmutable de los datasets actuales (raw + procesado)"
    )
    parser.add_argument("--id", dest="snapshot_id", default=None,
                        help="Id del snapshot (default: fecha de hoy, YYYY-MM-DD)")
    parser.add_argument("--tours", nargs="+", default=["atp", "wta"],
                        choices=["atp", "wta"], help="Tours a incluir (default: ambos)")
    args = parser.parse_args()
    run(snapshot_id=args.snapshot_id, tours=tuple(args.tours))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_create_snapshot.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Manual smoke test**

Run: `./tenis-env/Scripts/python.exe scripts/create_snapshot.py --help`
Expected: usage text includes `--id` and `--tours`, exits 0, no traceback.

- [ ] **Step 6: Commit**

```bash
git add scripts/create_snapshot.py tests/test_create_snapshot.py
git commit -m "feat: add scripts/create_snapshot.py CLI wrapper"
```

---

## Task 8: `src/pipeline.py` — `--snapshot` flag

**Files:**
- Modify: `src/pipeline.py` (full file — small enough to show as one change)
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:

```python
"""Tests for src/pipeline.py's --snapshot support in run_pipeline()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.data.snapshots as snapshots
from src.pipeline import run_pipeline


@pytest.fixture()
def fake_snapshot(tmp_path, monkeypatch):
    """A minimal but structurally valid snapshot with a real, if tiny,
    processed features CSV — enough for walk_forward_backtest to run
    end-to-end without touching live data."""
    snapshot_root = tmp_path / "snapshots"
    monkeypatch.setattr(snapshots, "SNAPSHOT_ROOT", snapshot_root)
    snapshot_dir = snapshot_root / "2020-01-01"
    (snapshot_dir / "processed").mkdir(parents=True)

    rng = np.random.default_rng(3)
    n = 500
    elo_diff = rng.normal(0, 150, n)
    df = pd.DataFrame({
        "year": rng.integers(2000, 2015, n),
        "elo_diff": elo_diff,
        "elo_prob": 1 / (1 + 10 ** (-elo_diff / 400)),
        "rank_diff": rng.normal(0, 50, n),
        "form_diff": rng.uniform(-0.5, 0.5, n),
        "surface_form_diff": rng.uniform(-0.5, 0.5, n),
        "h2h_rate": rng.uniform(0.3, 0.7, n),
        "rest_diff": rng.normal(0, 5, n),
        "rolling_elo_diff": rng.normal(0, 30, n),
        "age_multiplier_diff": rng.uniform(-0.3, 0.3, n),
        "rust_factor_diff": rng.uniform(-0.5, 0.5, n),
        "adjusted_elo_diff": elo_diff * rng.uniform(0.6, 1.0, n),
        "outcome": 1,
        "is_mirror": False,
    })
    df.to_csv(snapshot_dir / "processed" / "atp_features.csv", index=False)
    (snapshot_dir / "metadata.json").write_text("{}")
    return snapshot_dir


class TestRunPipelineSnapshot:
    def test_reads_from_snapshot_instead_of_live_loader(self, fake_snapshot, monkeypatch, capsys):
        def _boom(*a, **kw):
            raise AssertionError("should not call the live loader when --snapshot is given")
        monkeypatch.setattr("src.pipeline.load_atp_matches", _boom)

        run_pipeline(tour="atp", warmup_years=5, snapshot="2020-01-01")

        out = capsys.readouterr().out
        assert "snapshot" in out.lower()
        assert "BACKTEST RESULTS" in out

    def test_does_not_write_to_live_processed_dir(self, fake_snapshot, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # the non-snapshot path writes to "data/processed" (cwd-relative)

        run_pipeline(tour="atp", warmup_years=5, snapshot="2020-01-01")

        assert not (tmp_path / "data" / "processed").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_pipeline.py -v`
Expected: `TypeError: run_pipeline() got an unexpected keyword argument 'snapshot'`.

- [ ] **Step 3: Add `--snapshot` support to `src/pipeline.py`**

Change the whole file:

```python
"""End-to-end pipeline: load data -> features -> backtest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.walkforward import build_match_features, walk_forward_backtest
from src.data.loader import load_atp_matches, load_wta_matches
from src.features.engineering import FeatureBuilder
from src.models.elo import EloSystem


def run_pipeline(
    tour: str = "atp",
    start_year: int = 1990,
    end_year: int = 2023,
    warmup_years: int = 10,
    eval_start: int = None,
) -> None:
    print(f"Loading {tour.upper()} matches {start_year}-{end_year}...")
    loader = load_atp_matches if tour == "atp" else load_wta_matches
    df = loader(start_year, end_year)
    print(f"  Loaded {len(df):,} matches across {df['year'].nunique()} seasons")

    elo = EloSystem()
    fb = FeatureBuilder()

    print("Building features (sequential, no lookahead)...")
    match_df = build_match_features(df, elo, fb)
    original = match_df[~match_df["is_mirror"]]
    print(f"  {len(original):,} feature rows ({len(match_df):,} with mirrors)")

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{tour}_features.csv"
    original.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    print(f"\nWalk-forward backtest (warmup={warmup_years} years)...")
    results = walk_forward_backtest(match_df, warmup_years=warmup_years)

    # Optional filter: only report years >= eval_start
    if eval_start is not None:
        results = {y: m for y, m in results.items() if y >= eval_start}
        print(f"  (reporting years >= {eval_start})")

    header = f"{'Year':>6} | {'Acc':>7} | {'LogLoss':>8} | {'Brier':>7} | {'EloAcc':>7} | {'EloLL':>8} | {'N':>6}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  BACKTEST RESULTS")
    print(sep)
    print(header)
    print("-" * len(header))
    for year, m in sorted(results.items()):
        print(
            f"{year:>6} | {m['accuracy']:>7.4f} | {m['log_loss']:>8.4f} | "
            f"{m['brier_score']:>7.4f} | {m['elo_only_accuracy']:>7.4f} | "
            f"{m['elo_only_log_loss']:>8.4f} | {m['n_matches']:>6}"
        )
    print(sep)

    accs = [m["accuracy"] for m in results.values()]
    elo_accs = [m["elo_only_accuracy"] for m in results.values()]
    lls = [m["log_loss"] for m in results.values()]
    elo_lls = [m["elo_only_log_loss"] for m in results.values()]
    print(f"\n  Mean Accuracy : {np.mean(accs):.4f}  (Elo-only: {np.mean(elo_accs):.4f})")
    print(f"  Mean Log-Loss : {np.mean(lls):.4f}  (Elo-only: {np.mean(elo_lls):.4f})")
    print(f"  Years tested  : {len(results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ia-tenis pipeline")
    parser.add_argument("tour",        nargs="?", default="atp")
    parser.add_argument("start_year",  nargs="?", type=int, default=1990)
    parser.add_argument("end_year",    nargs="?", type=int, default=2023)
    parser.add_argument("--warmup",    type=int,  default=10,   dest="warmup_years")
    parser.add_argument("--eval-start", type=int, default=None, dest="eval_start")
    args = parser.parse_args()

    run_pipeline(
        tour=args.tour,
        start_year=args.start_year,
        end_year=args.end_year,
        warmup_years=args.warmup_years,
        eval_start=args.eval_start,
    )
```

to:

```python
"""End-to-end pipeline: load data -> features -> backtest."""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.walkforward import build_match_features, load_features_with_mirror, walk_forward_backtest
from src.data.loader import load_atp_matches, load_wta_matches
from src.data.snapshots import resolve_snapshot_path
from src.features.engineering import FeatureBuilder
from src.models.elo import EloSystem


def run_pipeline(
    tour: str = "atp",
    start_year: int = 1990,
    end_year: int = 2023,
    warmup_years: int = 10,
    eval_start: int = None,
    snapshot: Optional[str] = None,
) -> None:
    if snapshot is not None:
        features_path = resolve_snapshot_path(snapshot, tour, "processed")
        print(f"Loading {tour.upper()} features from snapshot '{snapshot}' ({features_path})...")
        match_df = load_features_with_mirror(features_path)
        original = match_df[~match_df["is_mirror"]]
        print(f"  {len(original):,} feature rows ({len(match_df):,} with mirrors) — pinned, not regenerated.")
    else:
        print(f"Loading {tour.upper()} matches {start_year}-{end_year}...")
        loader = load_atp_matches if tour == "atp" else load_wta_matches
        df = loader(start_year, end_year)
        print(f"  Loaded {len(df):,} matches across {df['year'].nunique()} seasons")

        elo = EloSystem()
        fb = FeatureBuilder()

        print("Building features (sequential, no lookahead)...")
        match_df = build_match_features(df, elo, fb)
        original = match_df[~match_df["is_mirror"]]
        print(f"  {len(original):,} feature rows ({len(match_df):,} with mirrors)")

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / f"{tour}_features.csv"
        original.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")

    print(f"\nWalk-forward backtest (warmup={warmup_years} years)...")
    results = walk_forward_backtest(match_df, warmup_years=warmup_years)

    # Optional filter: only report years >= eval_start
    if eval_start is not None:
        results = {y: m for y, m in results.items() if y >= eval_start}
        print(f"  (reporting years >= {eval_start})")

    header = f"{'Year':>6} | {'Acc':>7} | {'LogLoss':>8} | {'Brier':>7} | {'EloAcc':>7} | {'EloLL':>8} | {'N':>6}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  BACKTEST RESULTS")
    print(sep)
    print(header)
    print("-" * len(header))
    for year, m in sorted(results.items()):
        print(
            f"{year:>6} | {m['accuracy']:>7.4f} | {m['log_loss']:>8.4f} | "
            f"{m['brier_score']:>7.4f} | {m['elo_only_accuracy']:>7.4f} | "
            f"{m['elo_only_log_loss']:>8.4f} | {m['n_matches']:>6}"
        )
    print(sep)

    accs = [m["accuracy"] for m in results.values()]
    elo_accs = [m["elo_only_accuracy"] for m in results.values()]
    lls = [m["log_loss"] for m in results.values()]
    elo_lls = [m["elo_only_log_loss"] for m in results.values()]
    print(f"\n  Mean Accuracy : {np.mean(accs):.4f}  (Elo-only: {np.mean(elo_accs):.4f})")
    print(f"  Mean Log-Loss : {np.mean(lls):.4f}  (Elo-only: {np.mean(elo_lls):.4f})")
    print(f"  Years tested  : {len(results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ia-tenis pipeline")
    parser.add_argument("tour",        nargs="?", default="atp")
    parser.add_argument("start_year",  nargs="?", type=int, default=1990)
    parser.add_argument("end_year",    nargs="?", type=int, default=2023)
    parser.add_argument("--warmup",    type=int,  default=10,   dest="warmup_years")
    parser.add_argument("--eval-start", type=int, default=None, dest="eval_start")
    parser.add_argument("--snapshot",  type=str,  default=None,
                        help="Usar un snapshot pinned (data/snapshots/{id}/) en vez de datos en vivo")
    args = parser.parse_args()

    run_pipeline(
        tour=args.tour,
        start_year=args.start_year,
        end_year=args.end_year,
        warmup_years=args.warmup_years,
        eval_start=args.eval_start,
        snapshot=args.snapshot,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_pipeline.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures.

- [ ] **Step 6: Manual smoke test**

Run: `./tenis-env/Scripts/python.exe -m src.pipeline --help`
Expected: usage text includes `--snapshot`, exits 0, no traceback.

- [ ] **Step 7: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add --snapshot flag to src/pipeline.py"
```

---

## Task 9: `src/data/loader.py` — raw-directory override

**Files:**
- Modify: `src/data/loader.py` (full file — small enough to show as one change)
- Modify: `tests/test_loader.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_loader.py`, change the import line:

```python
import pandas as pd
import pytest
from io import StringIO
from src.data.loader import _clean, _clean_wta
```

to:

```python
import pandas as pd
import pytest
from io import StringIO
from src.data.loader import _clean, _clean_wta, load_atp_matches, load_wta_matches
```

Append to the end of the file:

```python
# ── raw_dir_override (snapshot-pinned loading) ──────────────────────────────

class TestLoadAtpMatchesRawDirOverride:
    def test_reads_from_override_directory_instead_of_default(self, tmp_path):
        override_dir = tmp_path / "custom_atp"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(SAMPLE_CSV)

        df = load_atp_matches(2023, 2023, raw_dir_override=override_dir)

        assert len(df) == 2
        assert set(df["winner_name"]) == {"Djokovic", "Nadal"}

    def test_raises_when_override_directory_has_no_matching_year_files(self, tmp_path):
        override_dir = tmp_path / "empty_atp"
        override_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_atp_matches(2023, 2023, raw_dir_override=override_dir)


class TestLoadWtaMatchesRawDirOverride:
    def test_reads_from_override_directory_instead_of_default(self, tmp_path):
        override_dir = tmp_path / "custom_wta"
        override_dir.mkdir()
        (override_dir / "2023w.csv").write_text(WTA_CSV)

        df = load_wta_matches(2023, 2023, raw_dir_override=override_dir)

        assert len(df) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_loader.py::TestLoadAtpMatchesRawDirOverride tests/test_loader.py::TestLoadWtaMatchesRawDirOverride -v`
Expected: `TypeError: load_atp_matches() got an unexpected keyword argument 'raw_dir_override'`.

- [ ] **Step 3: Add `raw_dir_override` to both loaders in `src/data/loader.py`**

Change:

```python
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")
```

to:

```python
from pathlib import Path
from typing import Optional

import pandas as pd

RAW_DATA_DIR = Path("data/raw")
```

Change:

```python
def load_atp_matches(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    """Load ATP matches from Tennismylife/TML-Database (data/raw/tennis_atp_tml/{year}.csv)."""
    tour_dir = RAW_DATA_DIR / "tennis_atp_tml"
    frames = []
```

to:

```python
def load_atp_matches(
    start_year: int = 1990, end_year: int = 2024, raw_dir_override: Optional[Path] = None,
) -> pd.DataFrame:
    """Load ATP matches from Tennismylife/TML-Database (data/raw/tennis_atp_tml/{year}.csv).

    raw_dir_override: read from this directory instead of the default
    data/raw/tennis_atp_tml — used for snapshot-pinned loading
    (src/data/snapshots.py's resolve_snapshot_path), where the directory is
    a frozen copy under data/snapshots/{id}/raw/tennis_atp_tml.
    """
    tour_dir = raw_dir_override if raw_dir_override is not None else (RAW_DATA_DIR / "tennis_atp_tml")
    frames = []
```

Change:

```python
def load_wta_matches(start_year: int = 2007, end_year: int = 2024) -> pd.DataFrame:
    """Load WTA matches from tennis-data.co.uk (data/raw/tennis_wta_tduk/{year}w.[xls|xlsx|csv]).

    Download files from tennis-data.co.uk/wta.php and place them in
    data/raw/tennis_wta_tduk/.  Run scripts/download_data.py to automate.
    """
    tour_dir = RAW_DATA_DIR / "tennis_wta_tduk"
    frames = []
```

to:

```python
def load_wta_matches(
    start_year: int = 2007, end_year: int = 2024, raw_dir_override: Optional[Path] = None,
) -> pd.DataFrame:
    """Load WTA matches from tennis-data.co.uk (data/raw/tennis_wta_tduk/{year}w.[xls|xlsx|csv]).

    Download files from tennis-data.co.uk/wta.php and place them in
    data/raw/tennis_wta_tduk/.  Run scripts/download_data.py to automate.

    raw_dir_override: same as load_atp_matches's, for
    data/raw/tennis_wta_tduk — used for snapshot-pinned loading.
    """
    tour_dir = raw_dir_override if raw_dir_override is not None else (RAW_DATA_DIR / "tennis_wta_tduk")
    frames = []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_loader.py -v`
Expected: all tests in the file PASS (16 previous + 4 new = 20).

- [ ] **Step 5: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures.

- [ ] **Step 6: Commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: add raw_dir_override to load_atp_matches/load_wta_matches"
```

---

## Task 10: `src/value_analysis.py` — snapshot-aware `load_model()`

**Files:**
- Modify: `src/value_analysis.py` (imports, `_cache_path`, `_save_cache`, `_load_cache`, `_build_elo_fb`, `load_model`)
- Modify: `tests/test_value_analysis.py` (new test class)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_value_analysis.py`, after the `TestTrainLRPipeline` class (end of file):

```python
class TestLoadModelSnapshot:
    """load_model(snapshot=...) pins loading to data/snapshots/{id}/ instead
    of live data, uses a separate never-expiring cache file, and skips the
    staleness check (a deliberately old pinned dataset isn't 'stale')."""

    _ATP_ROW = (
        "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
        "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
        "winner_ht,winner_ioc,winner_age,winner_rank,winner_rank_points,"
        "loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,"
        "loser_ioc,loser_age,loser_rank,loser_rank_points,score,best_of,round,minutes\n"
        "2020-1,AO,Hard,128,G,20200115,1,101,,,PlayerA,R,188,USA,25.0,1,10000,"
        "102,,,PlayerB,R,190,USA,26.0,2,9000,6-3 6-4,3,R32,85\n"
    )

    @staticmethod
    def _write_features_csv(path):
        rng = np.random.default_rng(11)
        n = 300
        elo_diff = rng.normal(0, 150, n)
        pd.DataFrame({
            "year": rng.integers(2000, 2015, n),
            "elo_diff": elo_diff,
            "elo_prob": 1 / (1 + 10 ** (-elo_diff / 400)),
            "rank_diff": rng.normal(0, 50, n),
            "form_diff": rng.uniform(-0.5, 0.5, n),
            "surface_form_diff": rng.uniform(-0.5, 0.5, n),
            "h2h_rate": rng.uniform(0.3, 0.7, n),
            "rest_diff": rng.normal(0, 5, n),
            "rolling_elo_diff": rng.normal(0, 30, n),
            "age_multiplier_diff": rng.uniform(-0.3, 0.3, n),
            "rust_factor_diff": rng.uniform(-0.5, 0.5, n),
            "adjusted_elo_diff": elo_diff * rng.uniform(0.6, 1.0, n),
            "outcome": 1,
            "is_mirror": False,
        }).to_csv(path, index=False)

    @pytest.fixture()
    def fake_snapshot(self, tmp_path, monkeypatch):
        import src.data.snapshots as snap_mod
        import src.value_analysis as va
        monkeypatch.setattr(snap_mod, "SNAPSHOT_ROOT", tmp_path / "snapshots")
        monkeypatch.setattr(va, "_CACHE_DIR", tmp_path / "model_cache")

        snapshot_dir = tmp_path / "snapshots" / "2020-01-01"
        raw_dir = snapshot_dir / "raw" / "tennis_atp_tml"
        processed_dir = snapshot_dir / "processed"
        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)

        (raw_dir / "2020.csv").write_text(self._ATP_ROW)
        self._write_features_csv(processed_dir / "atp_features.csv")
        (snapshot_dir / "metadata.json").write_text("{}")
        return "2020-01-01"

    def test_builds_from_snapshot_and_caches_separately(self, fake_snapshot):
        import src.value_analysis as va
        elo, fb, clf, rank_lookup, elo_tracker, age_lookup = va.load_model("atp", snapshot=fake_snapshot)

        assert "PlayerA" in elo.general_ratings
        assert va._cache_path("atp", fake_snapshot).exists()
        assert not va._cache_path("atp").exists()  # live cache untouched

    def test_snapshot_cache_never_expires(self, fake_snapshot):
        import pickle
        from datetime import datetime, timedelta
        import src.value_analysis as va

        va.load_model("atp", snapshot=fake_snapshot)  # first build, writes cache
        path = va._cache_path("atp", fake_snapshot)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        payload["timestamp"] = datetime.now() - timedelta(days=999)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        result = va._load_cache("atp", fake_snapshot)
        assert result is not None  # not treated as expired

    def test_skips_staleness_check_under_snapshot(self, fake_snapshot, capsys):
        import src.value_analysis as va
        va.load_model("atp", snapshot=fake_snapshot)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" not in out
        assert "CRITICO" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestLoadModelSnapshot -v`
Expected: `TypeError: load_model() got an unexpected keyword argument 'snapshot'`.

- [ ] **Step 3: Add the import**

Change:

```python
from src.backtest.walkforward import _MIRROR_FLIP_COLS, load_features_with_mirror
from src.data.staleness import StalenessLevel, evaluate_staleness
```

to:

```python
from src.backtest.walkforward import _MIRROR_FLIP_COLS, load_features_with_mirror
from src.data.snapshots import resolve_snapshot_path
from src.data.staleness import StalenessLevel, evaluate_staleness
```

- [ ] **Step 4: Add a `snapshot` dimension to `_cache_path`**

Change:

```python
def _cache_path(tour: str) -> Path:
    return _CACHE_DIR / f"{tour}.pkl"
```

to:

```python
def _cache_path(tour: str, snapshot: Optional[str] = None) -> Path:
    if snapshot is not None:
        return _CACHE_DIR / f"{tour}__snapshot-{snapshot}.pkl"
    return _CACHE_DIR / f"{tour}.pkl"
```

- [ ] **Step 5: Thread `snapshot` through `_save_cache` and `_load_cache`**

Change:

```python
def _save_cache(
    tour: str, elo, fb, clf, rank_lookup: dict, last_match_date: date,
    elo_tracker: EloHistoryTracker, age_lookup: dict,
) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "elo":              elo,
        "fb":               fb,
        "clf":              clf,
        "rank_lookup":      rank_lookup,
        "timestamp":        datetime.now(),
        "tour":             tour,
        "last_match_date":  last_match_date,
        "elo_tracker":      elo_tracker,
        "age_lookup":       age_lookup,
    }
    with open(_cache_path(tour), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache guardado en {_cache_path(tour)}")
```

to:

```python
def _save_cache(
    tour: str, elo, fb, clf, rank_lookup: dict, last_match_date: date,
    elo_tracker: EloHistoryTracker, age_lookup: dict, snapshot: Optional[str] = None,
) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "elo":              elo,
        "fb":               fb,
        "clf":              clf,
        "rank_lookup":      rank_lookup,
        "timestamp":        datetime.now(),
        "tour":             tour,
        "last_match_date":  last_match_date,
        "elo_tracker":      elo_tracker,
        "age_lookup":       age_lookup,
    }
    path = _cache_path(tour, snapshot)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache guardado en {path}")
```

Change:

```python
def _load_cache(tour: str):
    """Return (elo, fb, clf, rank_lookup, last_match_date, elo_tracker,
    age_lookup) from cache, or None if stale/missing."""
    path = _cache_path(tour)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f"  Advertencia: no se pudo leer cache ({e}). Reentrenando...")
        return None

    if "last_match_date" not in payload:
        print("  Cache de formato antiguo (sin last_match_date). Reentrenando...")
        return None

    age = datetime.now() - payload["timestamp"]
    if age >= timedelta(days=CACHE_MAX_AGE_DAYS):
        print(f"  Cache expirado ({age.days}d {age.seconds//3600}h). Reentrenando...")
        return None

    age_str = (f"{age.days}d " if age.days else "") + f"{age.seconds//3600}h {(age.seconds%3600)//60}m"
    print(f"  Cache cargado ({age_str} de antiguedad — maximo {CACHE_MAX_AGE_DAYS}d).")
    return (
        payload["elo"], payload["fb"], payload["clf"],
        payload["rank_lookup"], payload["last_match_date"],
        payload.get("elo_tracker") or EloHistoryTracker(),
        payload.get("age_lookup") or {},
    )
```

to:

```python
def _load_cache(tour: str, snapshot: Optional[str] = None):
    """Return (elo, fb, clf, rank_lookup, last_match_date, elo_tracker,
    age_lookup) from cache, or None if stale/missing.

    When snapshot is set, the cache never expires (CACHE_MAX_AGE_DAYS is
    ignored) — a snapshot-pinned cache reflects an immutable dataset, so
    there's nothing for it to go stale relative to.
    """
    path = _cache_path(tour, snapshot)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f"  Advertencia: no se pudo leer cache ({e}). Reentrenando...")
        return None

    if "last_match_date" not in payload:
        print("  Cache de formato antiguo (sin last_match_date). Reentrenando...")
        return None

    if snapshot is None:
        age = datetime.now() - payload["timestamp"]
        if age >= timedelta(days=CACHE_MAX_AGE_DAYS):
            print(f"  Cache expirado ({age.days}d {age.seconds//3600}h). Reentrenando...")
            return None
        age_str = (f"{age.days}d " if age.days else "") + f"{age.seconds//3600}h {(age.seconds%3600)//60}m"
        print(f"  Cache cargado ({age_str} de antiguedad — maximo {CACHE_MAX_AGE_DAYS}d).")
    else:
        print(f"  Cache de snapshot '{snapshot}' cargado (sin expiracion).")

    return (
        payload["elo"], payload["fb"], payload["clf"],
        payload["rank_lookup"], payload["last_match_date"],
        payload.get("elo_tracker") or EloHistoryTracker(),
        payload.get("age_lookup") or {},
    )
```

- [ ] **Step 6: Add `raw_dir_override` to `_build_elo_fb`**

Change:

```python
def _build_elo_fb(tour: str):
    """Rebuild EloSystem + FeatureBuilder from raw matches.

    Returns (elo, fb, rank_lookup, last_match_date, elo_tracker, age_lookup)
    where rank_lookup maps player names to their most recently observed
    ATP/WTA ranking, last_match_date is the most recent match_date seen in
    the raw data, elo_tracker holds each player's rolling pre-match Elo
    snapshots (see src/features/decay.py), and age_lookup maps player names
    to their most recently observed age + the date it was observed (empty
    for WTA — tennis-data.co.uk has no age column).
    """
    from src.data.loader import load_atp_matches, load_wta_matches
    from src.features.engineering import FeatureBuilder
    from src.models.elo import EloSystem
    from src.backtest.walkforward import build_match_features

    print(f"  Reconstruyendo Elo + FeatureBuilder ({tour.upper()})...")
    loader   = load_atp_matches if tour == "atp" else load_wta_matches
    start    = 1990 if tour == "atp" else 2007
    end      = date.today().year
    df_raw   = loader(start, end)
```

to:

```python
def _build_elo_fb(tour: str, raw_dir_override: Optional[Path] = None):
    """Rebuild EloSystem + FeatureBuilder from raw matches.

    Returns (elo, fb, rank_lookup, last_match_date, elo_tracker, age_lookup)
    where rank_lookup maps player names to their most recently observed
    ATP/WTA ranking, last_match_date is the most recent match_date seen in
    the raw data, elo_tracker holds each player's rolling pre-match Elo
    snapshots (see src/features/decay.py), and age_lookup maps player names
    to their most recently observed age + the date it was observed (empty
    for WTA — tennis-data.co.uk has no age column).

    raw_dir_override: read raw matches from here instead of the live
    data/raw/{tour dir} — used for snapshot-pinned loading
    (load_model(snapshot=...)).
    """
    from src.data.loader import load_atp_matches, load_wta_matches
    from src.features.engineering import FeatureBuilder
    from src.models.elo import EloSystem
    from src.backtest.walkforward import build_match_features

    print(f"  Reconstruyendo Elo + FeatureBuilder ({tour.upper()})...")
    loader   = load_atp_matches if tour == "atp" else load_wta_matches
    start    = 1990 if tour == "atp" else 2007
    end      = date.today().year
    df_raw   = loader(start, end, raw_dir_override=raw_dir_override)
```

- [ ] **Step 7: Add `snapshot` support to `load_model`**

Change:

```python
def load_model(tour: str = "atp", retrain: bool = False):
    """Load or rebuild model, with transparent cache management.

    Returns:
        elo          – EloSystem with full historical state
        fb           – FeatureBuilder with full historical state
        clf          – Pipeline(StandardScaler, LogisticRegression) trained on all available data
        rank_lookup  – dict {player_name: most_recent_rank}
        elo_tracker  – EloHistoryTracker (rolling pre-match Elo snapshots)
        age_lookup   – dict {player_name: (age, observed_date)}, empty for WTA
    """
    if not retrain:
        cached = _load_cache(tour)
        if cached is not None:
            elo, fb, clf, rank_lookup, last_match_date, elo_tracker, age_lookup = cached
            _check_staleness(tour, last_match_date)
            return elo, fb, clf, rank_lookup, elo_tracker, age_lookup

    print(f"Construyendo modelo desde cero ({tour.upper()}) — primera vez ~30-60 s...")
    elo, fb, rank_lkp, last_match_date, elo_tracker, age_lkp = _build_elo_fb(tour)
    clf               = _train_lr(tour)
    _save_cache(tour, elo, fb, clf, rank_lkp, last_match_date, elo_tracker, age_lkp)
    _check_staleness(tour, last_match_date)
    return elo, fb, clf, rank_lkp, elo_tracker, age_lkp
```

to:

```python
def load_model(tour: str = "atp", retrain: bool = False, snapshot: Optional[str] = None):
    """Load or rebuild model, with transparent cache management.

    snapshot: when set, pin loading to data/snapshots/{snapshot}/ instead of
    live data/raw/ + data/processed/. Uses its own cache file
    ({tour}__snapshot-{id}.pkl, no expiry — an immutable snapshot can't go
    stale relative to a cache-age clock) and skips the staleness check
    entirely, since a deliberately old, pinned dataset isn't "stale" — that
    it's old relative to today is the whole point of asking for it.

    Returns:
        elo          – EloSystem with full historical state
        fb           – FeatureBuilder with full historical state
        clf          – Pipeline(StandardScaler, LogisticRegression) trained on all available data
        rank_lookup  – dict {player_name: most_recent_rank}
        elo_tracker  – EloHistoryTracker (rolling pre-match Elo snapshots)
        age_lookup   – dict {player_name: (age, observed_date)}, empty for WTA
    """
    if not retrain:
        cached = _load_cache(tour, snapshot)
        if cached is not None:
            elo, fb, clf, rank_lookup, last_match_date, elo_tracker, age_lookup = cached
            if snapshot is None:
                _check_staleness(tour, last_match_date)
            return elo, fb, clf, rank_lookup, elo_tracker, age_lookup

    if snapshot is not None:
        print(f"Construyendo modelo pinned a snapshot '{snapshot}' ({tour.upper()})...")
        raw_dir       = resolve_snapshot_path(snapshot, tour, "raw")
        features_path = resolve_snapshot_path(snapshot, tour, "processed")
    else:
        print(f"Construyendo modelo desde cero ({tour.upper()}) — primera vez ~30-60 s...")
        raw_dir       = None
        features_path = None

    elo, fb, rank_lkp, last_match_date, elo_tracker, age_lkp = _build_elo_fb(tour, raw_dir_override=raw_dir)
    clf = _train_lr(tour, features_path=features_path)
    _save_cache(tour, elo, fb, clf, rank_lkp, last_match_date, elo_tracker, age_lkp, snapshot=snapshot)
    if snapshot is None:
        _check_staleness(tour, last_match_date)
    return elo, fb, clf, rank_lkp, elo_tracker, age_lkp
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestLoadModelSnapshot -v`
Expected: all 3 tests PASS.

- [ ] **Step 9: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures, no warnings.

- [ ] **Step 10: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py
git commit -m "feat: add snapshot-pinned model loading to load_model()"
```

---

## Task 11: `--snapshot` CLI flag on `src/value_analysis.py`

**Files:**
- Modify: `src/value_analysis.py` (`interactive_cli`, `main`)

No new automated tests — `interactive_cli` is an `input()`-driven REPL with no existing test coverage (consistent with the rest of the file, see the 2026-07-08 staleness-halt plan's Task 4 for precedent); verification is the full suite (regression guard) plus manual smoke tests.

- [ ] **Step 1: Add the import**

Change:

```python
from src.data.snapshots import resolve_snapshot_path
```

to:

```python
from src.data.snapshots import load_snapshot_metadata, resolve_snapshot_path
```

- [ ] **Step 2: Add `snapshot` parameter and pinned-dataset notice to `interactive_cli`**

Change:

```python
def interactive_cli(tour: str = "atp", retrain: bool = False, halt_on_suspicious: bool = False) -> None:
    """Run the interactive CLI session."""
    elo, fb, clf, rank_lookup, elo_tracker, age_lookup = load_model(tour, retrain=retrain)

    print()
    print("=" * 64)
    print(f"  ia-tenis  Value Bet Analyzer  ({tour.upper()})")
    print("=" * 64)
    print("  Escribe 'q' en cualquier prompt para salir.")
    print()
```

to:

```python
def interactive_cli(
    tour: str = "atp", retrain: bool = False, halt_on_suspicious: bool = False,
    snapshot: Optional[str] = None,
) -> None:
    """Run the interactive CLI session."""
    elo, fb, clf, rank_lookup, elo_tracker, age_lookup = load_model(tour, retrain=retrain, snapshot=snapshot)

    if snapshot is not None:
        meta = load_snapshot_metadata(snapshot)
        last_match = meta["tours"][tour]["last_match_date"]
        print(f"\n  *** Usando snapshot pinned '{snapshot}' — datos como al {last_match}. ***")

    print()
    print("=" * 64)
    print(f"  ia-tenis  Value Bet Analyzer  ({tour.upper()})")
    print("=" * 64)
    print("  Escribe 'q' en cualquier prompt para salir.")
    print()
```

- [ ] **Step 3: Add the flag to `main()`**

Change:

```python
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tennis value-bet analyzer — manual odds input"
    )
    parser.add_argument("--wta", action="store_true",
                        help="Usar modelo WTA en lugar de ATP (default)")
    parser.add_argument("--retrain", action="store_true",
                        help="Ignorar cache y reentrenar modelo desde cero")
    parser.add_argument("--halt-on-suspicious", action="store_true",
                        dest="halt_on_suspicious",
                        help="Bloquea el guardado en log si el edge supera "
                             f"{SUSPICIOUS_EDGE_THRESHOLD*100:.0f}%% (posible dato stale)")
    args = parser.parse_args()
    tour = "wta" if args.wta else "atp"
    interactive_cli(tour, retrain=args.retrain, halt_on_suspicious=args.halt_on_suspicious)
```

to:

```python
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tennis value-bet analyzer — manual odds input"
    )
    parser.add_argument("--wta", action="store_true",
                        help="Usar modelo WTA en lugar de ATP (default)")
    parser.add_argument("--retrain", action="store_true",
                        help="Ignorar cache y reentrenar modelo desde cero")
    parser.add_argument("--halt-on-suspicious", action="store_true",
                        dest="halt_on_suspicious",
                        help="Bloquea el guardado en log si el edge supera "
                             f"{SUSPICIOUS_EDGE_THRESHOLD*100:.0f}%% (posible dato stale)")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Usar un snapshot pinned (data/snapshots/{id}/) en vez de datos en vivo")
    args = parser.parse_args()
    tour = "wta" if args.wta else "atp"
    interactive_cli(
        tour, retrain=args.retrain, halt_on_suspicious=args.halt_on_suspicious,
        snapshot=args.snapshot,
    )
```

- [ ] **Step 4: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures, no warnings.

- [ ] **Step 5: Manual smoke tests**

Run: `./tenis-env/Scripts/python.exe -m src.value_analysis --help`
Expected: usage text includes `--snapshot` with its help string, exits 0, no traceback.

Run: `./tenis-env/Scripts/python.exe -c "import src.value_analysis"`
Expected: succeeds with no traceback (import-time sanity check after the edits).

- [ ] **Step 6: Commit**

```bash
git add src/value_analysis.py
git commit -m "feat: add --snapshot flag to src/value_analysis.py CLI"
```

---

## Task 12: Documentation

**Files:**
- Modify: `docs/runbooks/data-refresh-and-staleness.md` (append new section)

- [ ] **Step 1: Add a "Dataset snapshots" section**

Append to the end of `docs/runbooks/data-refresh-and-staleness.md` (after section 6, "Suspicious edges"):

```markdown

## 7. Dataset snapshots (reproducibility)

Full design: `docs/superpowers/specs/2026-07-24-snapshot-persistence-design.md`.

`tennis-data.co.uk`/TML can silently revise historical data (see
[[project-tennis-data-couk-silent-historical-updates]] in project memory) —
a backtest or a live prediction run today against `data/raw/`/`data/processed/`
is not guaranteed to be reproducible next month. A snapshot freezes a copy.

**Create a snapshot** (requires `data/processed/{tour}_features.csv` to
already exist for each tour — run `python -m src.pipeline {tour}` first if not):

```bash
./tenis-env/Scripts/python.exe scripts/create_snapshot.py                  # id = today
./tenis-env/Scripts/python.exe scripts/create_snapshot.py --id 2026-07-24  # explicit id
./tenis-env/Scripts/python.exe scripts/create_snapshot.py --tours atp      # one tour only
```

This copies both `data/raw/{tour dir}/` and `data/processed/{tour}_features.csv`
into `data/snapshots/{id}/`, and writes `data/snapshots/{id}/metadata.json`
(per-tour `last_match_date`, row count, and a sha256+byte-count per file).

**Run a backtest against a pinned snapshot** instead of live data:

```bash
./tenis-env/Scripts/python.exe -m src.pipeline atp --snapshot 2026-07-24
```

This skips loading/rebuilding features entirely — it reads the snapshot's
`{tour}_features.csv` directly and does **not** touch the live
`data/processed/{tour}_features.csv`.

**Load a value-bet model pinned to a snapshot**:

```bash
./tenis-env/Scripts/python.exe -m src.value_analysis --snapshot 2026-07-24
./tenis-env/Scripts/python.exe -m src.value_analysis --wta --snapshot 2026-07-24
```

A snapshot-pinned model uses its own cache file
(`data/model_cache/{tour}__snapshot-{id}.pkl`) that never expires — it
can't collide with or be evicted by a live `--retrain`. The usual staleness
warning (section 1) is skipped under `--snapshot`: a deliberately old,
pinned dataset isn't "stale" in the sense that warning exists to catch.

**What's tracked in git:** only `data/snapshots/{id}/metadata.json` — the
copied CSV/xlsx files themselves are gitignored, same as `data/raw/` and
`data/processed/` already are (see `.gitignore`). This keeps a permanent,
lightweight, diffable record of every snapshot ever taken without
versioning tens of MB of data per snapshot.

**Verify a snapshot hasn't been locally corrupted or tampered with:**

```bash
./tenis-env/Scripts/python.exe -c "
from src.data.snapshots import verify_snapshot_integrity
print(verify_snapshot_integrity('2026-07-24'))
"
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/data-refresh-and-staleness.md
git commit -m "docs: add dataset snapshots section to data-refresh runbook"
```

---

## Task 13: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -v`
Expected: every test passes (213 pre-existing + ~4 Task 1 + 4 Task 2 + 8 Task 3 + 5 Task 4 + 7 Task 5 + 3 Task 7 + 2 Task 8 + 4 Task 9 + 3 Task 10 ≈ 253 total), zero failures, zero warnings (in particular, no new `sklearn` feature-name warnings — every new `.fit()`/`.predict_proba()` call site added by this plan uses `.values` or a pre-built ndarray, consistent with the existing convention).

- [ ] **Step 2: Confirm `.gitignore` behavior once more against a real (throwaway) snapshot**

Run:
```bash
./tenis-env/Scripts/python.exe -m src.pipeline atp 1990 2026
./tenis-env/Scripts/python.exe scripts/create_snapshot.py --id smoke-test --tours atp
git status
```
Expected: `git status` shows `data/snapshots/smoke-test/metadata.json` as untracked (ready to `git add`), and does **not** list anything under `data/snapshots/smoke-test/processed/` or `data/snapshots/smoke-test/raw/`.

Clean up the throwaway snapshot afterward:
```bash
rm -rf data/snapshots/smoke-test
```

No commit for this task — it's verification only, confirming Tasks 1-12 integrate correctly end to end.

---

## Self-Review

### Spec Coverage

| Spec requirement (`docs/superpowers/specs/2026-07-24-snapshot-persistence-design.md`) | Covered by |
|---|---|
| `data/snapshots/{date}/metadata.json` with `last_match_date`, `n_rows`, per-file `sha256`+`bytes` | Task 3 |
| `metadata.json` also includes `git_commit`, `created_at` | Task 3 |
| Snapshot includes raw + processed (Q1: confirmed (b)) | Task 3 (`create_snapshot` copies both) |
| `metadata.json` committed, data files gitignored (Q2) | Task 6 |
| Separate snapshot cache file, no expiry (Q3) | Task 10 Step 5 |
| Skip staleness check under `--snapshot` (Q4) | Task 10 Step 7 |
| New `scripts/create_snapshot.py`, not a pipeline subcommand (Q5) | Task 7 |
| `--snapshot` on `src/pipeline.py` for backtest reproducibility | Task 8 |
| `--snapshot` on `src/value_analysis.py` for pinned model loading | Task 9, Task 10, Task 11 |
| Snapshots immutable (no silent overwrite) | Task 3 (`SnapshotExistsError`) |
| Integrity verification (hashes are checkable, not decorative) | Task 5 (`verify_snapshot_integrity`) |
| No raw-data reproducibility gap for live models (the central finding in the spec) | Task 9 (`raw_dir_override`) + Task 10 (`_build_elo_fb` uses it) |

### Placeholder Scan

No TBDs, no "add appropriate error handling," no "similar to Task N" — every step shows complete code.

### Type Consistency

- `load_features_with_mirror(path: Path) -> pd.DataFrame` (Task 1) — same signature used by `_train_lr` (Task 1 Step 7) and `run_pipeline` (Task 8 Step 3).
- `_train_lr(tour: str, features_path: Optional[Path] = None) -> Pipeline` (Task 1) — matches its call in `load_model` (Task 10 Step 7: `_train_lr(tour, features_path=features_path)`).
- `create_snapshot(snapshot_id: str | None = None, tours: tuple[str, ...] = ("atp", "wta")) -> Path` (Task 3) — matches `scripts/create_snapshot.py`'s `run()` (Task 7) and every test call across Tasks 3, 7.
- `resolve_snapshot_path(snapshot_id: str, tour: str, kind: Literal["processed", "raw"]) -> Path` (Task 5) — matches its calls in `run_pipeline` (Task 8: `kind="processed"`) and `load_model` (Task 10 Step 7: both `kind="raw"` and `kind="processed"`).
- `load_atp_matches(start_year, end_year, raw_dir_override: Optional[Path] = None)` / `load_wta_matches(...)` (Task 9) — matches `_build_elo_fb`'s call (Task 10 Step 6: `loader(start, end, raw_dir_override=raw_dir_override)`).
- `_build_elo_fb(tour: str, raw_dir_override: Optional[Path] = None)` (Task 10 Step 6) — matches its call in `load_model` (Task 10 Step 7).
- `_cache_path(tour: str, snapshot: Optional[str] = None) -> Path` (Task 10 Step 4) — matches every call site updated in Task 10 Steps 5, 7 and the new tests in Task 10 Step 1.
- `_save_cache(..., snapshot: Optional[str] = None)` / `_load_cache(tour, snapshot: Optional[str] = None)` (Task 10 Step 5) — matches `load_model`'s calls (Task 10 Step 7).
- `load_model(tour, retrain, snapshot: Optional[str] = None)` (Task 10 Step 7) — matches `interactive_cli`'s call (Task 11 Step 2) and every test call (Task 10 Step 1).
- `interactive_cli(tour, retrain, halt_on_suspicious, snapshot: Optional[str] = None)` (Task 11 Step 2) — matches `main()`'s call (Task 11 Step 3).

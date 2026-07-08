# TML ATP Data Source Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the frozen `git clone Tennismylife/TML-Database` ATP download path in `scripts/download_data.py` with the active `stats.tennismylife.org` HTTP API, merging the live `ongoing_tourneys.csv` feed into the current year's file, with zero changes to `src/data/loader.py` or anything downstream.

**Architecture:** All new code lives in `scripts/download_data.py` as a set of small, independently-testable pure functions (manifest filter, skip-decision, atomic download+validate, merge/dedup, legacy cleanup) wired together by one orchestration function, `download_atp()`, called from the existing `download()` entry point in place of the old git-clone block. WTA code is untouched.

**Tech Stack:** Python 3.14, `requests` + `certifi` (explicit choice over `urllib` — see rationale in Task 1), `pandas`, `pytest` with `monkeypatch`.

Reference spec: `docs/superpowers/specs/2026-07-08-tml-api-migration-design.md`

---

## Shared context for every task below

- **File under modification:** `D:\ia-tenis\scripts\download_data.py`. Current content (as of this plan) starts with a module docstring, then `DATA_RAW`, `ATP_REPO` (dict with `dest`/`url` for the git clone — this gets **deleted**), `WTA_START_YEAR`, `WTA_DIR`, `_WTA_HEADERS`, format-detection helpers, `_download_wta_year`, and `download()`. **None of the WTA code changes in this plan.**
- **Test file:** `D:\ia-tenis\tests\test_download_data.py` — does not exist yet, created in Task 1.
- **Run tests with:** `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py -v` (Windows path; the repo's venv is at `tenis-env/`).
- **Full suite regression check** (used in the final task): `./tenis-env/Scripts/python.exe -m pytest -v`.
- **The 50-column ATP schema** (captured from the live API and confirmed identical to the local frozen-clone `2025.csv` during design): see `ATP_SCHEMA_COLUMNS` in Task 1 — every task after Task 1 references this exact constant, do not redefine it elsewhere.
- **Why `requests` + `certifi` instead of `urllib`** (used by the existing WTA code in this same file): Python's `ssl` module on Windows can fall back to the OS certificate store when no explicit CA bundle is given, which routes through Windows' schannel and has shown flaky revocation-check behavior in prior diagnostics on this machine. `requests`, when `certifi` is installed (it is, transitively, but this plan makes it a direct dependency in Task 1), defaults `verify=True` to `certifi.where()` — a bundled CA file — never touching the OS store. This is a deliberate choice, not an oversight; don't "simplify" it back to `urllib` in review. Under no circumstance pass `verify=False` anywhere in this file.

---

### Task 1: Manifest fetch + filter, and the ATP schema constant

**Files:**
- Modify: `requirements.txt`
- Modify: `scripts/download_data.py`
- Test: `tests/test_download_data.py` (new)

- [ ] **Step 1: Add `requests` as a direct dependency**

Open `requirements.txt` and add this line (anywhere in the file, alphabetical position not required by existing style — the file isn't sorted):

```
requests>=2.34.0
```

`certifi` is not added explicitly — it installs automatically as a dependency of `requests`.

- [ ] **Step 2: Write the failing tests for manifest filtering**

Create `tests/test_download_data.py`:

```python
from __future__ import annotations

import datetime

import pandas as pd

from scripts import download_data as dd


class TestFilterAtpManifest:
    def test_keeps_year_files_and_ongoing(self):
        files = [
            {"name": "1990.csv", "url": "u", "size": 100, "mtime": "t"},
            {"name": "2026.csv", "url": "u", "size": 200, "mtime": "t"},
            {"name": "ongoing_tourneys.csv", "url": "u", "size": 300, "mtime": "t"},
        ]
        result = dd._filter_atp_manifest(files)
        assert [f["name"] for f in result] == [
            "1990.csv", "2026.csv", "ongoing_tourneys.csv",
        ]

    def test_drops_challenger_quali_and_aggregate_files(self):
        files = [
            {"name": "2026_challenger.csv", "url": "u", "size": 1, "mtime": "t"},
            {"name": "atp_quali/2026_atp_quali.csv", "url": "u", "size": 1, "mtime": "t"},
            {"name": "challenger_ongoing_tourneys.csv", "url": "u", "size": 1, "mtime": "t"},
            {"name": "ATP_Database.csv", "url": "u", "size": 1, "mtime": "t"},
        ]
        assert dd._filter_atp_manifest(files) == []

    def test_schema_constant_has_fifty_columns(self):
        assert len(dd.ATP_SCHEMA_COLUMNS) == 50
        assert dd.ATP_SCHEMA_COLUMNS[0] == "tourney_id"
        assert dd.ATP_SCHEMA_COLUMNS[-1] == "l_bpFaced"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py -v`
Expected: FAIL with `AttributeError: module 'scripts.download_data' has no attribute '_filter_atp_manifest'` (or `ATP_SCHEMA_COLUMNS`).

- [ ] **Step 4: Implement the schema constant and manifest filter**

In `scripts/download_data.py`, replace the top of the file (imports through `ATP_REPO`) with:

```python
"""
Download tennis datasets.

ATP  → stats.tennismylife.org API  (public)  → data/raw/tennis_atp_tml/{year}.csv
       Manifest: GET https://stats.tennismylife.org/api/data-files
       Uses `requests` + certifi's bundled CA bundle explicitly (NOT urllib's
       OS-trust-store default) — on Windows, urllib's default SSL context can
       fall back to the OS certificate store (schannel), which has shown
       flaky revocation-check behavior in prior diagnostics. requests, with
       certifi installed, verifies against a bundled root CA list instead,
       sidestepping that store entirely. Do not swap this back to urllib.
WTA  → tennis-data.co.uk           (public)  → data/raw/tennis_wta_tduk/{year}w.xls
       URL pattern: http://www.tennis-data.co.uk/{year}w/{year}w.xls
       Available from 2007.  Verify exact URLs at tennis-data.co.uk/wta.php if
       downloads fail (the site occasionally restructures paths between seasons).
"""
import datetime
import re
import shutil
import urllib.request
from pathlib import Path

import pandas as pd
import requests

DATA_RAW = Path("data/raw")

# ── ATP ──────────────────────────────────────────────────────────────────────
ATP_DIR = DATA_RAW / "tennis_atp_tml"
ATP_MANIFEST_URL = "https://stats.tennismylife.org/api/data-files"

ATP_SCHEMA_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "indoor",
    "tourney_date", "match_num", "winner_id", "winner_seed", "winner_entry", "winner_name",
    "winner_hand", "winner_ht", "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand", "loser_ht", "loser_ioc",
    "loser_age", "loser_rank", "loser_rank_points", "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved",
    "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced",
]

_ATP_YEAR_FILE_RE = re.compile(r"^\d{4}\.csv$")
_ATP_MERGE_KEY = ["tourney_id", "tourney_name", "round", "match_num"]
_ATP_LEGACY_PATHS = [
    ".git", ".github", "README.md", "logo.jpg", "ATP_Database.csv", "ongoing_tourneys.csv",
]


def _filter_atp_manifest(files: list[dict]) -> list[dict]:
    """Keep only main-tour year files and the live ongoing-tourneys feed.

    Drops challenger files, qualifying-draw files, and the ATP_Database.csv
    aggregate — none of these are read by src/data/loader.py.
    """
    return [
        f for f in files
        if _ATP_YEAR_FILE_RE.match(f["name"]) or f["name"] == "ongoing_tourneys.csv"
    ]


def _fetch_atp_manifest() -> list[dict]:
    """GET the file manifest from the API and return the filtered file list."""
    resp = requests.get(ATP_MANIFEST_URL, timeout=30)
    resp.raise_for_status()
    return _filter_atp_manifest(resp.json()["files"])
```

Delete the old `ATP_REPO = {...}` dict entirely — it's replaced by `ATP_DIR` and `ATP_MANIFEST_URL` above. Leave everything from `WTA_START_YEAR` onward untouched for now (the `import subprocess` that was at the top is also removed here since nothing in the file will use it after Task 6 removes the git-clone block in `download()` — if `subprocess` is still referenced later in this task's diff by leftover WTA code, keep it for now and remove in Task 6 instead; check with `grep -n subprocess scripts/download_data.py` before deleting the import).

- [ ] **Step 5: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Install requests into the venv and commit**

```bash
./tenis-env/Scripts/python.exe -m pip install -r requirements.txt
git add requirements.txt scripts/download_data.py tests/test_download_data.py
git commit -m "feat: add ATP manifest fetch/filter for stats.tennismylife.org API"
```

---

### Task 2: Skip-download decision

**Files:**
- Modify: `scripts/download_data.py`
- Test: `tests/test_download_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_download_data.py`:

```python
class TestShouldDownload:
    def test_missing_local_file(self, tmp_path):
        assert dd._should_download(tmp_path / "missing.csv", 100) is True

    def test_size_mismatch_triggers_download(self, tmp_path):
        f = tmp_path / "2020.csv"
        f.write_bytes(b"a" * 50)
        assert dd._should_download(f, 100) is True

    def test_size_match_skips_download(self, tmp_path):
        f = tmp_path / "2020.csv"
        f.write_bytes(b"a" * 100)
        assert dd._should_download(f, 100) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py::TestShouldDownload -v`
Expected: FAIL with `AttributeError: module 'scripts.download_data' has no attribute '_should_download'`

- [ ] **Step 3: Implement `_should_download`**

Add directly below `_fetch_atp_manifest` in `scripts/download_data.py`:

```python
def _should_download(local_path: Path, remote_size: int) -> bool:
    """True if local_path is missing or its size differs from remote_size."""
    if not local_path.exists():
        return True
    return local_path.stat().st_size != remote_size
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py::TestShouldDownload -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/download_data.py tests/test_download_data.py
git commit -m "feat: add size-based skip-download decision for ATP files"
```

---

### Task 3: Schema validation + atomic download

**Files:**
- Modify: `scripts/download_data.py`
- Test: `tests/test_download_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_download_data.py`:

```python
class TestValidateAtpCsv:
    def test_valid_header_passes(self):
        header = ",".join(dd.ATP_SCHEMA_COLUMNS).encode("utf-8")
        data = header + b"\n2026-001,Test Open,Hard,32,A,O,20260101,1,,,,\n"
        assert dd._validate_atp_csv(data) is True

    def test_html_error_page_fails(self):
        data = b"<html><body>502 Bad Gateway</body></html>"
        assert dd._validate_atp_csv(data) is False

    def test_wrong_columns_fail(self):
        data = b"foo,bar,baz\n1,2,3\n"
        assert dd._validate_atp_csv(data) is False

    def test_empty_bytes_fail(self):
        assert dd._validate_atp_csv(b"") is False


class _FakeResponse:
    def __init__(self, content: bytes = b""):
        self.content = content

    def raise_for_status(self):
        pass


class TestDownloadFile:
    def _valid_body(self) -> bytes:
        header = ",".join(dd.ATP_SCHEMA_COLUMNS)
        return (header + "\n2026-001,Test,Hard,32,A,O,20260101,1\n").encode("utf-8")

    def test_successful_download_writes_dest_with_no_tmp_left_behind(self, tmp_path, monkeypatch):
        dest = tmp_path / "2020.csv"
        body = self._valid_body()
        monkeypatch.setattr(dd.requests, "get", lambda url, timeout=None: _FakeResponse(content=body))

        assert dd._download_file("http://x/2020.csv", dest) is True
        assert dest.read_bytes() == body
        assert not (tmp_path / "2020.csv.tmp").exists()

    def test_invalid_schema_leaves_existing_file_untouched(self, tmp_path, monkeypatch):
        dest = tmp_path / "2020.csv"
        dest.write_bytes(b"good,old,data\n1,2,3\n")
        monkeypatch.setattr(
            dd.requests, "get",
            lambda url, timeout=None: _FakeResponse(content=b"<html>error page</html>"),
        )

        assert dd._download_file("http://x/2020.csv", dest) is False
        assert dest.read_bytes() == b"good,old,data\n1,2,3\n"
        assert not (tmp_path / "2020.csv.tmp").exists()

    def test_network_error_leaves_existing_file_untouched(self, tmp_path, monkeypatch):
        dest = tmp_path / "2020.csv"
        dest.write_bytes(b"good,old,data\n1,2,3\n")

        def boom(url, timeout=None):
            raise ConnectionError("network down")

        monkeypatch.setattr(dd.requests, "get", boom)

        assert dd._download_file("http://x/2020.csv", dest) is False
        assert dest.read_bytes() == b"good,old,data\n1,2,3\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py::TestValidateAtpCsv tests/test_download_data.py::TestDownloadFile -v`
Expected: FAIL with `AttributeError` for `_validate_atp_csv` / `_download_file`.

- [ ] **Step 3: Implement `_validate_atp_csv` and `_download_file`**

Add directly below `_should_download` in `scripts/download_data.py`:

```python
def _validate_atp_csv(data: bytes) -> bool:
    """Cheap check that `data` looks like a real ATP CSV: decodable, with a
    header line matching ATP_SCHEMA_COLUMNS exactly. Catches truncated
    downloads and HTML error pages served with a 200 status."""
    if not data:
        return False
    try:
        header_line = data.split(b"\n", 1)[0].decode("utf-8").strip()
    except UnicodeDecodeError:
        return False
    return header_line.split(",") == ATP_SCHEMA_COLUMNS


def _download_file(url: str, dest: Path) -> bool:
    """Download url and atomically write it to dest. Returns True on success.

    On any failure (network error or schema validation failure), dest is left
    completely untouched — no partial or invalid file is ever written there,
    and no .tmp file is left behind either way.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.content
    except Exception as e:
        print(f"  WARNING: could not download {url}: {e}")
        return False

    if not _validate_atp_csv(data):
        print(f"  WARNING: {dest.name} failed schema validation, skipping.")
        return False

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py -v`
Expected: PASS (all tests so far, 10 total)

- [ ] **Step 5: Commit**

```bash
git add scripts/download_data.py tests/test_download_data.py
git commit -m "feat: add schema-validated atomic download for ATP files"
```

---

### Task 4: Merge ongoing_tourneys.csv into the current year

**Files:**
- Modify: `scripts/download_data.py`
- Test: `tests/test_download_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_download_data.py` (`pandas` is already imported at the top of the file
from Task 1):

```python
def _atp_row(**overrides) -> dict:
    row = {col: "" for col in dd.ATP_SCHEMA_COLUMNS}
    row.update(overrides)
    return row


class TestMergeOngoingIntoYear:
    def test_no_overlap_keeps_all_rows(self):
        year_df = pd.DataFrame([
            _atp_row(tourney_id="2026-580", tourney_name="Wimbledon", round="F",
                     match_num=1, winner_name="A"),
        ])
        ongoing_df = pd.DataFrame([
            _atp_row(tourney_id="2026-540", tourney_name="Wimbledon Q", round="R32",
                     match_num=5, winner_name="B"),
        ])
        merged = dd._merge_ongoing_into_year(year_df, ongoing_df)
        assert len(merged) == 2

    def test_conflicting_row_year_file_wins(self):
        year_df = pd.DataFrame([
            _atp_row(tourney_id="2026-540", tourney_name="Wimbledon", round="F",
                     match_num=1, winner_name="ARCHIVED_WINNER"),
        ])
        ongoing_df = pd.DataFrame([
            _atp_row(tourney_id="2026-540", tourney_name="Wimbledon", round="F",
                     match_num=1, winner_name="LIVE_WINNER"),
        ])
        merged = dd._merge_ongoing_into_year(year_df, ongoing_df)
        assert len(merged) == 1
        assert merged.iloc[0]["winner_name"] == "ARCHIVED_WINNER"

    def test_tourney_id_collision_across_different_tournaments_not_collapsed(self):
        # Real bug seen live in the API's 2026.csv: tourney_id "2026-416" is
        # reused by both Munich and Rome Masters, both with match_num=1.
        # (tourney_id, match_num) alone would wrongly collapse these into one
        # row; the composite key must not.
        year_df = pd.DataFrame([
            _atp_row(tourney_id="2026-416", tourney_name="Munich", round="R32",
                     match_num=1, winner_name="MUNICH_WINNER"),
        ])
        ongoing_df = pd.DataFrame([
            _atp_row(tourney_id="2026-416", tourney_name="Rome Masters", round="R128",
                     match_num=1, winner_name="ROME_WINNER"),
        ])
        merged = dd._merge_ongoing_into_year(year_df, ongoing_df)
        assert len(merged) == 2

    def test_result_column_order_matches_schema(self):
        year_df = pd.DataFrame([_atp_row(tourney_id="A", tourney_name="X", round="F", match_num=1)])
        ongoing_df = pd.DataFrame([_atp_row(tourney_id="B", tourney_name="Y", round="F", match_num=1)])
        merged = dd._merge_ongoing_into_year(year_df, ongoing_df)
        assert list(merged.columns) == dd.ATP_SCHEMA_COLUMNS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py::TestMergeOngoingIntoYear -v`
Expected: FAIL with `AttributeError: module 'scripts.download_data' has no attribute '_merge_ongoing_into_year'`

- [ ] **Step 3: Implement `_merge_ongoing_into_year`**

Add directly below `_download_file` in `scripts/download_data.py`:

```python
def _merge_ongoing_into_year(year_df: pd.DataFrame, ongoing_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the archived year file with the live ongoing-tourneys feed,
    deduplicating on a composite key that's robust to tourney_id being reused
    across unrelated tournaments (observed live: "2026-416" used by both
    Munich and Rome Masters). On a genuine duplicate match, the archived
    year_df row wins over the live ongoing_df row.
    """
    year_df = year_df.copy()
    ongoing_df = ongoing_df.copy()
    year_df["_source_priority"] = 0
    ongoing_df["_source_priority"] = 1

    combined = pd.concat([year_df, ongoing_df], ignore_index=True)
    combined = combined.sort_values("_source_priority", kind="stable")
    combined = combined.drop_duplicates(subset=_ATP_MERGE_KEY, keep="first")
    combined = combined.drop(columns=["_source_priority"])
    return combined[ATP_SCHEMA_COLUMNS].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py -v`
Expected: PASS (all tests so far, 14 total)

- [ ] **Step 5: Commit**

```bash
git add scripts/download_data.py tests/test_download_data.py
git commit -m "feat: merge ongoing_tourneys.csv into the current year with collision-safe dedup"
```

---

### Task 5: One-time legacy git-clone cleanup

**Files:**
- Modify: `scripts/download_data.py`
- Test: `tests/test_download_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_download_data.py`:

```python
class TestCleanupLegacyGitClone:
    def test_removes_git_clone_artifacts_but_keeps_year_files(self, tmp_path):
        dest = tmp_path / "tennis_atp_tml"
        dest.mkdir()
        (dest / ".git").mkdir()
        (dest / ".git" / "HEAD").write_text("ref: refs/heads/master")
        (dest / ".github").mkdir()
        (dest / "README.md").write_text("readme")
        (dest / "logo.jpg").write_bytes(b"\xff\xd8")
        (dest / "ATP_Database.csv").write_text("old,aggregate")
        (dest / "ongoing_tourneys.csv").write_text("stale,ongoing")
        (dest / "2025.csv").write_text("keep,me")

        dd._cleanup_legacy_git_clone(dest)

        assert not (dest / ".git").exists()
        assert not (dest / ".github").exists()
        assert not (dest / "README.md").exists()
        assert not (dest / "logo.jpg").exists()
        assert not (dest / "ATP_Database.csv").exists()
        assert not (dest / "ongoing_tourneys.csv").exists()
        assert (dest / "2025.csv").exists()

    def test_noop_when_not_a_git_clone(self, tmp_path):
        dest = tmp_path / "tennis_atp_tml"
        dest.mkdir()
        (dest / "2025.csv").write_text("keep,me")

        dd._cleanup_legacy_git_clone(dest)

        assert (dest / "2025.csv").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py::TestCleanupLegacyGitClone -v`
Expected: FAIL with `AttributeError: module 'scripts.download_data' has no attribute '_cleanup_legacy_git_clone'`

- [ ] **Step 3: Implement `_cleanup_legacy_git_clone`**

Add directly below `_merge_ongoing_into_year` in `scripts/download_data.py`:

```python
def _cleanup_legacy_git_clone(dest: Path) -> None:
    """Remove artifacts from the old git-clone-based ATP source, if present.

    `.git` is the migration marker: if it's gone, this is a no-op, so it's
    safe to call on every run. Existing {year}.csv files are never touched
    here — the download loop (download_atp) decides whether each gets
    refreshed, based on remote size.
    """
    if not (dest / ".git").exists():
        return
    print("  Migrating away from git clone: removing legacy artifacts...")
    for name in _ATP_LEGACY_PATHS:
        path = dest / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py -v`
Expected: PASS (all tests so far, 16 total)

- [ ] **Step 5: Commit**

```bash
git add scripts/download_data.py tests/test_download_data.py
git commit -m "feat: add one-time cleanup of legacy ATP git-clone artifacts"
```

---

### Task 6: Orchestration — `download_atp()` and wiring into `download()`

**Files:**
- Modify: `scripts/download_data.py`
- Test: `tests/test_download_data.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_download_data.py`:

```python
class TestDownloadAtpOrchestration:
    def test_full_flow_downloads_merges_and_skips_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "ATP_DIR", tmp_path / "tennis_atp_tml")
        current_year = datetime.date.today().year
        current_year_name = f"{current_year}.csv"

        header = ",".join(dd.ATP_SCHEMA_COLUMNS)

        def make_row(**overrides):
            vals = {col: "" for col in dd.ATP_SCHEMA_COLUMNS}
            vals.update(overrides)
            return ",".join(str(vals[c]) for c in dd.ATP_SCHEMA_COLUMNS)

        year_csv = (header + "\n" + make_row(
            tourney_id="Y-1", tourney_name="Old Open", round="F",
            match_num=1, winner_name="OLD_WINNER") + "\n").encode("utf-8")
        ongoing_csv = (header + "\n" + make_row(
            tourney_id="Y-2", tourney_name="Live Open", round="R16",
            match_num=1, winner_name="LIVE_WINNER") + "\n").encode("utf-8")

        manifest_json = {
            "count": 3,
            "files": [
                {"name": "1990.csv", "url": "http://x/1990.csv", "size": 999999, "mtime": "t"},
                {"name": current_year_name, "url": "http://x/year.csv", "size": len(year_csv), "mtime": "t"},
                {"name": "ongoing_tourneys.csv", "url": "http://x/ongoing.csv", "size": len(ongoing_csv), "mtime": "t"},
            ],
        }

        class FakeManifestResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return manifest_json

        class FakeFileResponse:
            def __init__(self, content):
                self.content = content

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=None):
            if url == dd.ATP_MANIFEST_URL:
                return FakeManifestResponse()
            if url == "http://x/year.csv":
                return FakeFileResponse(year_csv)
            if url == "http://x/ongoing.csv":
                return FakeFileResponse(ongoing_csv)
            if url == "http://x/1990.csv":
                raise AssertionError("1990.csv should have been skipped: size already matches")
            raise AssertionError(f"unexpected url requested: {url}")

        monkeypatch.setattr(dd.requests, "get", fake_get)

        dd.ATP_DIR.mkdir(parents=True)
        (dd.ATP_DIR / "1990.csv").write_bytes(b"x" * 999999)

        dd.download_atp()

        result = pd.read_csv(dd.ATP_DIR / current_year_name)
        assert set(result["winner_name"]) == {"OLD_WINNER", "LIVE_WINNER"}
        assert (dd.ATP_DIR / "ongoing_tourneys.csv").exists()
        assert (dd.ATP_DIR / "1990.csv").stat().st_size == 999999

    def test_removes_legacy_git_clone_before_downloading(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "ATP_DIR", tmp_path / "tennis_atp_tml")
        dd.ATP_DIR.mkdir(parents=True)
        (dd.ATP_DIR / ".git").mkdir()
        (dd.ATP_DIR / "README.md").write_text("readme")

        def fake_get(url, timeout=None):
            class Empty:
                def raise_for_status(self_inner):
                    pass

                def json(self_inner):
                    return {"count": 0, "files": []}

            return Empty()

        monkeypatch.setattr(dd.requests, "get", fake_get)

        dd.download_atp()

        assert not (dd.ATP_DIR / ".git").exists()
        assert not (dd.ATP_DIR / "README.md").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py::TestDownloadAtpOrchestration -v`
Expected: FAIL with `AttributeError: module 'scripts.download_data' has no attribute 'download_atp'`

- [ ] **Step 3: Implement `download_atp()` and wire it into `download()`**

Add directly below `_cleanup_legacy_git_clone` in `scripts/download_data.py`:

```python
def download_atp() -> None:
    """Download ATP main-tour year files + ongoing_tourneys.csv from the
    stats.tennismylife.org API, then merge the live ongoing feed into the
    current year's file so load_atp_matches() sees it with zero changes."""
    ATP_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_git_clone(ATP_DIR)

    try:
        manifest = _fetch_atp_manifest()
    except Exception as e:
        print(f"  ERROR: could not fetch ATP manifest: {e}")
        return

    current_year_name = f"{datetime.date.today().year}.csv"
    downloaded, skipped = 0, 0
    have_ongoing = have_current_year = False

    for entry in manifest:
        dest = ATP_DIR / entry["name"]
        if entry["name"] == "ongoing_tourneys.csv":
            have_ongoing = True
        elif entry["name"] == current_year_name:
            have_current_year = True

        if not _should_download(dest, entry["size"]):
            skipped += 1
            continue
        if _download_file(entry["url"], dest):
            downloaded += 1
        elif dest.exists():
            skipped += 1

    print(f"  ATP: {downloaded} downloaded, {skipped} already current or unchanged.")

    year_path = ATP_DIR / current_year_name
    ongoing_path = ATP_DIR / "ongoing_tourneys.csv"
    if have_ongoing and have_current_year and year_path.exists() and ongoing_path.exists():
        year_df = pd.read_csv(year_path, low_memory=False)
        ongoing_df = pd.read_csv(ongoing_path, low_memory=False)
        merged = _merge_ongoing_into_year(year_df, ongoing_df)
        tmp = year_path.with_suffix(year_path.suffix + ".tmp")
        merged.to_csv(tmp, index=False)
        tmp.replace(year_path)
        print(f"  Merged ongoing_tourneys.csv into {current_year_name} ({len(merged)} rows).")
```

Now find the `download()` function near the bottom of the file and replace its ATP section. It currently looks like:

```python
def download():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # ── ATP ──────────────────────────────────────────────────────────────────
    dest = DATA_RAW / ATP_REPO["dest"]
    url = ATP_REPO["url"]
    if dest.exists():
        print("tennis_atp_tml: already exists, pulling latest...")
        try:
            subprocess.run(["git", "-C", str(dest), "pull"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: pull failed — {e}")
    else:
        print("Cloning Tennismylife/TML-Database (ATP)...")
        try:
            subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)
            print(f"  Done: {dest}")
        except subprocess.CalledProcessError:
            print(f"\n  ERROR: could not clone {url}\n")

    # ── WTA ──────────────────────────────────────────────────────────────────
    import datetime
    current_year = datetime.date.today().year
```

Replace it with:

```python
def download():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # ── ATP ──────────────────────────────────────────────────────────────────
    print("Downloading ATP data from stats.tennismylife.org...")
    download_atp()

    # ── WTA ──────────────────────────────────────────────────────────────────
    current_year = datetime.date.today().year
```

Note `import datetime` is dropped from inside `download()` since `datetime` is now imported at module level (Task 1). Confirm no other local `import datetime` remains in the file with `grep -n "import datetime" scripts/download_data.py` — there should be exactly one, at the top.

Also confirm `subprocess` is no longer referenced anywhere in the file (`grep -n subprocess scripts/download_data.py` should return nothing) and remove the `import subprocess` line from the top of the file if it's still there from before Task 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_download_data.py -v`
Expected: PASS (all tests, 18 total)

- [ ] **Step 5: Run the full existing suite to confirm nothing else broke**

Run: `./tenis-env/Scripts/python.exe -m pytest -v`
Expected: PASS, same pass count as before this plan started plus the new `test_download_data.py` tests. `tests/test_loader.py` in particular must be unaffected.

- [ ] **Step 6: Commit**

```bash
git add scripts/download_data.py tests/test_download_data.py
git commit -m "feat: wire ATP HTTP download orchestration into download(), remove git-clone path"
```

---

### Task 7: Real-network validation run (manual, not a unit test)

This task exercises the real API and the real local `data/raw/tennis_atp_tml/` directory — it's the first time the code from Tasks 1-6 runs against the live network instead of mocks. Do this only after Task 6's commit.

**Files:** none (no code changes) — this task is entirely `Bash`/`PowerShell` commands and manual verification.

- [ ] **Step 1: Snapshot current state for comparison**

```bash
cd D:\ia-tenis
ls data/raw/tennis_atp_tml/ | wc -l
ls data/raw/tennis_atp_tml/.git 2>/dev/null && echo "still a git clone (expected before this run)"
```

- [ ] **Step 2: Run the real download**

```bash
./tenis-env/Scripts/python.exe scripts/download_data.py
```

Expected: prints "Downloading ATP data from stats.tennismylife.org...", then "Migrating away from git clone: removing legacy artifacts...", then a mix of downloaded/skipped counts, then a "Merged ongoing_tourneys.csv into {year}.csv (N rows)" line, then proceeds into the unchanged WTA section. No TLS warnings, no `verify=False` anywhere in the output or code path.

- [ ] **Step 3: Verify the legacy artifacts are gone and year files remain**

```bash
ls data/raw/tennis_atp_tml/.git 2>/dev/null && echo "FAIL: .git still present" || echo "OK: .git removed"
ls data/raw/tennis_atp_tml/README.md 2>/dev/null && echo "FAIL: README.md still present" || echo "OK: README.md removed"
ls data/raw/tennis_atp_tml/*.csv | wc -l
```

Expected: `.git` and `README.md` gone; CSV count should be roughly the same as (or larger than, if the API has years the frozen clone lacked) the original snapshot from Step 1.

- [ ] **Step 4: Confirm loader.py still loads the data with zero code changes**

```bash
./tenis-env/Scripts/python.exe -c "
from src.data.loader import load_atp_matches
df = load_atp_matches(1990, 2026)
print('rows:', len(df))
print('most recent match_date:', df['match_date'].max())
"
```

Expected: succeeds, `most recent match_date` should be close to today (within `STALENESS_WARNING_DAYS = 30`), not `2026-01-27` or earlier. This is the concrete proof the migration achieved its purpose.

- [ ] **Step 5: Confirm the full test suite is still green against the real refreshed data on disk**

```bash
./tenis-env/Scripts/python.exe -m pytest -v
```

Expected: all tests PASS. (The new `test_download_data.py` tests use `tmp_path` and monkeypatched `ATP_DIR`/`requests.get`, so they're unaffected by what's actually on disk in `data/raw/`; this run is confirming the rest of the suite — loader, features, backtest, value_analysis — still works against real, freshly-downloaded data.)

No commit in this task — it's verification only, nothing under version control changed (recall `data/raw/` is `.gitignore`d).

---

### Task 8: Update the runbook and close out

**Files:**
- Modify: `docs/runbooks/data-refresh-and-staleness.md`

- [ ] **Step 1: Replace the "Refreshing ATP" section**

In `docs/runbooks/data-refresh-and-staleness.md`, find section `## 2. Refreshing ATP` (currently describes `git pull` against `Tennismylife/TML-Database` and checking the upstream repo's last commit date). Replace its entire body with:

`````markdown
## 2. Refreshing ATP

```bash
cd D:\ia-tenis
./tenis-env/Scripts/python.exe scripts/download_data.py
```

As of 2026-07-08 this downloads from the `stats.tennismylife.org` API (the
previous source, `git clone` against `Tennismylife/TML-Database`, froze on
2026-01-27 and was migrated away from — see
`docs/superpowers/specs/2026-07-08-tml-api-migration-design.md`). Each file's
local size is compared against the API manifest's reported size; **a file
being skipped ("N already current or unchanged") is the expected, normal
case for historical years, not a failure** — most years' data never changes
between runs. The current year's file and `ongoing_tourneys.csv` (live,
in-progress tournaments) do re-download on essentially every run, since their
sizes change as new matches are played; the script automatically merges
`ongoing_tourneys.csv` into the current year's `{year}.csv` afterward, so
`load_atp_matches()` sees it with no extra steps.

If the manifest fetch itself fails (`ERROR: could not fetch ATP manifest`),
that's a real problem worth investigating — check connectivity to
`https://stats.tennismylife.org/api/data-files` directly:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" https://stats.tennismylife.org/api/data-files
```

A non-200 here means the API itself is down; wait and retry. This is
different from the old failure mode ("upstream repo hasn't published new
matches") — the API is a live service, not a snapshot repo, so a persistent
failure here is either an outage or a genuine bug, not just "no new data yet."

**Windows TLS note:** the ATP download code uses `requests` with `certifi`'s
bundled CA file explicitly, not `urllib`'s OS-trust-store default. This is
deliberate: prior diagnostics on this machine found that tools routing
through Windows' schannel (native `curl.exe`, PowerShell's
`Invoke-WebRequest`, or Python's `ssl` module falling back to the OS cert
store) can intermittently fail revocation checks against otherwise-healthy
HTTPS endpoints. `requests` + `certifi` sidesteps the OS store entirely. If
you ever see `curl.exe` or PowerShell fail TLS against this endpoint while
Python succeeds, that's this known Windows quirk — not a reason to add
`verify=False` anywhere in this codebase.
`````

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/data-refresh-and-staleness.md
git commit -m "docs: update ATP refresh runbook for stats.tennismylife.org API migration"
```

---

## Post-plan note (not a task — context for whoever runs Fase 4/5 of the original request)

This plan covers the code migration only (spec's Phases 0-3). The original request's Fase 4
(retrain ATP model, before/after backtest comparison, Cobolli/Fery/Alcaraz Elo spot-checks,
re-running today's 2 ATP matches through the refreshed model without auto-logging) and Fase 5
(final closeout checklist) are follow-up work to do **after** this plan's 8 tasks are complete
and committed — they are validation/analysis activities, not implementation tasks, and don't
belong in a code-change plan. Pick them up as a separate conversation turn once Task 8 is
committed, using `docs/runbooks/data-refresh-and-staleness.md` sections 4-6 (retraining,
before/after validation, suspicious-edge handling) as the procedural reference.

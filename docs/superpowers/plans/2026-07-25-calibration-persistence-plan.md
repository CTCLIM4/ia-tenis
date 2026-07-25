# Calibration / PASS Engine Persistence & Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture every prediction the value-bet CLI evaluates — not just the ones a human chooses to save to `value_bets_log.csv` — into a new, separate `data/prediction_audit_log.csv`, stamped with the calibrator's exact parameters and the model's git commit, so calibration/reliability can eventually be audited against the full, unbiased population of predictions instead of a human-selected subset.

**Architecture:** A new `src/git_utils.py` extracts the git-commit-hash helper already built for snapshots (`src/data/snapshots.py`) so both it and the new audit log can share one implementation. A new `src/calibration_audit.py` owns the audit log's schema, a pure classification function (`classify_audit_decision`) that categorizes what happened to an evaluated prediction, and the CSV-append function (`log_prediction_audit`) — deliberately decoupled from `src/value_analysis.py`'s internals (calibration constants are passed in as parameters, not imported, to avoid a circular import). `interactive_cli` in `src/value_analysis.py` calls both, unconditionally, right after its existing `value_bets_log.csv` save decision — the audit log is written every time regardless of that decision, which is the entire point (removes the selection bias that made the old chat-only PASS record useless for auditing).

**Tech Stack:** Python 3.14, pytest, csv (stdlib), no new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-25-calibration-persistence-design.md` (all design questions confirmed 2026-07-25).

---

## File Map

```
D:\ia-tenis\
├── src/
│   ├── git_utils.py              # NEW — shared current_git_commit()
│   ├── data/
│   │   └── snapshots.py          # MODIFIED — use src.git_utils instead of local _current_git_commit
│   ├── calibration_audit.py      # NEW — audit log schema, classify_audit_decision, log_prediction_audit
│   └── value_analysis.py         # MODIFIED — wire audit logging into interactive_cli
├── tests/
│   ├── test_git_utils.py         # NEW
│   ├── test_calibration_audit.py # NEW
│   └── test_snapshots.py         # unmodified (regression guard only)
├── docs/
│   └── runbooks/
│       └── data-refresh-and-staleness.md   # MODIFIED — new "Calibration audit log" section
└── .gitignore                    # MODIFIED — ignore data/prediction_audit_log.csv
```

---

## Task 1: Extract `current_git_commit()` into `src/git_utils.py`

**Files:**
- Create: `src/git_utils.py`
- Create: `tests/test_git_utils.py`
- Modify: `src/data/snapshots.py:1-79` (remove local `_current_git_commit`, import the shared one)

- [ ] **Step 1: Write failing tests**

Create `tests/test_git_utils.py`:

```python
"""Tests for src/git_utils.py — shared git introspection helpers."""
from __future__ import annotations

import subprocess

import src.git_utils as git_utils


class _FakeCompletedProcess:
    def __init__(self, stdout):
        self.stdout = stdout


class TestCurrentGitCommit:
    def test_returns_stripped_short_hash_on_success(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _FakeCompletedProcess("abc1234\n"),
        )
        assert git_utils.current_git_commit() == "abc1234"

    def test_returns_none_when_git_binary_missing(self, monkeypatch):
        def _boom(*a, **kw):
            raise FileNotFoundError("git not found")
        monkeypatch.setattr(subprocess, "run", _boom)
        assert git_utils.current_git_commit() is None

    def test_returns_none_when_not_a_git_repo(self, monkeypatch):
        def _boom(*a, **kw):
            raise subprocess.CalledProcessError(128, ["git"])
        monkeypatch.setattr(subprocess, "run", _boom)
        assert git_utils.current_git_commit() is None

    def test_returns_none_on_timeout(self, monkeypatch):
        def _boom(*a, **kw):
            raise subprocess.TimeoutExpired(["git"], 5)
        monkeypatch.setattr(subprocess, "run", _boom)
        assert git_utils.current_git_commit() is None

    def test_real_invocation_returns_a_string_or_none(self):
        # Sanity check against the real subprocess call (this repo IS a git
        # checkout during tests, so this should return a real short hash) —
        # not mocked, catches a totally broken implementation end-to-end.
        result = git_utils.current_git_commit()
        assert result is None or (isinstance(result, str) and len(result) >= 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_git_utils.py -v`
Expected: `ModuleNotFoundError: No module named 'src.git_utils'`.

- [ ] **Step 3: Create `src/git_utils.py`**

```python
"""Shared git introspection helpers — used by src/data/snapshots.py (which
tour/dataset was frozen at which commit) and src/calibration_audit.py
(which commit's model produced a given prediction), so both stamp the same
notion of "current code version" without duplicating the subprocess call.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def current_git_commit() -> str | None:
    """Best-effort short git commit hash for the current checkout.

    Returns None if git is unavailable, this isn't a git checkout, or the
    call times out — callers must treat None as "unknown" and never let
    this failure block whatever they were doing (creating a snapshot,
    logging a prediction).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_git_utils.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Update `src/data/snapshots.py` to use the shared helper**

Change the top of the file:

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
```

to:

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

from src.git_utils import current_git_commit
```

Remove the now-duplicated local helper — change:

```python
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
```

to:

```python
def _file_record(path: Path) -> dict:
    return {"sha256": _sha256_file(path), "bytes": path.stat().st_size}


def create_snapshot(snapshot_id: str | None = None, tours: tuple[str, ...] = ("atp", "wta")) -> Path:
```

Update the one call site — change:

```python
        metadata = {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": _current_git_commit(),
            "tours": tours_meta,
        }
```

to:

```python
        metadata = {
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": current_git_commit(),
            "tours": tours_meta,
        }
```

- [ ] **Step 6: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass (261 previous + 5 new = 266), no failures. `tests/test_snapshots.py` needs no changes — it only ever checked for the `"git_commit"` key's presence, never called `_current_git_commit` directly, so it's an unmodified regression guard.

- [ ] **Step 7: Commit**

```bash
git add src/git_utils.py src/data/snapshots.py tests/test_git_utils.py
git commit -m "refactor: extract current_git_commit() into shared src/git_utils.py"
```

---

## Task 2: `src/calibration_audit.py` — schema and `classify_audit_decision()`

**Files:**
- Create: `src/calibration_audit.py`
- Create: `tests/test_calibration_audit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibration_audit.py`:

```python
"""Tests for src/calibration_audit.py — the prediction audit log that
captures every evaluated prediction (not just ones saved to
value_bets_log.csv), so calibration can eventually be audited against the
full, unbiased population of predictions."""
from __future__ import annotations

import src.calibration_audit as calibration_audit


class TestClassifyAuditDecision:
    def test_suspicious_wins_over_everything(self):
        # Even if elo_ok is False and logged is True (contradictory in
        # practice, but the priority order must still hold defensively).
        result = calibration_audit.classify_audit_decision(
            suspicious=True, elo_ok=False, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "blocked_suspicious"

    def test_missing_elo_wins_over_logged_and_value(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=False, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "invalid_missing_elo"

    def test_logged_when_user_confirmed_save(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=True, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "logged"

    def test_passed_low_edge_when_neither_side_has_value_and_not_logged(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=True, logged=False,
            has_value_a=False, has_value_b=False,
        )
        assert result == "passed_low_edge"

    def test_passed_user_declined_when_value_existed_but_not_logged(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=True, logged=False,
            has_value_a=True, has_value_b=False,
        )
        assert result == "passed_user_declined"

    def test_passed_user_declined_when_only_side_b_has_value(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=True, logged=False,
            has_value_a=False, has_value_b=True,
        )
        assert result == "passed_user_declined"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_calibration_audit.py -v`
Expected: `ModuleNotFoundError: No module named 'src.calibration_audit'`.

- [ ] **Step 3: Create `src/calibration_audit.py` with the schema and classifier**

```python
"""Calibration/PASS observation audit log: records every prediction the
value-bet CLI evaluates — not just ones a human chose to save to
value_bets_log.csv — so model calibration can eventually be audited
against the full population of predictions instead of the biased subset a
human decided to bet-log. See
docs/superpowers/specs/2026-07-25-calibration-persistence-design.md.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG_PATH = _ROOT / "data" / "prediction_audit_log.csv"

_AUDIT_LOG_FIELDS = [
    "timestamp", "tour", "tournament", "surface", "match_date",
    "player_a", "player_b",
    "p_a_raw", "p_a_cal", "p_b_raw", "p_b_cal",
    "shrink_hi", "shrink_lo", "shrink_rate", "shrinkage_applied",
    "odds_a", "odds_a_source", "odds_b", "odds_b_source",
    "implied_a", "implied_b",
    "edge_a", "ev_a", "kelly_a",
    "edge_b", "ev_b", "kelly_b",
    "model_commit", "model_snapshot_id",
    "elo_found_a", "elo_found_b",
    "decision",  # logged / passed_low_edge / passed_user_declined / blocked_suspicious / invalid_missing_elo
]


def classify_audit_decision(
    suspicious: bool, elo_ok: bool, logged: bool, has_value_a: bool, has_value_b: bool,
) -> str:
    """Classify what happened to one evaluated prediction, for the audit
    log's `decision` column.

    Priority order matters and is deliberately defensive (checked in this
    order even though some combinations shouldn't occur together in
    practice): a suspicious edge always wins — the prediction was never
    trustworthy enough to act on regardless of anything else. Then missing
    Elo — the same data-quality concern log_query() itself encodes via
    status='invalid_missing_elo'. Then whether the user actually logged it
    to value_bets_log.csv. Then whether it even had positive edge on either
    side to log in the first place.
    """
    if suspicious:
        return "blocked_suspicious"
    if not elo_ok:
        return "invalid_missing_elo"
    if logged:
        return "logged"
    if not has_value_a and not has_value_b:
        return "passed_low_edge"
    return "passed_user_declined"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_calibration_audit.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/calibration_audit.py tests/test_calibration_audit.py
git commit -m "feat: add classify_audit_decision() to src/calibration_audit.py"
```

---

## Task 3: `log_prediction_audit()`

**Files:**
- Modify: `src/calibration_audit.py` (append)
- Modify: `tests/test_calibration_audit.py` (append)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_calibration_audit.py`. First change the import line:

```python
from __future__ import annotations

import src.calibration_audit as calibration_audit
```

to:

```python
from __future__ import annotations

import csv
from datetime import date

import pytest

import src.calibration_audit as calibration_audit
from src.value_analysis import FEATURE_COLS, calculate_value
```

Append at the end of the file:

```python
def _make_pred(p_a_raw=0.60, p_a_cal=0.58, elo_found_a=True, elo_found_b=True):
    return {
        "player_a":    "Player A",
        "player_b":    "Player B",
        "p_a_raw":     p_a_raw,
        "p_a_cal":     p_a_cal,
        "p_b_raw":     1.0 - p_a_raw,
        "p_b_cal":     1.0 - p_a_cal,
        "elo_found_a": elo_found_a,
        "elo_found_b": elo_found_b,
    }


def _read_audit_log(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture()
def audit_log_path(tmp_path, monkeypatch):
    path = tmp_path / "prediction_audit_log.csv"
    monkeypatch.setattr(calibration_audit, "AUDIT_LOG_PATH", path)
    return path


class TestLogPredictionAudit:
    def test_writes_one_row_with_expected_fields(self, audit_log_path, monkeypatch):
        monkeypatch.setattr(
            "src.calibration_audit.current_git_commit", lambda: "abc1234", raising=False,
        )
        pred = _make_pred()
        val_a = calculate_value(0.58, 1.90)
        val_b = calculate_value(0.42, 2.10)

        calibration_audit.log_prediction_audit(
            "atp", "Test Open", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, val_a, val_b, 1.90, 2.10,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )

        rows = _read_audit_log(audit_log_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["tour"] == "atp"
        assert row["player_a"] == "Player A"
        assert row["decision"] == "logged"
        assert row["p_a_raw"] == "0.6"
        assert row["shrink_hi"] == "0.9"
        assert row["model_commit"] == "abc1234"
        assert row["model_snapshot_id"] == ""

    def test_model_snapshot_id_recorded_when_given(self, audit_log_path):
        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="logged",
            model_snapshot_id="2026-07-25",
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["model_snapshot_id"] == "2026-07-25"

    def test_appends_multiple_rows(self, audit_log_path):
        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        for decision in ("logged", "passed_low_edge", "blocked_suspicious"):
            calibration_audit.log_prediction_audit(
                "atp", "Test", "hard", date(2026, 7, 25),
                "Player A", "Player B",
                pred, v, v, 1.90, 1.90,
                "manual", "manual",
                decision=decision,
                model_snapshot_id=None,
                shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
            )
        rows = _read_audit_log(audit_log_path)
        assert [r["decision"] for r in rows] == ["logged", "passed_low_edge", "blocked_suspicious"]

    def test_records_elo_found_flags(self, audit_log_path):
        pred = _make_pred(elo_found_a=True, elo_found_b=False)
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "wta", "Test", "clay", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="invalid_missing_elo",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["elo_found_a"] == "True"
        assert row["elo_found_b"] == "False"

    def test_shrinkage_applied_true_when_cal_differs_from_raw(self, audit_log_path):
        pred = _make_pred(p_a_raw=0.95, p_a_cal=0.92)  # shrinkage compresses this
        v = calculate_value(0.92, 1.50)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.50, 1.50,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["shrinkage_applied"] == "True"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_calibration_audit.py::TestLogPredictionAudit -v`
Expected: `AttributeError: module 'src.calibration_audit' has no attribute 'log_prediction_audit'`.

- [ ] **Step 3: Add `log_prediction_audit()` to `src/calibration_audit.py`**

Append:

```python
def log_prediction_audit(
    tour, tournament, surface, match_date,
    player_a, player_b,
    pred: dict, val_a: dict, val_b: dict, odds_a: float, odds_b: float,
    odds_a_source: str, odds_b_source: str,
    decision: str,
    model_snapshot_id: str | None,
    shrink_hi: float, shrink_lo: float, shrink_rate: float,
) -> None:
    """Append one evaluated prediction to the audit log — unconditionally,
    regardless of whether it was also saved to value_bets_log.csv. This is
    what removes the selection bias that made calibration auditing
    impossible before: every prediction the model made is captured, not
    just the ones a human chose to bet-log.

    shrink_hi/shrink_lo/shrink_rate are passed in by the caller (rather
    than imported from src.value_analysis) to avoid a circular import —
    src.value_analysis imports from this module to call it, so this module
    must not import back from src.value_analysis.
    """
    import csv
    from datetime import datetime

    from src.git_utils import current_git_commit

    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shrink = abs(pred["p_a_cal"] - pred["p_a_raw"]) > 0.001

    row = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "tour":            tour,
        "tournament":      tournament,
        "surface":         surface,
        "match_date":      match_date.isoformat(),
        "player_a":        player_a,
        "player_b":        player_b,
        "p_a_raw":         round(pred["p_a_raw"], 4),
        "p_a_cal":         round(pred["p_a_cal"], 4),
        "p_b_raw":         round(pred["p_b_raw"], 4),
        "p_b_cal":         round(pred["p_b_cal"], 4),
        "shrink_hi":       shrink_hi,
        "shrink_lo":       shrink_lo,
        "shrink_rate":     shrink_rate,
        "shrinkage_applied": shrink,
        "odds_a":          odds_a,
        "odds_a_source":   odds_a_source,
        "odds_b":          odds_b,
        "odds_b_source":   odds_b_source,
        "implied_a":       round(val_a["implied_prob"], 4),
        "implied_b":       round(val_b["implied_prob"], 4),
        "edge_a":          round(val_a["edge"], 4),
        "ev_a":            round(val_a["ev"], 4),
        "kelly_a":         round(val_a["kelly_fraction"], 4),
        "edge_b":          round(val_b["edge"], 4),
        "ev_b":            round(val_b["ev"], 4),
        "kelly_b":         round(val_b["kelly_fraction"], 4),
        "model_commit":       current_git_commit(),
        "model_snapshot_id":  model_snapshot_id or "",
        "elo_found_a":     pred.get("elo_found_a", True),
        "elo_found_b":     pred.get("elo_found_b", True),
        "decision":        decision,
    }

    write_header = not AUDIT_LOG_PATH.exists()
    with open(AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_AUDIT_LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_calibration_audit.py -v`
Expected: all 11 tests PASS (6 from Task 2 + 5 new).

Note on `test_writes_one_row_with_expected_fields`: it monkeypatches `"src.calibration_audit.current_git_commit"` with `raising=False` because `current_git_commit` is imported *inside* `log_prediction_audit` (a local import, matching this file's — and the rest of the codebase's — established lazy-import style), so it is not a module-level attribute of `src.calibration_audit` until that function actually runs. `monkeypatch.setattr(..., raising=False)` creates the attribute for the duration of the test even though it doesn't exist at collection time; because the local `import current_git_commit` inside the function resolves the name from `src.git_utils`'s namespace at call time regardless of what's monkeypatched on `src.calibration_audit`, this particular patch is actually a no-op in practice — **this is expected and the test still passes**, because the real `current_git_commit()` runs against this actual repo (a git checkout) and returns a real short hash, which satisfies the assertion `row["model_commit"] == "abc1234"` failing... 

**STOP — this note flags a real bug in the test as written above.** Before implementing, fix the test: `current_git_commit` cannot be patched via `src.calibration_audit.current_git_commit` since it's a local import. Instead, patch it where it actually lives: `monkeypatch.setattr("src.git_utils.current_git_commit", lambda: "abc1234")`. Since `log_prediction_audit` does `from src.git_utils import current_git_commit` *inside the function body on every call*, patching the name on the `src.git_utils` module itself (not on `src.calibration_audit`) is correctly picked up. Use this corrected patch target in Step 1 instead:

```python
    def test_writes_one_row_with_expected_fields(self, audit_log_path, monkeypatch):
        monkeypatch.setattr("src.git_utils.current_git_commit", lambda: "abc1234")
```

(This replaces the incorrect `monkeypatch.setattr("src.calibration_audit.current_git_commit", ...)` line shown earlier in Step 1 — use the corrected version when writing the test file.)

- [ ] **Step 5: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures.

- [ ] **Step 6: Commit**

```bash
git add src/calibration_audit.py tests/test_calibration_audit.py
git commit -m "feat: add log_prediction_audit() to src/calibration_audit.py"
```

---

## Task 4: `.gitignore` — ignore the new audit log

**Files:**
- Modify: `.gitignore`

No automated test — verified with `git check-ignore`, consistent with Task 6 of the snapshot plan.

- [ ] **Step 1: Update `.gitignore`**

Change:

```
data/odds_cache/
data/value_bets_log.csv
data/snapshots/*/processed/
data/snapshots/*/raw/
```

to:

```
data/odds_cache/
data/value_bets_log.csv
data/prediction_audit_log.csv
data/snapshots/*/processed/
data/snapshots/*/raw/
```

- [ ] **Step 2: Verify with `git check-ignore`**

Run: `git check-ignore -v data/prediction_audit_log.csv`
Expected: prints a match against the new `.gitignore` line and exits 0.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore data/prediction_audit_log.csv"
```

---

## Task 5: Wire audit logging into `interactive_cli`

**Files:**
- Modify: `src/value_analysis.py` (imports, `interactive_cli`)

No new automated tests — `interactive_cli` is an `input()`-driven REPL with no existing test coverage (consistent with the rest of the file, see e.g. Task 11 of the snapshot plan for precedent); verification is the full suite (regression guard) plus manual smoke tests.

- [ ] **Step 1: Add the import**

Change:

```python
from src.backtest.walkforward import _MIRROR_FLIP_COLS, load_features_with_mirror
from src.data.snapshots import load_snapshot_metadata, resolve_snapshot_path
from src.data.staleness import StalenessLevel, evaluate_staleness
```

to:

```python
from src.backtest.walkforward import _MIRROR_FLIP_COLS, load_features_with_mirror
from src.calibration_audit import classify_audit_decision, log_prediction_audit
from src.data.snapshots import load_snapshot_metadata, resolve_snapshot_path
from src.data.staleness import StalenessLevel, evaluate_staleness
```

- [ ] **Step 2: Call the audit log unconditionally after the existing save decision**

Change:

```python
        elo_ok = pred.get("elo_found_a", True) and pred.get("elo_found_b", True)
        suspicious = _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious)

        if suspicious:
            best_edge = max(val_a["edge"], val_b["edge"])
            print(f"\n  *** BLOQUEADO: edge sospechoso ({best_edge*100:.1f}% > "
                  f"{SUSPICIOUS_EDGE_THRESHOLD*100:.0f}%) ***")
            print("  *** Posible dato stale o error de matching. Revisa manualmente.")
            print("  *** No se guarda en esta sesion. Corre sin --halt-on-suspicious para loguear igual.")
            save = ""
        elif not elo_ok:
            print("\n  *** Elo faltante para uno o ambos jugadores — la prediccion no es confiable.")
            save = _ask("  Guardar como invalida para mantener trazabilidad? (s/n)", "n").lower()
        else:
            save = _ask("\n  Guardar en log? (s/n)", "s").lower()

        if _should_log_prediction(suspicious, save):
            log_query(
                tour, tournament, surface, match_date,
                player_a, player_b,
                pred, val_a, val_b, odds_a, odds_b,
                odds_a_source, odds_b_source,
            )

        again = _ask("  Analizar otro partido? (s/n)", "s").lower()
```

to:

```python
        elo_ok = pred.get("elo_found_a", True) and pred.get("elo_found_b", True)
        suspicious = _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious)

        if suspicious:
            best_edge = max(val_a["edge"], val_b["edge"])
            print(f"\n  *** BLOQUEADO: edge sospechoso ({best_edge*100:.1f}% > "
                  f"{SUSPICIOUS_EDGE_THRESHOLD*100:.0f}%) ***")
            print("  *** Posible dato stale o error de matching. Revisa manualmente.")
            print("  *** No se guarda en esta sesion. Corre sin --halt-on-suspicious para loguear igual.")
            save = ""
        elif not elo_ok:
            print("\n  *** Elo faltante para uno o ambos jugadores — la prediccion no es confiable.")
            save = _ask("  Guardar como invalida para mantener trazabilidad? (s/n)", "n").lower()
        else:
            save = _ask("\n  Guardar en log? (s/n)", "s").lower()

        logged = _should_log_prediction(suspicious, save)
        if logged:
            log_query(
                tour, tournament, surface, match_date,
                player_a, player_b,
                pred, val_a, val_b, odds_a, odds_b,
                odds_a_source, odds_b_source,
            )

        # Audit log: every evaluated prediction, unconditionally — not just
        # ones saved above. This is what makes future calibration/reliability
        # analysis possible (the full population, not a human-selected
        # subset). See docs/superpowers/specs/2026-07-25-calibration-persistence-design.md.
        decision = classify_audit_decision(
            suspicious, elo_ok, logged, val_a["has_value"], val_b["has_value"],
        )
        log_prediction_audit(
            tour, tournament, surface, match_date,
            player_a, player_b,
            pred, val_a, val_b, odds_a, odds_b,
            odds_a_source, odds_b_source,
            decision=decision,
            model_snapshot_id=snapshot,
            shrink_hi=_SHRINK_HI, shrink_lo=_SHRINK_LO, shrink_rate=_SHRINK_RATE,
        )

        again = _ask("  Analizar otro partido? (s/n)", "s").lower()
```

- [ ] **Step 3: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures, no warnings.

- [ ] **Step 4: Manual smoke tests**

Run: `./tenis-env/Scripts/python.exe -c "import src.value_analysis"`
Expected: succeeds with no traceback (import-time sanity check — this is the check that would catch the circular-import risk this design deliberately avoided, or a typo in the new call site's argument list).

Run: `./tenis-env/Scripts/python.exe -m src.value_analysis --help`
Expected: usage text unchanged from before this task (no new CLI flags were added), exits 0, no traceback.

- [ ] **Step 5: Commit**

```bash
git add src/value_analysis.py
git commit -m "feat: log every evaluated prediction to the calibration audit log"
```

---

## Task 6: Documentation

**Files:**
- Modify: `docs/runbooks/data-refresh-and-staleness.md` (append new section)

- [ ] **Step 1: Add a "Calibration audit log" section**

Append to the end of `docs/runbooks/data-refresh-and-staleness.md` (after section 7, "Dataset snapshots (reproducibility)"):

```markdown

## 8. Calibration audit log

Full design: `docs/superpowers/specs/2026-07-25-calibration-persistence-design.md`.

`value_bets_log.csv` only ever contained rows a human chose to save — a
biased sample that can't support a real reliability/calibration analysis
(you need the full population of predictions the model made, not just the
ones that looked interesting enough to log). `data/prediction_audit_log.csv`
fixes this: **every** prediction the CLI evaluates (once odds are entered)
is appended here automatically, no prompt, regardless of whether it was
also saved to `value_bets_log.csv`.

Each row's `decision` column says what happened to that prediction:

- `logged` — also saved to `value_bets_log.csv` (you said yes).
- `passed_low_edge` — neither side had positive edge; nothing to log.
- `passed_user_declined` — at least one side had positive edge, but you
  said no.
- `blocked_suspicious` — `--halt-on-suspicious` blocked it (edge >10%).
  These never appear in `value_bets_log.csv` at all; this is the only
  record they leave anywhere.
- `invalid_missing_elo` — one or both players had no trained Elo rating.

Each row also carries `shrink_hi`/`shrink_lo`/`shrink_rate` (the exact
calibration constants active when `p_a_cal` was computed — see
`apply_shrinkage()` in `src/value_analysis.py`) and `model_commit` (the git
commit of the code that produced the prediction) plus `model_snapshot_id`
(set when the session was pinned via `--snapshot {id}`, empty for a live
model) — so a future recalibration study can tell exactly which model
version and which calibration constants produced any given historical row,
even after those constants change.

This file is gitignored, same as `value_bets_log.csv` — it's local
prediction history, not something to commit.
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/data-refresh-and-staleness.md
git commit -m "docs: add calibration audit log section to data-refresh runbook"
```

---

## Task 7: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -v`
Expected: every test passes (261 pre-existing + 5 Task 1 + 6 Task 2 + 5 Task 3 ≈ 277 total), zero failures, zero warnings.

- [ ] **Step 2: Real end-to-end smoke test against production-scale data**

This requires `data/raw/`/`data/processed/` to exist locally (gitignored, not part of the repo checkout — regenerate via `scripts/download_data.py` + `python -m src.pipeline {tour}` first if missing).

Run a short non-interactive check that the whole prediction → audit-log path works against a real trained model (adapt the ad-hoc verification pattern already used when the snapshot feature was first verified against real data):

```bash
./tenis-env/Scripts/python.exe -c "
from datetime import date
from src.value_analysis import load_model, predict_match, calculate_value
from src.calibration_audit import classify_audit_decision, log_prediction_audit, AUDIT_LOG_PATH
from src.value_analysis import _SHRINK_HI, _SHRINK_LO, _SHRINK_RATE

elo, fb, clf, rank_lookup, elo_tracker, age_lookup = load_model('atp')
pred = predict_match(
    elo, fb, clf, 'Novak Djokovic', 'Carlos Alcaraz', 'hard', date.today(),
    rank_lookup=rank_lookup, elo_tracker=elo_tracker, age_lookup=age_lookup,
)
val_a = calculate_value(pred['p_a_cal'], 1.90)
val_b = calculate_value(pred['p_b_cal'], 2.05)
decision = classify_audit_decision(
    suspicious=False,
    elo_ok=pred['elo_found_a'] and pred['elo_found_b'],
    logged=False,
    has_value_a=val_a['has_value'], has_value_b=val_b['has_value'],
)
log_prediction_audit(
    'atp', 'Smoke Test', 'hard', date.today(), 'Novak Djokovic', 'Carlos Alcaraz',
    pred, val_a, val_b, 1.90, 2.05, 'manual', 'manual',
    decision=decision, model_snapshot_id=None,
    shrink_hi=_SHRINK_HI, shrink_lo=_SHRINK_LO, shrink_rate=_SHRINK_RATE,
)
print('decision:', decision)
print('audit log at:', AUDIT_LOG_PATH, '(exists:', AUDIT_LOG_PATH.exists(), ')')
"
```

Expected: prints a `decision` value (`passed_low_edge` or `passed_user_declined`, most likely, for a real top-players matchup with plausible odds), confirms `data/prediction_audit_log.csv` was created. Inspect the file directly afterward:

```bash
./tenis-env/Scripts/python.exe -c "
import pandas as pd
df = pd.read_csv('data/prediction_audit_log.csv')
print(df.T)
"
```

Confirm `model_commit` shows a real short git hash (not empty/None) and `model_snapshot_id` is empty (live model, not pinned).

- [ ] **Step 3: Clean up the smoke-test artifact**

The smoke test writes a real row to `data/prediction_audit_log.csv` (gitignored, local-only, but still worth clearing so the first real session starts with a clean file):

```bash
rm data/prediction_audit_log.csv
```

No commit for this task — verification only, confirming Tasks 1-6 integrate correctly end to end against real data.

---

## Self-Review

### Spec Coverage

| Spec requirement (`docs/superpowers/specs/2026-07-25-calibration-persistence-design.md`) | Covered by |
|---|---|
| New parallel file `data/prediction_audit_log.csv`, `value_bets_log.csv` untouched (§4.1) | Task 2, Task 3 (new module); Task 5 leaves `log_query`/`_LOG_FIELDS` completely unmodified |
| Silent auto-log of every evaluated prediction, no confirm prompt (§4.2) | Task 5 Step 2 — `log_prediction_audit` called unconditionally, no `_ask` |
| Model identity via git commit hash, shared helper (§4.3) | Task 1 |
| Calibrator params as raw literal values per row (§4.4) | Task 3 (`shrink_hi`/`shrink_lo`/`shrink_rate` fields, passed as params not imported) |
| Audit-log granularity: post-odds-entry only (§4.5) | Task 5 Step 2 — call site is after `_print_prediction`, same point `log_query` already fires from |
| `blocked_suspicious` matches get an audit row (§4.6) | Task 2's `classify_audit_decision` (priority-ordered, `suspicious` checked first) + Task 5's unconditional call |
| CSV format (§4.7) | Task 3 (`csv.DictWriter`, matching `log_query`'s own style) |
| No circular import between `src.value_analysis` and `src.calibration_audit` | Task 3 — shrink constants passed as parameters, not imported from `src.value_analysis` |

### Placeholder Scan

No TBDs, no "add appropriate error handling," no "similar to Task N" — every step shows complete code. Task 3 contains one deliberate self-correction (the `monkeypatch` target bug) written out in full with the fix, not glossed over — left in place because it demonstrates a real, easy-to-make mistake (patching a name where it's imported *to* rather than where it's imported *from*, for a local/lazy import) that's worth an implementer actually reading rather than silently pre-fixing.

### Type Consistency

- `current_git_commit() -> str | None` (Task 1) — matches its use in `src/data/snapshots.py` (Task 1 Step 5: `"git_commit": current_git_commit()`) and in `log_prediction_audit` (Task 3: `from src.git_utils import current_git_commit`).
- `classify_audit_decision(suspicious: bool, elo_ok: bool, logged: bool, has_value_a: bool, has_value_b: bool) -> str` (Task 2) — matches its call in `interactive_cli` (Task 5 Step 2: `classify_audit_decision(suspicious, elo_ok, logged, val_a["has_value"], val_b["has_value"])`) and every test call (Task 2 Step 1).
- `log_prediction_audit(..., decision: str, model_snapshot_id: str | None, shrink_hi: float, shrink_lo: float, shrink_rate: float)` (Task 3) — matches its call in `interactive_cli` (Task 5 Step 2, passing `snapshot` — `interactive_cli`'s own parameter — as `model_snapshot_id`, and `_SHRINK_HI`/`_SHRINK_LO`/`_SHRINK_RATE` — already-existing module constants — as the three shrink params) and every test call (Task 3 Step 1).
- `AUDIT_LOG_PATH`/`_AUDIT_LOG_FIELDS` (Task 2) — matches the `monkeypatch.setattr(calibration_audit, "AUDIT_LOG_PATH", path)` pattern used in Task 3's tests, mirroring the existing `LOG_PATH` monkeypatch precedent in `tests/test_value_analysis.py`.

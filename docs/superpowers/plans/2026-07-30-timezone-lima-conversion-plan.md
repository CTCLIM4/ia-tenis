# Conversión explícita a America/Lima — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convertir explícitamente los `commence_time` UTC de The Odds API a America/Lima antes de derivar `match_date` (daily_scanner) y antes de calcular el `reference_date` por defecto de los chequeos de staleness, corrigiendo el bug de fecha-cruzada-por-UTC-vs-Lima descrito en el spec.

**Architecture:** Módulo nuevo `src/data/timezone_utils.py` con `to_lima()`/`lima_today()`, consumido por `src/daily_scanner.py` (extracción de `match_date`) y `src/data/staleness.py` / `src/value_analysis.py` (default de `reference_date`). Ningún otro archivo cambia.

**Tech Stack:** `zoneinfo` (stdlib, Python 3.9+) + paquete `tzdata` (base de datos IANA, requerida en Windows/imágenes mínimas de Linux).

**Spec:** `docs/superpowers/specs/2026-07-30-timezone-lima-conversion-design.md`

---

### Task 1: `src/data/timezone_utils.py`

**Files:**
- Create: `src/data/timezone_utils.py`
- Test: `tests/test_timezone_utils.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_timezone_utils.py`:

```python
"""Tests for src/data/timezone_utils.py — explicit America/Lima conversion
for timestamps sourced from The Odds API (UTC)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.data.timezone_utils import LIMA_TZ, lima_today, to_lima


class TestToLima:
    def test_converts_aware_utc_datetime_crossing_day_boundary(self):
        dt = datetime(2026, 7, 27, 2, 0, tzinfo=timezone.utc)
        result = to_lima(dt)
        assert result == datetime(2026, 7, 26, 21, 0, tzinfo=LIMA_TZ)

    def test_offset_is_minus_five_in_july(self):
        dt = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        assert to_lima(dt).utcoffset().total_seconds() / 3600 == -5

    def test_offset_is_minus_five_in_january(self):
        # Lima does not observe DST — offset must stay -05:00 year-round.
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert to_lima(dt).utcoffset().total_seconds() / 3600 == -5

    def test_raises_on_naive_datetime(self):
        naive = datetime(2026, 7, 27, 2, 0)
        with pytest.raises(ValueError):
            to_lima(naive)


class TestLimaToday:
    def test_returns_a_date_instance(self):
        assert isinstance(lima_today(), date)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_timezone_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.timezone_utils'`

- [ ] **Step 3: Write the implementation**

Create `src/data/timezone_utils.py`:

```python
"""Explicit America/Lima conversion for timestamps sourced from The Odds
API (UTC).

The Odds API reports commence_time in UTC. Taking .date() directly off
that UTC datetime silently mislabels matches near the UTC day boundary
(e.g. a 02:00Z match is still the previous day in Lima) — to_lima() forces
that conversion to happen explicitly, everywhere match_date is derived.
See docs/superpowers/specs/2026-07-30-timezone-lima-conversion-design.md.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

LIMA_TZ = ZoneInfo("America/Lima")


def to_lima(dt: datetime) -> datetime:
    """Convert an aware datetime to America/Lima.

    Raises ValueError on a naive datetime — its origin timezone is
    ambiguous and must never be guessed (same "never guess" convention as
    src.player_matcher and src.surface_resolver).
    """
    if dt.tzinfo is None:
        raise ValueError("to_lima() requires a timezone-aware datetime")
    return dt.astimezone(LIMA_TZ)


def lima_today() -> date:
    """Today's calendar date in America/Lima — the app's canonical 'today'."""
    return datetime.now(LIMA_TZ).date()
```

- [ ] **Step 4: Add the `tzdata` dependency**

In `requirements.txt`, add a new line (any position is fine; append after `python-dotenv>=1.0.0`):

```
tzdata>=2024.1
```

`zoneinfo` is stdlib, but on Windows and minimal Linux images there is no
system IANA timezone database — `tzdata` provides it. Already installed on
this machine, but must be declared for reproducibility elsewhere.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_timezone_utils.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add src/data/timezone_utils.py tests/test_timezone_utils.py requirements.txt
git commit -m "feat: add America/Lima timezone conversion helper"
```

---

### Task 2: `src/daily_scanner.py` — `match_date` from the correct calendar day

**Files:**
- Modify: `src/daily_scanner.py:53` (import), `src/daily_scanner.py:182` (conversion), `src/daily_scanner.py:112` (comment)
- Test: `tests/test_daily_scanner.py`

- [ ] **Step 1: Write the failing regression test**

In `tests/test_daily_scanner.py`, inside `class TestDiscoverMatches`, insert this new test right after `test_discovers_and_matches_full_pipeline` (which ends at the `assert m.match_date == date(2026, 7, 27)` line) and before `test_skips_event_outside_time_window`:

```python
    def test_match_date_uses_lima_calendar_day_not_utc(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "fetch_sports_index", lambda api_key: [
            {"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon"},
        ])
        monkeypatch.setattr(
            scanner, "fetch_odds_events_by_key",
            lambda sport_key, api_key: [
                _event("Novak Djokovic", "Jannik Sinner", bookmaker_key=DEFAULT_BOOKMAKER,
                       commence_time="2026-07-27T02:00:00Z")
            ],
        )
        now = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(scanner, "_now", lambda: now)

        matches = discover_matches(
            api_key="fake",
            canonical_names_by_tour={"atp": ["Novak Djokovic", "Jannik Sinner"], "wta": []},
            days_ahead=1,
        )
        assert len(matches) == 1
        # 2026-07-27T02:00Z is 2026-07-26 21:00 in America/Lima (UTC-5) —
        # match_date must reflect the Lima calendar day, not the UTC one.
        assert matches[0].match_date == date(2026, 7, 26)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_daily_scanner.py::TestDiscoverMatches::test_match_date_uses_lima_calendar_day_not_utc -v`
Expected: FAIL — `assert date(2026, 7, 27) == date(2026, 7, 26)`

- [ ] **Step 3: Write the implementation**

In `src/daily_scanner.py`, add the import (insert after line 53, `from src.calibration_audit import classify_audit_decision, log_prediction_audit`, before `from src.odds_api import (`):

```python
from src.data.timezone_utils import to_lima
```

Add a short comment above `_is_within_window` (directly above line 112, `def _is_within_window(...)`) explaining why it is *not* changed:

```python
# _now()/_is_within_window compare timezone-aware instants, not calendar
# dates — that comparison is correct regardless of which tz the operands
# are expressed in (Python normalizes internally), so no Lima conversion
# is needed here. Only the calendar-date extraction below needs it.
```

Change line 182 from:

```python
            match_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).date()
```

to:

```python
            match_date = to_lima(datetime.fromisoformat(commence_time.replace("Z", "+00:00"))).date()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_daily_scanner.py -v`
Expected: PASS, all tests in the file green (including the new one and the pre-existing `test_discovers_and_matches_full_pipeline`, whose `commence_time="2026-07-27T12:00:00Z"` stays on `2026-07-27` in Lima too since 12:00Z is 07:00 Lima the same day).

- [ ] **Step 5: Commit**

```bash
git add src/daily_scanner.py tests/test_daily_scanner.py
git commit -m "fix: derive match_date from America/Lima, not raw UTC date"
```

---

### Task 3: `src/data/staleness.py` — Lima-explicit default `reference_date`

**Files:**
- Modify: `src/data/staleness.py:17` (import), `src/data/staleness.py:151` (default)
- Test: `tests/test_staleness.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_staleness.py`, replace the existing `test_defaults_reference_date_to_today` test (lines 175-179) with:

```python
    def test_defaults_reference_date_to_lima_today(self, monkeypatch):
        fixed_today = date(2026, 7, 30)
        monkeypatch.setattr("src.data.staleness.lima_today", lambda: fixed_today)
        df = _df(fixed_today)
        report = check_dataset_staleness(df)
        assert report.days_stale == 0
```

This monkeypatches `lima_today` directly instead of relying on the real
system clock, so the test is deterministic regardless of which machine or
timezone runs the suite (unlike the old version, which compared
`date.today()` against itself and could never actually fail).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_staleness.py::TestCheckDatasetStalenessDataFrameHandling::test_defaults_reference_date_to_lima_today -v`

(If the test class name differs, use `pytest tests/test_staleness.py -k test_defaults_reference_date_to_lima_today -v` instead.)

Expected: FAIL — `AttributeError: <module 'src.data.staleness'> does not have the attribute 'lima_today'`

- [ ] **Step 3: Write the implementation**

In `src/data/staleness.py`, add the import after `import pandas as pd` (line 17):

```python
from src.data.timezone_utils import lima_today
```

Change line 151 from:

```python
        reference_date = date.today()
```

to:

```python
        reference_date = lima_today()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_staleness.py -v`
Expected: PASS, all tests in the file green.

- [ ] **Step 5: Commit**

```bash
git add src/data/staleness.py tests/test_staleness.py
git commit -m "fix: default staleness reference_date to America/Lima, not system today"
```

---

### Task 4: `src/value_analysis.py` — `_check_staleness` Lima-explicit default

**Files:**
- Modify: `src/value_analysis.py:43` (import), `src/value_analysis.py:341` (default)
- Test: `tests/test_value_analysis.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_value_analysis.py`, inside `class TestCheckStaleness`, add this test after `test_live_tournament_mode_warns_sooner_than_regular_week` (the last method in the class):

```python
    def test_defaults_reference_date_to_lima_today(self, monkeypatch, capsys):
        fixed_today = date(2026, 7, 30)
        monkeypatch.setattr("src.value_analysis.lima_today", lambda: fixed_today)
        _check_staleness("atp", fixed_today, reference_date=None)
        assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_value_analysis.py -k test_defaults_reference_date_to_lima_today -v`
Expected: FAIL — `AttributeError: <module 'src.value_analysis'> does not have the attribute 'lima_today'`

- [ ] **Step 3: Write the implementation**

In `src/value_analysis.py`, add the import after line 43 (`from src.data.staleness import StalenessLevel, evaluate_staleness`):

```python
from src.data.timezone_utils import lima_today
```

Change line 341 from:

```python
        reference_date or date.today(),
```

to:

```python
        reference_date or lima_today(),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_value_analysis.py -v`
Expected: PASS, all tests in the file green.

- [ ] **Step 5: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py
git commit -m "fix: default _check_staleness reference_date to America/Lima"
```

---

### Task 5: Full regression verification

**Files:** none (verification only, no commit)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass (426 pre-existing + 5 new in `test_timezone_utils.py` + 1 new in `test_daily_scanner.py` + 1 new in `test_value_analysis.py`, `test_defaults_reference_date_to_today` replaced in-place in `test_staleness.py` so its count is unchanged). No failures, no errors.

- [ ] **Step 2: Confirm no stray changes**

Run: `git status`
Expected: working tree clean (everything already committed in Tasks 1-4).

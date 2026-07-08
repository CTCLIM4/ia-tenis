# Odds API Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-fill decimal odds in `src/value_analysis.py`'s interactive CLI from The Odds API, falling back to today's manual entry whenever the key is missing, the match isn't found, or the call fails.

**Architecture:** A new standalone module `src/odds_api.py` owns HTTP fetch, a 15-minute file cache (`data/odds_cache/{tour}.json`), and name-based match-finding against a fixed bookmaker. `value_analysis.py` calls it through a thin `try_auto_odds()` wrapper that never raises, pre-fills `_ask_odds()`'s prompt default, and records whether each logged price was `auto` or `manual` in `value_bets_log.csv`.

**Tech Stack:** Python 3.14, stdlib `urllib`/`json` (no new dependencies), pytest.

**Spec:** `docs/superpowers/specs/2026-07-07-odds-api-integration-design.md`

---

## File Map

```
D:\ia-tenis\
├── src/
│   ├── odds_api.py              # NEW — fetch, cache, name matching
│   └── value_analysis.py        # MODIFIED — auto-fill wiring, log fields
├── tests/
│   ├── test_odds_api.py         # NEW
│   └── test_value_analysis.py   # MODIFIED — odds_a_source/odds_b_source coverage
└── .gitignore                   # MODIFIED — ignore data/odds_cache/
```

---

## Task 1: Odds API — name normalization and match-finding (TDD)

**Files:**
- Create: `src/odds_api.py`
- Create: `tests/test_odds_api.py`

- [ ] **Step 1: Write failing tests for normalization and matching**

Create `tests/test_odds_api.py`:

```python
"""Tests for src/odds_api.py — name matching, no real HTTP calls."""
from __future__ import annotations

from src.odds_api import MatchOdds, _normalize_name, find_match_odds


# ── _normalize_name ──────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_lowercases(self):
        assert _normalize_name("Djokovic") == "djokovic"

    def test_strips_accents(self):
        assert _normalize_name("Alcaraz Garfia") == _normalize_name("Alcaraz Garfía")

    def test_strips_periods(self):
        assert _normalize_name("Sabalenka A.") == "sabalenka a"

    def test_collapses_whitespace(self):
        assert _normalize_name("Novak   Djokovic") == "novak djokovic"


# ── find_match_odds ───────────────────────────────────────────────────────────

def _event(home, away, bookmaker_key="bet365", prices=None):
    prices = prices or {home: 1.50, away: 2.60}
    return {
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": bookmaker_key,
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": prices[home]},
                            {"name": away, "price": prices[away]},
                        ],
                    }
                ],
            }
        ],
    }


class TestFindMatchOdds:
    def test_direct_order_match(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Novak Djokovic", "Jannik Sinner")
        assert result == MatchOdds(
            odds_a=1.50, odds_b=2.60,
            matched_home="Novak Djokovic", matched_away="Jannik Sinner",
        )

    def test_reversed_order_match(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Jannik Sinner", "Novak Djokovic")
        assert result == MatchOdds(
            odds_a=2.60, odds_b=1.50,
            matched_home="Novak Djokovic", matched_away="Jannik Sinner",
        )

    def test_no_match_returns_none(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Carlos Alcaraz", "Daniil Medvedev")
        assert result is None

    def test_bookmaker_absent_returns_none(self):
        events = [_event("Novak Djokovic", "Jannik Sinner", bookmaker_key="pinnacle")]
        result = find_match_odds(events, "Novak Djokovic", "Jannik Sinner", bookmaker="bet365")
        assert result is None

    def test_matches_regardless_of_periods_and_accents(self):
        events = [_event("Sabalenka A.", "Swiatek I.")]
        result = find_match_odds(events, "Aryna Sabalenka", "Iga Swiatek")
        assert result is not None
        assert result.matched_home == "Sabalenka A."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_odds_api.py -v`
Expected: `ModuleNotFoundError: No module named 'src.odds_api'`

- [ ] **Step 3: Create `src/odds_api.py` with normalization, dataclass, and matching**

```python
"""The Odds API client: fetch, cache, and match tennis odds by player name."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "data" / "odds_cache"

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
DEFAULT_BOOKMAKER = "bet365"
DEFAULT_CACHE_MINUTES = 15


@dataclass
class MatchOdds:
    odds_a: float
    odds_b: float
    matched_home: str
    matched_away: str


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents/periods, collapse whitespace for name matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.replace(".", "")
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


def _surname(name: str) -> str:
    parts = _normalize_name(name).split(" ")
    return parts[-1] if parts else ""


def find_match_odds(
    events: list[dict],
    player_a: str,
    player_b: str,
    bookmaker: str = DEFAULT_BOOKMAKER,
) -> Optional[MatchOdds]:
    """Find the event matching player_a/player_b (either order) and extract
    that bookmaker's h2h prices.

    Returns None when no event's participants match, or when a matching
    event exists but the configured bookmaker didn't quote it — callers
    must not substitute a different bookmaker.
    """
    surname_a = _surname(player_a)
    surname_b = _surname(player_b)

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        surname_home = _surname(home)
        surname_away = _surname(away)

        if surname_home == surname_a and surname_away == surname_b:
            order = (home, away)
        elif surname_home == surname_b and surname_away == surname_a:
            order = (away, home)
        else:
            continue

        for bk in event.get("bookmakers", []):
            if bk.get("key") != bookmaker:
                continue
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if order[0] in prices and order[1] in prices:
                    return MatchOdds(
                        odds_a=prices[order[0]],
                        odds_b=prices[order[1]],
                        matched_home=home,
                        matched_away=away,
                    )
        return None  # event matched but this bookmaker didn't quote it

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_odds_api.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/odds_api.py tests/test_odds_api.py
git commit -m "feat: odds_api name-matching core (no HTTP yet)"
```

---

## Task 2: Odds API — HTTP fetch and TTL cache (TDD)

**Files:**
- Modify: `src/odds_api.py`
- Modify: `tests/test_odds_api.py`

- [ ] **Step 1: Write failing tests for cache behavior**

Append to `tests/test_odds_api.py`:

```python
import json as _json  # only for clarity in fixtures below; json already stdlib
from datetime import datetime, timedelta, timezone


class TestGetEvents:
    def test_fresh_cache_is_used_without_refetch(self, tmp_path, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.setattr(odds_api, "CACHE_DIR", tmp_path)

        cache_file = tmp_path / "atp.json"
        cache_file.write_text(
            json.dumps({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "events": [{"home_team": "A", "away_team": "B", "bookmakers": []}],
            }),
            encoding="utf-8",
        )

        def _boom(tour, api_key):
            raise AssertionError("should not refetch when cache is fresh")

        monkeypatch.setattr(odds_api, "fetch_odds_events", _boom)
        events = odds_api.get_events("atp", api_key="fake", cache_minutes=15)
        assert events == [{"home_team": "A", "away_team": "B", "bookmakers": []}]

    def test_stale_cache_triggers_refetch(self, tmp_path, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.setattr(odds_api, "CACHE_DIR", tmp_path)

        cache_file = tmp_path / "atp.json"
        old_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        cache_file.write_text(
            json.dumps({"fetched_at": old_time.isoformat(), "events": []}),
            encoding="utf-8",
        )

        fresh_events = [{"home_team": "X", "away_team": "Y", "bookmakers": []}]
        monkeypatch.setattr(odds_api, "fetch_odds_events", lambda tour, api_key: fresh_events)

        events = odds_api.get_events("atp", api_key="fake", cache_minutes=15)
        assert events == fresh_events

    def test_missing_cache_triggers_fetch_and_saves(self, tmp_path, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.setattr(odds_api, "CACHE_DIR", tmp_path)

        fresh_events = [{"home_team": "X", "away_team": "Y", "bookmakers": []}]
        monkeypatch.setattr(odds_api, "fetch_odds_events", lambda tour, api_key: fresh_events)

        events = odds_api.get_events("atp", api_key="fake", cache_minutes=15)
        assert events == fresh_events
        assert (tmp_path / "atp.json").exists()
```

Add `import json` to the top of `tests/test_odds_api.py` (needed by the fixtures above):

```python
import json
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_odds_api.py::TestGetEvents -v`
Expected: `AttributeError: module 'src.odds_api' has no attribute 'get_events'` (and no `fetch_odds_events`).

- [ ] **Step 3: Add fetch, cache, and orchestration functions to `src/odds_api.py`**

Append to `src/odds_api.py` (after `find_match_odds`):

```python
def fetch_odds_events(tour: str, api_key: str) -> list[dict]:
    """Fetch raw upcoming h2h odds events for a tour from The Odds API."""
    sport_key = f"tennis_{tour}"
    url = (
        f"{ODDS_API_BASE}/{sport_key}/odds/"
        f"?apiKey={api_key}&regions=eu&markets=h2h&oddsFormat=decimal"
    )
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read())


def _cache_path(tour: str) -> Path:
    return CACHE_DIR / f"{tour}.json"


def _load_cache(tour: str, max_age_minutes: int) -> Optional[list[dict]]:
    path = _cache_path(tour)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    fetched_at = datetime.fromisoformat(payload["fetched_at"])
    if datetime.now(timezone.utc) - fetched_at >= timedelta(minutes=max_age_minutes):
        return None
    return payload["events"]


def _save_cache(tour: str, events: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_at": datetime.now(timezone.utc).isoformat(), "events": events}
    _cache_path(tour).write_text(json.dumps(payload), encoding="utf-8")


def get_events(tour: str, api_key: str, cache_minutes: int = DEFAULT_CACHE_MINUTES) -> list[dict]:
    """Return cached events if fresh, otherwise fetch and cache."""
    cached = _load_cache(tour, cache_minutes)
    if cached is not None:
        return cached
    events = fetch_odds_events(tour, api_key)
    _save_cache(tour, events)
    return events


def get_match_odds(
    tour: str,
    player_a: str,
    player_b: str,
    api_key: str,
    bookmaker: str = DEFAULT_BOOKMAKER,
    cache_minutes: int = DEFAULT_CACHE_MINUTES,
) -> Optional[MatchOdds]:
    """Top-level lookup: cached/fetched events -> matched odds for this pairing."""
    events = get_events(tour, api_key, cache_minutes)
    return find_match_odds(events, player_a, player_b, bookmaker)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_odds_api.py -v`
Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/odds_api.py tests/test_odds_api.py
git commit -m "feat: odds_api HTTP fetch and TTL file cache"
```

---

## Task 3: Track odds source (auto/manual) in the value-bet log (TDD)

**Files:**
- Modify: `src/value_analysis.py:462-477` (`_LOG_FIELDS`)
- Modify: `src/value_analysis.py:480-482` (`log_query` signature)
- Modify: `src/value_analysis.py:496-527` (`log_query` row dict)
- Modify: `tests/test_value_analysis.py`

- [ ] **Step 1: Write failing tests for the new log fields**

Add to `tests/test_value_analysis.py`, inside `TestLogQueryStatus` (or as a new class right after it):

```python
class TestLogQueryOddsSource:
    def test_odds_source_defaults_to_manual(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        row = _read_log(log_path)[0]
        assert row["odds_a_source"] == "manual"
        assert row["odds_b_source"] == "manual"

    def test_odds_source_records_auto_when_provided(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10,
                  odds_a_source="auto", odds_b_source="auto")
        row = _read_log(log_path)[0]
        assert row["odds_a_source"] == "auto"
        assert row["odds_b_source"] == "auto"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestLogQueryOddsSource -v`
Expected: `TypeError: log_query() got an unexpected keyword argument 'odds_a_source'` (second test), and `KeyError: 'odds_a_source'` (first test, since the column doesn't exist yet).

- [ ] **Step 3: Extend `_LOG_FIELDS`**

In `src/value_analysis.py`, change:

```python
    "odds_a", "odds_b",
```

to:

```python
    "odds_a", "odds_a_source", "odds_b", "odds_b_source",
```

- [ ] **Step 4: Extend `log_query` signature and row dict**

Change:

```python
def log_query(tour, tournament, surface, match_date,
              player_a, player_b,
              pred, val_a, val_b, odds_a, odds_b) -> None:
```

to:

```python
def log_query(tour, tournament, surface, match_date,
              player_a, player_b,
              pred, val_a, val_b, odds_a, odds_b,
              odds_a_source: str = "manual", odds_b_source: str = "manual") -> None:
```

Change:

```python
        "odds_a":          odds_a,
        "odds_b":          odds_b,
```

to:

```python
        "odds_a":          odds_a,
        "odds_a_source":   odds_a_source,
        "odds_b":          odds_b,
        "odds_b_source":   odds_b_source,
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py -v`
Expected: all tests PASS (existing log tests unaffected — new params are keyword-only with defaults).

- [ ] **Step 6: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py
git commit -m "feat: record odds source (auto/manual) in value_bets_log.csv"
```

---

## Task 4: Wire auto-fetch into the interactive CLI

**Files:**
- Modify: `src/value_analysis.py` (docstring, imports, module constants, `_ask_float`/`_ask_odds`, `interactive_cli`)

This task touches the interactive CLI loop, which has no existing unit tests (it's a `input()`-driven REPL) — the same pattern as the rest of `interactive_cli`. Verification is a manual smoke test in Step 6, plus the full automated suite in Step 5.

- [ ] **Step 1: Update the module docstring**

Replace the whole docstring at the top of `src/value_analysis.py` (lines 1-21):

```python
"""
Tennis value-bet analysis — manual odds input.

Calcula Edge / EV / Kelly para un partido dado, usando el modelo
LR+features entrenado sobre el histórico completo de ATP o WTA.

Uso:
  python -m src.value_analysis          # ATP (default)
  python -m src.value_analysis --wta    # WTA
  python -m src.value_analysis --retrain # forzar reentrenamiento

El modelo se reconstruye desde cero la primera vez (~30-60 s ATP),
y luego queda cacheado 7 días en data/model_cache/{tour}.pkl.

TODO: enchufar The Odds API (free tier, 500 req/mes) para poblar las
cuotas automáticamente en lugar de pedirlas al usuario:
  Base URL: https://api.the-odds-api.com/v4/sports/tennis/odds/
  Param: apiKey, regions=eu, markets=h2h, oddsFormat=decimal
  Ver: https://the-odds-api.com/liveapi/guides/v4/
  Filtrar por event que coincida con tournament + player names.
"""
```

with:

```python
"""
Tennis value-bet analysis — auto-fetched odds with manual fallback.

Calcula Edge / EV / Kelly para un partido dado, usando el modelo
LR+features entrenado sobre el histórico completo de ATP o WTA.

Uso:
  python -m src.value_analysis          # ATP (default)
  python -m src.value_analysis --wta    # WTA
  python -m src.value_analysis --retrain # forzar reentrenamiento

El modelo se reconstruye desde cero la primera vez (~30-60 s ATP),
y luego queda cacheado 7 días en data/model_cache/{tour}.pkl.

Cuotas: si ODDS_API_KEY esta configurada en el entorno, se intenta
autocompletar la cuota de cada jugador via The Odds API (bookmaker fijo,
default "bet365" — configurable con ODDS_API_BOOKMAKER, cache de eventos
configurable con ODDS_API_CACHE_MINUTES). Si no hay key, no hay match, o
falla la llamada, se pide la cuota a mano igual que antes — el auto-fetch
nunca bloquea el flujo.
"""
```

- [ ] **Step 2: Add imports and module constants**

Change:

```python
import argparse
import csv
import pickle
import sys
```

to:

```python
import argparse
import csv
import os
import pickle
import sys
```

Change:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
```

to:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.odds_api import DEFAULT_BOOKMAKER, DEFAULT_CACHE_MINUTES, MatchOdds, get_match_odds
```

Change:

```python
KELLY_CAP         = 0.05   # max 5% of bankroll (conservative)
CACHE_MAX_AGE_DAYS = 7     # rebuild if cache older than this
```

to:

```python
KELLY_CAP         = 0.05   # max 5% of bankroll (conservative)
CACHE_MAX_AGE_DAYS = 7     # rebuild if cache older than this

_odds_warned = False       # print the auto-fetch failure warning once per session
```

- [ ] **Step 3: Replace `_ask_float` + `_ask_odds` with `try_auto_odds` + the new `_ask_odds`**

Replace:

```python
def _ask_float(prompt: str, default: Optional[float] = None,
               min_val: float = 0.0, max_val: float = 9999.0) -> Optional[float]:
    dflt_str = str(int(default)) if default is not None else ""
    while True:
        raw = _ask(prompt, dflt_str)
        if not raw:
            return None
        if raw.lower() == "q":
            raise KeyboardInterrupt
        try:
            v = float(raw)
            if min_val < v <= max_val:
                return v
            print(f"    Valor fuera de rango ({min_val} < v <= {max_val}). Reintenta.")
        except ValueError:
            print("    Valor invalido. Ingresa un numero.")


def _ask_odds(player: str) -> float:
    while True:
        val = _ask_float(f"Cuota decimal para {player}", min_val=1.0, max_val=100.0)
        if val is not None:
            return val
        print("    Cuota requerida (> 1.0).")
```

with:

```python
def try_auto_odds(tour: str, player_a: str, player_b: str) -> Optional[MatchOdds]:
    """Best-effort odds auto-fill via The Odds API. Never raises, never blocks.

    Returns None silently when ODDS_API_KEY isn't set or no match/bookmaker
    quote was found. Prints a one-time-per-session warning on genuine
    failures (network error, bad key, rate limit) so the user knows why
    it's not filling in — the manual flow always still works.
    """
    global _odds_warned
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        return None
    bookmaker = os.environ.get("ODDS_API_BOOKMAKER", DEFAULT_BOOKMAKER)
    cache_minutes = int(os.environ.get("ODDS_API_CACHE_MINUTES", DEFAULT_CACHE_MINUTES))
    try:
        return get_match_odds(tour, player_a, player_b, api_key, bookmaker, cache_minutes)
    except Exception as e:
        if not _odds_warned:
            print(f"\n  Aviso: no se pudieron obtener cuotas automaticas ({e}). "
                  f"Se pediran las cuotas a mano el resto de la sesion.")
            _odds_warned = True
        return None


def _ask_odds(player: str, default: Optional[float] = None) -> tuple[float, str]:
    """Ask for decimal odds. Returns (value, source); source is 'auto' when the
    caller-supplied default (from the Odds API) was accepted via Enter."""
    dflt_str = f"{default:.2f}" if default is not None else ""
    while True:
        raw = _ask(f"Cuota decimal para {player}", dflt_str)
        if raw.lower() == "q":
            raise KeyboardInterrupt
        if not raw:
            print("    Cuota requerida (> 1.0).")
            continue
        source = "auto" if (default is not None and raw == dflt_str) else "manual"
        try:
            v = float(raw)
            if 1.0 < v <= 100.0:
                return v, source
            print("    Cuota fuera de rango (1.0 < v <= 100.0). Reintenta.")
        except ValueError:
            print("    Valor invalido. Ingresa un numero.")
```

- [ ] **Step 4: Fetch odds right after player names are captured**

In `interactive_cli`, find the ambiguity-warning block that ends with:

```python
                    print(f"  Usando {chosen[0][:-1] if chosen[0].endswith('.') else chosen[0]}. "
                          f"Si esto es incorrecto, escribe el nombre exacto del indice "
                          f"(ej. '{example_key}').")

            # Rankings — show auto-found values as defaults
```

and insert a new block between them:

```python
                    print(f"  Usando {chosen[0][:-1] if chosen[0].endswith('.') else chosen[0]}. "
                          f"Si esto es incorrecto, escribe el nombre exacto del indice "
                          f"(ej. '{example_key}').")

            # Cuotas automaticas (best-effort, nunca bloquea el flujo)
            auto_odds = try_auto_odds(tour, player_a, player_b)
            if auto_odds is not None:
                bookmaker_label = os.environ.get("ODDS_API_BOOKMAKER", DEFAULT_BOOKMAKER)
                print(f"\n  Cuotas encontradas ({bookmaker_label}): "
                      f"{auto_odds.matched_home} vs {auto_odds.matched_away}")

            # Rankings — show auto-found values as defaults
```

- [ ] **Step 5: Use the auto-fetched odds as defaults, and thread the source into the log call**

Replace:

```python
        # Odds input
        try:
            print(f"\n  Ingresa las cuotas decimales:")
            odds_a = _ask_odds(player_a)
            odds_b = _ask_odds(player_b)
        except KeyboardInterrupt:
            break
```

with:

```python
        # Odds input
        try:
            print(f"\n  Ingresa las cuotas decimales:")
            odds_a, odds_a_source = _ask_odds(
                player_a, default=auto_odds.odds_a if auto_odds else None
            )
            odds_b, odds_b_source = _ask_odds(
                player_b, default=auto_odds.odds_b if auto_odds else None
            )
        except KeyboardInterrupt:
            break
```

Replace:

```python
            save = _ask("\n  Guardar en log? (s/n)", "s").lower()
            if save in ("s", "si", "y", "yes", ""):
                log_query(
                    tour, tournament, surface, match_date,
                    player_a, player_b,
                    pred, val_a, val_b, odds_a, odds_b,
                )
```

with:

```python
            save = _ask("\n  Guardar en log? (s/n)", "s").lower()
            if save in ("s", "si", "y", "yes", ""):
                log_query(
                    tour, tournament, surface, match_date,
                    player_a, player_b,
                    pred, val_a, val_b, odds_a, odds_b,
                    odds_a_source, odds_b_source,
                )
```

- [ ] **Step 6: Run the full automated test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: `89 passed` (pre-existing 75 + 9 from Task 1 + 3 from Task 2 + 2 from Task 3), no failures or collection errors.

- [ ] **Step 7: Manual smoke test without an API key (fallback path)**

Run: `./tenis-env/Scripts/python.exe -m src.value_analysis`
Expected: since `ODDS_API_KEY` is unset in the shell, the flow behaves exactly as before — no "Cuotas encontradas" line appears, `_ask_odds` prompts with no default. Enter a torneo/superficie/fecha/jugadores/cuotas manually and confirm the prediction and log-save steps still work. Answer `q` to exit.

- [ ] **Step 8: Manual smoke test with an API key (auto-fill path)**

Requires a real key from https://the-odds-api.com (free tier signup).

```bash
export ODDS_API_KEY="<your-key>"
./tenis-env/Scripts/python.exe -m src.value_analysis
```

Expected: for a real upcoming ATP match, after typing both player names a `Cuotas encontradas (bet365): <home> vs <away>` line appears, and the subsequent odds prompts show that price as `[default]` — confirm pressing Enter accepts it and the value table computes correctly. Try a player/tournament combination unlikely to be listed and confirm it falls back to blank prompts without any error.

- [ ] **Step 9: Commit**

```bash
git add src/value_analysis.py
git commit -m "feat: auto-fill odds from The Odds API in the interactive CLI"
```

---

## Task 5: Ignore the odds cache directory

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add the cache directory to `.gitignore`**

Change:

```
data/raw/
data/processed/
data/model_cache/
data/value_bets_log.csv
```

to:

```
data/raw/
data/processed/
data/model_cache/
data/odds_cache/
data/value_bets_log.csv
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore odds API cache directory"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Covered by |
|---|---|
| `src/odds_api.py`: `fetch_odds_events`, `_normalize_name`, `find_match_odds`, `MatchOdds` | Task 1, Task 2 |
| Event cache at `data/odds_cache/{tour}.json`, TTL via `ODDS_API_CACHE_MINUTES` | Task 2, Task 5 |
| `ODDS_API_KEY` / `ODDS_API_BOOKMAKER` env config | Task 4 Step 3 (`try_auto_odds`) |
| Auto-fetch inside existing 1-by-1 CLI flow (not a scan mode) | Task 4 Step 4-5 |
| Fixed bookmaker only, no best-of/average | Task 1 (`find_match_odds` bookmaker param, no fallback to other books) |
| Fallback to manual entry on any failure, never blocks | Task 4 Step 3 (`try_auto_odds` catches all exceptions) |
| One warning per session, not per match | Task 4 Step 3 (`_odds_warned` flag) |
| No new fuzzy-matching dependency | Task 1 (`_normalize_name`/`_surname`, stdlib only) |
| CSV log records auto vs manual odds source | Task 3 |

### Placeholder Scan

No TBDs or "implement later" — all steps show complete code.

### Type Consistency

- `MatchOdds` fields (`odds_a`, `odds_b`, `matched_home`, `matched_away`) defined in Task 1, used unchanged in Task 4's `interactive_cli` wiring and in `try_auto_odds`'s return type.
- `find_match_odds(events, player_a, player_b, bookmaker=DEFAULT_BOOKMAKER)` signature (Task 1) matches its call from `get_match_odds` (Task 2).
- `get_match_odds(tour, player_a, player_b, api_key, bookmaker, cache_minutes)` (Task 2) matches its call in `try_auto_odds` (Task 4).
- `_ask_odds(player, default=None) -> tuple[float, str]` (Task 4) matches both call sites (`odds_a, odds_a_source = _ask_odds(...)`).
- `log_query(..., odds_a_source="manual", odds_b_source="manual")` (Task 3) matches its call in `interactive_cli` (Task 4), which always passes both positionally after `odds_b`.

# Staleness Check + --halt-on-suspicious + Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an always-on dataset-staleness warning and an opt-in `--halt-on-suspicious` CLI flag to `src/value_analysis.py`, plus a runbook documenting the refresh/retrain/validate procedure — codifying what was done by hand to catch a real stale-WTA-data bug this session.

**Architecture:** `last_match_date` (the newest match date seen in the raw data) is threaded through the existing cache payload (`_build_elo_fb` → `_save_cache`/`_load_cache` → `load_model`) so a warning can fire on every `load_model()` call, whether served from cache or freshly built. A second, independent pure function decides whether an edge is "suspicious" given a threshold and a flag; `interactive_cli` calls it right before its existing log-save prompt and hard-blocks saving when it returns `True`.

**Tech Stack:** Python 3.14, pytest, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-08-staleness-halt-runbook-design.md`

---

## File Map

```
D:\ia-tenis\
├── src/
│   └── value_analysis.py        # MODIFIED — staleness check, cache payload, halt-on-suspicious
├── tests/
│   └── test_value_analysis.py   # MODIFIED — staleness, cache round-trip, halt-decision tests
└── docs/
    └── runbooks/
        └── data-refresh-and-staleness.md   # NEW
```

---

## Task 1: Staleness warning (TDD)

**Files:**
- Modify: `src/value_analysis.py:50-53` (module constants)
- Modify: `src/value_analysis.py:246` (insert function after `_cache_path`)
- Modify: `tests/test_value_analysis.py` (imports + new test class)

- [ ] **Step 1: Write failing tests**

In `tests/test_value_analysis.py`, change the import block:

```python
from datetime import date
```

to:

```python
from datetime import date, timedelta
```

Change the `from src.value_analysis import (...)` block:

```python
from src.value_analysis import (
    FEATURE_COLS,
    KELLY_CAP,
    _LOG_FIELDS,
    _is_elo_known,
    _resolve_player_name,
    apply_shrinkage,
    calculate_value,
    log_query,
)
```

to:

```python
from src.value_analysis import (
    FEATURE_COLS,
    KELLY_CAP,
    STALENESS_WARNING_DAYS,
    _LOG_FIELDS,
    _check_staleness,
    _is_elo_known,
    _resolve_player_name,
    apply_shrinkage,
    calculate_value,
    log_query,
)
```

Add a new test class anywhere after the imports (e.g. right after `TestApplyShrinkage`):

```python
class TestCheckStaleness:
    def test_warns_when_data_older_than_threshold(self, capsys):
        old_date = date.today() - timedelta(days=STALENESS_WARNING_DAYS + 15)
        _check_staleness("atp", old_date)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" in out
        assert "ATP" in out
        assert str(old_date) in out

    def test_no_warning_when_data_recent(self, capsys):
        recent_date = date.today() - timedelta(days=5)
        _check_staleness("wta", recent_date)
        assert capsys.readouterr().out == ""

    def test_no_warning_exactly_at_threshold(self, capsys):
        boundary_date = date.today() - timedelta(days=STALENESS_WARNING_DAYS)
        _check_staleness("atp", boundary_date)
        assert capsys.readouterr().out == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestCheckStaleness -v`
Expected: `ImportError: cannot import name 'STALENESS_WARNING_DAYS'` (or `_check_staleness`).

- [ ] **Step 3: Add the constant and function**

In `src/value_analysis.py`, change:

```python
KELLY_CAP         = 0.05   # max 5% of bankroll (conservative)
CACHE_MAX_AGE_DAYS = 7     # rebuild if cache older than this
```

to:

```python
KELLY_CAP         = 0.05   # max 5% of bankroll (conservative)
CACHE_MAX_AGE_DAYS = 7     # rebuild if cache older than this
STALENESS_WARNING_DAYS = 30  # warn if newest match in the data is older than this
```

In `src/value_analysis.py`, right after `_cache_path` (the function ending `return _CACHE_DIR / f"{tour}.pkl"`), insert:

```python
def _check_staleness(tour: str, last_match_date: date) -> None:
    """Warn when the newest match in the dataset is more than
    STALENESS_WARNING_DAYS old relative to today.

    Runs on every load_model() call regardless of cache hit/miss, so the
    warning reflects true data age (how recent is the underlying match data)
    rather than cache age (how old is the pickle file) — those are different
    things and a week-old cache can still wrap multi-month-old match data.
    """
    days_stale = (date.today() - last_match_date).days
    if days_stale > STALENESS_WARNING_DAYS:
        print(f"\n  *** ADVERTENCIA: dataset {tour.upper()} desactualizado ***")
        print(f"  *** Ultimo partido en los datos: {last_match_date} ({days_stale} dias atras).")
        print("  *** Las predicciones no incorporan resultados posteriores a esa fecha.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestCheckStaleness -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py
git commit -m "feat: add dataset staleness warning (30-day threshold)"
```

---

## Task 2: Thread `last_match_date` through the model cache (TDD)

**Files:**
- Modify: `src/value_analysis.py:287-311` (`_build_elo_fb`)
- Modify: `src/value_analysis.py:248-260` (`_save_cache`)
- Modify: `src/value_analysis.py:263-282` (`_load_cache`)
- Modify: `src/value_analysis.py:339-357` (`load_model`)
- Modify: `tests/test_value_analysis.py`

- [ ] **Step 1: Write failing tests for cache round-trip**

Add to `tests/test_value_analysis.py` (needs `SimpleNamespace`, already imported at top of the file):

```python
class TestCacheRoundTripsLastMatchDate:
    def test_last_match_date_round_trips(self, tmp_path, monkeypatch):
        import src.value_analysis as va
        monkeypatch.setattr(va, "_CACHE_DIR", tmp_path)

        fake_elo = _fake_elo({"A": 1500.0})
        fake_fb = SimpleNamespace()
        fake_clf = SimpleNamespace()
        rank_lookup = {"A": 10}
        last_match_date = date(2025, 12, 22)

        va._save_cache("atp", fake_elo, fake_fb, fake_clf, rank_lookup, last_match_date)
        cached = va._load_cache("atp")

        assert cached is not None
        elo, fb, clf, rank_lkp, cached_date = cached
        assert cached_date == last_match_date
        assert rank_lkp == rank_lookup

    def test_missing_last_match_date_treated_as_cache_miss(self, tmp_path, monkeypatch):
        import pickle
        from datetime import datetime
        import src.value_analysis as va
        monkeypatch.setattr(va, "_CACHE_DIR", tmp_path)

        # Simulate a cache file written before this feature existed: no
        # "last_match_date" key in the payload.
        old_payload = {
            "elo":         _fake_elo({"A": 1500.0}),
            "fb":          SimpleNamespace(),
            "clf":         SimpleNamespace(),
            "rank_lookup": {"A": 10},
            "timestamp":   datetime.now(),
            "tour":        "atp",
        }
        path = tmp_path / "atp.pkl"
        with open(path, "wb") as f:
            pickle.dump(old_payload, f)

        assert va._load_cache("atp") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestCacheRoundTripsLastMatchDate -v`
Expected: `TypeError: _save_cache() missing 1 required positional argument: 'last_match_date'`.

- [ ] **Step 3: Thread `last_match_date` through `_build_elo_fb`**

Change:

```python
def _build_elo_fb(tour: str):
    """Rebuild EloSystem + FeatureBuilder from raw matches.

    Returns (elo, fb, rank_lookup) where rank_lookup maps player names to
    their most recently observed ATP/WTA ranking.
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
    # Sort by date so rank lookup iteration is chronological
    df_raw   = df_raw.sort_values("match_date").reset_index(drop=True)
    rank_lkp = _build_rank_lookup(df_raw)
    elo      = EloSystem()
    fb       = FeatureBuilder()
    build_match_features(df_raw, elo, fb)   # mutates elo and fb in-place
    print(f"  Listo: {len(elo.general_ratings):,} jugadores, "
          f"{len(rank_lkp):,} con ranking conocido.")
    return elo, fb, rank_lkp
```

to:

```python
def _build_elo_fb(tour: str):
    """Rebuild EloSystem + FeatureBuilder from raw matches.

    Returns (elo, fb, rank_lookup, last_match_date) where rank_lookup maps
    player names to their most recently observed ATP/WTA ranking, and
    last_match_date is the most recent match_date seen in the raw data.
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
    # Sort by date so rank lookup iteration is chronological
    df_raw   = df_raw.sort_values("match_date").reset_index(drop=True)
    rank_lkp = _build_rank_lookup(df_raw)
    last_match_date = df_raw["match_date"].max().date()
    elo      = EloSystem()
    fb       = FeatureBuilder()
    build_match_features(df_raw, elo, fb)   # mutates elo and fb in-place
    print(f"  Listo: {len(elo.general_ratings):,} jugadores, "
          f"{len(rank_lkp):,} con ranking conocido.")
    return elo, fb, rank_lkp, last_match_date
```

- [ ] **Step 4: Thread `last_match_date` through `_save_cache`**

Change:

```python
def _save_cache(tour: str, elo, fb, clf, rank_lookup: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "elo":          elo,
        "fb":           fb,
        "clf":          clf,
        "rank_lookup":  rank_lookup,
        "timestamp":    datetime.now(),
        "tour":         tour,
    }
    with open(_cache_path(tour), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache guardado en {_cache_path(tour)}")
```

to:

```python
def _save_cache(tour: str, elo, fb, clf, rank_lookup: dict, last_match_date: date) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "elo":              elo,
        "fb":               fb,
        "clf":              clf,
        "rank_lookup":      rank_lookup,
        "timestamp":        datetime.now(),
        "tour":             tour,
        "last_match_date":  last_match_date,
    }
    with open(_cache_path(tour), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache guardado en {_cache_path(tour)}")
```

- [ ] **Step 5: Thread `last_match_date` through `_load_cache`**

Change:

```python
def _load_cache(tour: str):
    """Return (elo, fb, clf, rank_lookup) from cache, or None if stale/missing."""
    path = _cache_path(tour)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f"  Advertencia: no se pudo leer cache ({e}). Reentrenando...")
        return None

    age = datetime.now() - payload["timestamp"]
    if age >= timedelta(days=CACHE_MAX_AGE_DAYS):
        print(f"  Cache expirado ({age.days}d {age.seconds//3600}h). Reentrenando...")
        return None

    age_str = (f"{age.days}d " if age.days else "") + f"{age.seconds//3600}h {(age.seconds%3600)//60}m"
    print(f"  Cache cargado ({age_str} de antiguedad — maximo {CACHE_MAX_AGE_DAYS}d).")
    return payload["elo"], payload["fb"], payload["clf"], payload["rank_lookup"]
```

to:

```python
def _load_cache(tour: str):
    """Return (elo, fb, clf, rank_lookup, last_match_date) from cache, or None if stale/missing."""
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
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestCacheRoundTripsLastMatchDate -v`
Expected: both tests PASS.

- [ ] **Step 7: Wire `_check_staleness` into `load_model`**

`load_model`'s public return signature stays a 4-tuple `(elo, fb, clf, rank_lookup)` — unchanged for every existing caller (`interactive_cli` and any ad-hoc script). `last_match_date` is consumed internally only to run the staleness check.

Change:

```python
def load_model(tour: str = "atp", retrain: bool = False):
    """Load or rebuild model, with transparent cache management.

    Returns:
        elo          – EloSystem with full historical state
        fb           – FeatureBuilder with full historical state
        clf          – LogisticRegression trained on all available data
        rank_lookup  – dict {player_name: most_recent_rank}
    """
    if not retrain:
        cached = _load_cache(tour)
        if cached is not None:
            return cached   # (elo, fb, clf, rank_lookup)

    print(f"Construyendo modelo desde cero ({tour.upper()}) — primera vez ~30-60 s...")
    elo, fb, rank_lkp = _build_elo_fb(tour)
    clf               = _train_lr(tour)
    _save_cache(tour, elo, fb, clf, rank_lkp)
    return elo, fb, clf, rank_lkp
```

to:

```python
def load_model(tour: str = "atp", retrain: bool = False):
    """Load or rebuild model, with transparent cache management.

    Returns:
        elo          – EloSystem with full historical state
        fb           – FeatureBuilder with full historical state
        clf          – LogisticRegression trained on all available data
        rank_lookup  – dict {player_name: most_recent_rank}
    """
    if not retrain:
        cached = _load_cache(tour)
        if cached is not None:
            elo, fb, clf, rank_lookup, last_match_date = cached
            _check_staleness(tour, last_match_date)
            return elo, fb, clf, rank_lookup

    print(f"Construyendo modelo desde cero ({tour.upper()}) — primera vez ~30-60 s...")
    elo, fb, rank_lkp, last_match_date = _build_elo_fb(tour)
    clf               = _train_lr(tour)
    _save_cache(tour, elo, fb, clf, rank_lkp, last_match_date)
    _check_staleness(tour, last_match_date)
    return elo, fb, clf, rank_lkp
```

- [ ] **Step 8: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures or collection errors.

- [ ] **Step 9: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py
git commit -m "feat: thread last_match_date through model cache for staleness checks"
```

---

## Task 3: `--halt-on-suspicious` decision logic (TDD)

**Files:**
- Modify: `src/value_analysis.py:50-53` (module constants)
- Modify: `src/value_analysis.py:110` (insert function after `calculate_value`)
- Modify: `tests/test_value_analysis.py`

- [ ] **Step 1: Write failing tests**

Add to the `from src.value_analysis import (...)` block in `tests/test_value_analysis.py` (from Task 1's edit):

```python
from src.value_analysis import (
    FEATURE_COLS,
    KELLY_CAP,
    STALENESS_WARNING_DAYS,
    SUSPICIOUS_EDGE_THRESHOLD,
    _LOG_FIELDS,
    _check_staleness,
    _is_elo_known,
    _resolve_player_name,
    _should_halt_on_suspicious_edge,
    apply_shrinkage,
    calculate_value,
    log_query,
)
```

Add a new test class:

```python
class TestShouldHaltOnSuspiciousEdge:
    def test_true_when_edge_above_threshold_and_flag_on(self):
        val_a = {"edge": 0.15}
        val_b = {"edge": -0.05}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is True

    def test_false_when_flag_off_even_if_edge_high(self):
        val_a = {"edge": 0.20}
        val_b = {"edge": -0.05}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=False) is False

    def test_false_when_edge_below_threshold(self):
        val_a = {"edge": 0.05}
        val_b = {"edge": 0.02}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is False

    def test_false_exactly_at_threshold(self):
        val_a = {"edge": SUSPICIOUS_EDGE_THRESHOLD}
        val_b = {"edge": 0.0}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is False

    def test_checks_both_sides_takes_max(self):
        val_a = {"edge": -0.5}
        val_b = {"edge": 0.11}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestShouldHaltOnSuspiciousEdge -v`
Expected: `ImportError: cannot import name 'SUSPICIOUS_EDGE_THRESHOLD'` (or `_should_halt_on_suspicious_edge`).

- [ ] **Step 3: Add the constant and function**

Change:

```python
KELLY_CAP         = 0.05   # max 5% of bankroll (conservative)
CACHE_MAX_AGE_DAYS = 7     # rebuild if cache older than this
STALENESS_WARNING_DAYS = 30  # warn if newest match in the data is older than this
```

to:

```python
KELLY_CAP         = 0.05   # max 5% of bankroll (conservative)
CACHE_MAX_AGE_DAYS = 7     # rebuild if cache older than this
STALENESS_WARNING_DAYS = 30  # warn if newest match in the data is older than this
SUSPICIOUS_EDGE_THRESHOLD = 0.10  # hard-block logging above this when --halt-on-suspicious
```

Right after `calculate_value`'s closing `return {...}` block (before the `# ── rank lookup ───` section header), insert:

```python
def _should_halt_on_suspicious_edge(val_a: dict, val_b: dict, halt_on_suspicious: bool) -> bool:
    """True when --halt-on-suspicious is active and either side's edge exceeds
    SUSPICIOUS_EDGE_THRESHOLD — a signal the prediction may be based on stale
    data or a name-matching error rather than genuine market inefficiency."""
    return halt_on_suspicious and max(val_a["edge"], val_b["edge"]) > SUSPICIOUS_EDGE_THRESHOLD
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestShouldHaltOnSuspiciousEdge -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py
git commit -m "feat: add suspicious-edge decision logic for --halt-on-suspicious"
```

---

## Task 4: Wire `--halt-on-suspicious` into the interactive CLI

**Files:**
- Modify: `src/value_analysis.py:752` (`interactive_cli` signature)
- Modify: `src/value_analysis.py:877-888` (log-save block)
- Modify: `src/value_analysis.py:899-909` (`main`)

This task has no new automated tests — `interactive_cli` is an `input()`-driven REPL with no existing test coverage (consistent with the rest of the file); verification is the full suite (regression guard) plus a manual smoke test.

- [ ] **Step 1: Add the parameter to `interactive_cli`**

Change:

```python
def interactive_cli(tour: str = "atp", retrain: bool = False) -> None:
```

to:

```python
def interactive_cli(tour: str = "atp", retrain: bool = False, halt_on_suspicious: bool = False) -> None:
```

- [ ] **Step 2: Insert the hard-block check before the log-save prompt**

Change:

```python
        elo_ok = pred.get("elo_found_a", True) and pred.get("elo_found_b", True)
        if not elo_ok:
            print("\n  Log bloqueado: prediccion invalida (Elo faltante). Corrige el nombre del jugador.")
        else:
            save = _ask("\n  Guardar en log? (s/n)", "s").lower()
            if save in ("s", "si", "y", "yes", ""):
                log_query(
                    tour, tournament, surface, match_date,
                    player_a, player_b,
                    pred, val_a, val_b, odds_a, odds_b,
                    odds_a_source, odds_b_source,
                )
```

to:

```python
        elo_ok = pred.get("elo_found_a", True) and pred.get("elo_found_b", True)
        if _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious):
            best_edge = max(val_a["edge"], val_b["edge"])
            print(f"\n  *** BLOQUEADO: edge sospechoso ({best_edge*100:.1f}% > "
                  f"{SUSPICIOUS_EDGE_THRESHOLD*100:.0f}%) ***")
            print("  *** Posible dato stale o error de matching. Revisa manualmente.")
            print("  *** No se guarda en esta sesion. Corre sin --halt-on-suspicious para loguear igual.")
        elif not elo_ok:
            print("\n  Log bloqueado: prediccion invalida (Elo faltante). Corrige el nombre del jugador.")
        else:
            save = _ask("\n  Guardar en log? (s/n)", "s").lower()
            if save in ("s", "si", "y", "yes", ""):
                log_query(
                    tour, tournament, surface, match_date,
                    player_a, player_b,
                    pred, val_a, val_b, odds_a, odds_b,
                    odds_a_source, odds_b_source,
                )
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
    args = parser.parse_args()
    tour = "wta" if args.wta else "atp"
    interactive_cli(tour, retrain=args.retrain)
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
                             f"{SUSPICIOUS_EDGE_THRESHOLD*100:.0f}% (posible dato stale)")
    args = parser.parse_args()
    tour = "wta" if args.wta else "atp"
    interactive_cli(tour, retrain=args.retrain, halt_on_suspicious=args.halt_on_suspicious)
```

- [ ] **Step 4: Run the full test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no failures or collection errors.

- [ ] **Step 5: Manual smoke test**

Run: `./tenis-env/Scripts/python.exe -m src.value_analysis --help`
Expected: usage text includes `--halt-on-suspicious` with its help string, exits 0, no traceback.

Run: `./tenis-env/Scripts/python.exe -c "import src.value_analysis"`
Expected: succeeds with no traceback (import-time sanity check after the edits).

- [ ] **Step 6: Commit**

```bash
git add src/value_analysis.py
git commit -m "feat: add --halt-on-suspicious flag to hard-block logging large edges"
```

---

## Task 5: Runbook

**Files:**
- Create: `docs/runbooks/data-refresh-and-staleness.md`

- [ ] **Step 1: Create the runbook**

```markdown
# Runbook: Refreshing tennis data, retraining, and handling suspicious edges

Context: on 2026-07-08 the WTA dataset was frozen at 2023 (tennis-data.co.uk
was briefly unreachable, then recovered). This produced two artificially large
(>10%) value-bet edges on real Wimbledon QF matchups that vanished once the
data was refreshed and the model retrained. This runbook is the procedure used
to diagnose and fix that, so it doesn't have to be reconstructed from scratch.

## 1. How staleness shows up

Every `load_model(tour)` call (in `src/value_analysis.py`) now prints a warning
automatically when the newest match in the dataset is more than 30 days old
relative to today:

```
*** ADVERTENCIA: dataset ATP desactualizado ***
*** Ultimo partido en los datos: 2026-01-27 (162 dias atras).
*** Las predicciones no incorporan resultados posteriores a esa fecha.
```

This fires whether the model came from cache or was freshly rebuilt — it's
about the age of the *match data*, not the age of the *cache file* (a separate,
already-existing 7-day cache-expiry concept, `CACHE_MAX_AGE_DAYS`).

If you see this warning before a prediction you're about to trust, refresh
that tour's data first (sections 2-3), then retrain (section 4), then
optionally validate (section 5) before proceeding.

## 2. Refreshing ATP

```bash
cd D:\ia-tenis
./tenis-env/Scripts/python.exe scripts/download_data.py
```

This runs `git pull` against `Tennismylife/TML-Database`. **"Already up to
date" does not necessarily mean the data is current** — it means *our local
clone* matches the *upstream repo*, which may itself not have been updated in
a while. Before assuming a bug, check the upstream repo's own last commit:

```bash
git -C data/raw/tennis_atp_tml log -1 --format="%ci %s"
```

If that commit date is old (e.g. months ago), there is nothing more to refresh
— the public data source simply hasn't published newer matches yet. This is
not fixable from this repo; note it in your analysis and treat predictions for
recent time periods as based on stale ATP data until the upstream source
catches up.

## 3. Refreshing WTA

Same script, different source (tennis-data.co.uk, per-year `.xlsx`/`.xls`
files). If specific recent years fail to download ("no pattern worked"),
check whether the site itself is reachable before assuming the URL patterns
in `scripts/download_data.py` are wrong:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 -A "Mozilla/5.0" http://www.tennis-data.co.uk/
```

A `503` or an SSL handshake failure means **the site is temporarily down**,
not a bug in this repo — the download patterns (`_download_wta_year` in
`scripts/download_data.py`) already try `{year}w/{year}.xlsx`,
`{year}w/{year}w.xlsx`, and several other historical variants. Wait a few
minutes and retry:

```bash
./tenis-env/Scripts/python.exe scripts/download_data.py
```

Once the site responds (root URL returns `200`), re-run the same command —
already-downloaded years are skipped automatically, only missing years are
fetched.

## 4. Retraining

Refreshing the raw files under `data/raw/` does **not** automatically update
the cached model in `data/model_cache/{tour}.pkl` (that cache has its own
7-day expiry, `CACHE_MAX_AGE_DAYS`, unrelated to how current the underlying
match data is). Force a rebuild directly:

```bash
cd D:\ia-tenis
./tenis-env/Scripts/python.exe -c "
from src.value_analysis import load_model
load_model('wta', retrain=True)   # or 'atp'
"
```

This also regenerates the `last_match_date` stored in the cache, so the
staleness warning (section 1) reflects the refreshed data going forward.

## 5. Validating before/after

Before touching `data/processed/{tour}_features.csv`, capture a baseline from
the *current* (pre-refresh) file:

```bash
./tenis-env/Scripts/python.exe -c "
import numpy as np, pandas as pd
from src.backtest.walkforward import walk_forward_backtest

df = pd.read_csv('data/processed/wta_features.csv', parse_dates=['match_date'])
mirror = df.copy()
for col in ('elo_diff', 'rank_diff', 'form_diff', 'surface_form_diff', 'rest_diff'):
    mirror[col] = -mirror[col]
mirror['elo_prob'] = 1.0 - mirror['elo_prob']
mirror['h2h_rate'] = 1.0 - mirror['h2h_rate']
mirror['outcome'] = 0
mirror['is_mirror'] = True
full = pd.concat([df, mirror], ignore_index=True)

results = walk_forward_backtest(full, warmup_years=10)
accs = [m['accuracy'] for m in results.values()]
lls = [m['log_loss'] for m in results.values()]
briers = [m['brier_score'] for m in results.values()]
print(f'Accuracy: {np.mean(accs):.4f}  LogLoss: {np.mean(lls):.4f}  Brier: {np.mean(briers):.4f}')
print(f'Years tested: {len(results)}')
"
```

Then regenerate the features CSV with the refreshed raw data and re-run the
same snippet against the new file:

```bash
./tenis-env/Scripts/python.exe -m src.pipeline wta 2007 2026
```

Compare the two runs' Accuracy/LogLoss/Brier. Expect no material degradation
— if metrics get meaningfully worse after a refresh, investigate the loader
before trusting the new data (e.g. verify a specific player's known result
history against the raw file directly, the way Jasmine Paolini's 2024
Wimbledon run was cross-checked against `data/raw/tennis_wta_tduk/2024w.xlsx`
on 2026-07-08).

## 6. Suspicious edges

`src/value_analysis.py` defines `SUSPICIOUS_EDGE_THRESHOLD = 0.10`. Run with
`--halt-on-suspicious` when you want the CLI to refuse to log any match whose
best-side edge exceeds 10% — useful for routine/unattended sessions where you
won't be scrutinizing every prediction by hand:

```bash
./tenis-env/Scripts/python.exe -m src.value_analysis --halt-on-suspicious
./tenis-env/Scripts/python.exe -m src.value_analysis --wta --halt-on-suspicious
```

When it blocks a match, the log-save prompt is skipped entirely for that
match in that session — there is no way to override it without re-running
without the flag. Omit the flag when you've already manually verified a large
edge is real (e.g. after confirming the data isn't stale) and want to log it
anyway.

A large edge is not proof of a bug or of real market value by itself — it's a
prompt to check: is the data for both players current (section 1)? Does the
player-name match the right person (see the WTA disambiguation logic in
`_resolve_player_name`)? Only log it once you can answer both confidently.
```

- [ ] **Step 2: Commit**

```bash
git add docs/runbooks/data-refresh-and-staleness.md
git commit -m "docs: add data refresh, retrain, and suspicious-edge runbook"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Covered by |
|---|---|
| `STALENESS_WARNING_DAYS = 30` constant | Task 1 |
| `_check_staleness` prints warning, always runs (cache hit or miss) | Task 1, Task 2 Step 7 |
| `last_match_date` stored in cache payload | Task 2 |
| Old-format cache (missing `last_match_date`) treated as cache miss, no crash | Task 2 Steps 1, 5 |
| `SUSPICIOUS_EDGE_THRESHOLD = 0.10` constant, fixed (not env-configurable) | Task 3 |
| `--halt-on-suspicious` flag, hard block (not a flippable default) | Task 4 |
| Testable seam (`_should_halt_on_suspicious_edge`) instead of testing REPL internals | Task 3 |
| Order: suspicious-edge check before missing-Elo check | Task 4 Step 2 (`if ... elif not elo_ok ... else`) |
| Runbook: staleness, ATP refresh + upstream-lag caveat, WTA refresh + 503 troubleshooting, retrain, before/after validation, suspicious-edge flag usage | Task 5 |

### Placeholder Scan

No TBDs or "implement later" — every step shows complete code or complete runbook prose.

### Type Consistency

- `_build_elo_fb` now returns a 4-tuple `(elo, fb, rank_lkp, last_match_date)` (Task 2 Step 3) — matches its sole call site in `load_model` (Task 2 Step 7), which unpacks exactly 4 values.
- `_save_cache(tour, elo, fb, clf, rank_lookup, last_match_date)` (Task 2 Step 4) matches its call in `load_model` (Task 2 Step 7), which passes `last_match_date` as the 6th argument.
- `_load_cache` now returns a 5-tuple or `None` (Task 2 Step 5) — matches its unpacking in both `load_model` (Task 2 Step 7: `elo, fb, clf, rank_lookup, last_match_date = cached`) and the new test (Task 2 Step 1).
- `load_model`'s public return signature is unchanged (4-tuple) — matches every existing caller (`interactive_cli` line 754, unmodified) with no breaking change.
- `_should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious)` (Task 3) signature matches its call in `interactive_cli` (Task 4 Step 2) and every test call (Task 3 Step 1).
- `interactive_cli(tour, retrain, halt_on_suspicious)` (Task 4 Step 1) matches its call in `main()` (Task 4 Step 3).

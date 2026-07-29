# Módulo 3, Sub-proyecto 3b: Fatiga/Carga de Partidos — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar `fatigue_multiplier_diff` a `FEATURE_COLS`, calculada a partir de partidos+sets jugados en ventanas de 7 y 14 días, integrada como multiplicador de `adjusted_elo_surface` (mismo patrón que `age_multiplier`/`rust_factor`).

**Architecture:** Normalizar `sets_played` en la capa de carga de datos (`loader.py`) → trackear historial de carga por jugador en `FeatureBuilder` (paralelo al historial existente, sin tocarlo) → función pura `fatigue_multiplier` en `decay.py` → integrar en `calculate_decay_features` → conectar en los dos puntos de consumo (`walkforward.py` para entrenamiento, `value_analysis.py` para predicción) → validar con backtest.

**Tech Stack:** Python, numpy, pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-07-28-fatigue-workload-design.md` — leer primero para las fórmulas exactas y el razonamiento (por qué RET cuenta, por qué los targets no están calibrados, por qué el tope es 15%).

**Baseline reutilizado de 3a** (mismo estado de `master`, no hace falta re-correr):
```
ATP: Accuracy 0.6688, Log-Loss 0.6024
WTA: Accuracy 0.6443, Log-Loss 0.6264
```

---

## Task 1: `sets_played` — normalizar en `src/data/loader.py`

**Files:**
- Modify: `src/data/loader.py`
- Modify: `tests/test_loader.py`

- [ ] **Step 1: Escribir los tests que fallan**

Agregar a `tests/test_loader.py` (agregar `_count_sets_played` al import existente
`from src.data.loader import _clean, _clean_wta, load_atp_matches, load_wta_matches`):

```python
class TestCountSetsPlayed:
    def test_two_straight_sets(self):
        assert _count_sets_played("6-4 6-2") == 2

    def test_tiebreak_set_counts_as_one(self):
        assert _count_sets_played("7-6(5) 6-4") == 2

    def test_three_sets(self):
        assert _count_sets_played("6-4 3-6 6-2") == 3

    def test_retirement_counts_the_partial_set(self):
        # confirmed decision: a RET set still involved real games played
        assert _count_sets_played("6-3 2-4 RET") == 2

    def test_walkover_is_zero_sets(self):
        assert _count_sets_played("W/O") == 0

    def test_missing_score_is_zero_sets(self):
        assert _count_sets_played(None) == 0
        assert _count_sets_played(float("nan")) == 0


def test_clean_computes_sets_played_for_atp():
    df = _clean(pd.read_csv(StringIO(SAMPLE_CSV)))
    # SAMPLE_CSV rows: "6-3 6-4" and "7-5 6-3" — both 2 sets
    assert list(df["sets_played"]) == [2, 2]


def test_clean_wta_computes_sets_played_from_wsets_lsets():
    df = _clean_wta(_wta_raw(), 2023)
    # WTA_CSV rows: Wsets/Lsets = (2,0), (2,1), (2,0) -> sets_played 2,3,2
    assert list(df["sets_played"]) == [2, 3, 2]


def test_clean_wta_sets_played_defaults_to_zero_when_columns_missing():
    raw = pd.read_csv(StringIO(WTA_CSV))  # original fixture predates Wsets/Lsets addition test
    raw = raw.drop(columns=["Wsets", "Lsets"], errors="ignore")
    df = _clean_wta(raw, 2023)
    assert list(df["sets_played"]) == [0, 0, 0]
```

Update the `WTA_CSV` fixture (near the top of the file) to include `Wsets`/`Lsets`
columns, and add `_count_sets_played` to the existing import line:

```python
from src.data.loader import _clean, _clean_wta, _count_sets_played, load_atp_matches, load_wta_matches

WTA_CSV = (
    "Date,Surface,Winner,Loser,WRank,LRank,Wsets,Lsets,B365W,B365L\n"
    "15/01/2023,Hard,Swiatek I.,Kvitova P.,1,6,2,0,1.15,5.50\n"
    "16/01/2023,Clay,Sabalenka A.,Rybakina E.,5,25,2,1,1.80,1.95\n"
    "17/01/2023,Grass,Gauff C.,Pegula J.,6,4,2,0,2.10,1.75\n"
)
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_loader.py -v`
Expected: FAIL — `ImportError: cannot import name '_count_sets_played'`

- [ ] **Step 3: Implementar en `src/data/loader.py`**

Agregar cerca del top del archivo (después de `_SURFACE_MAP`):

```python
import re

_SET_SCORE_PATTERN = re.compile(r"\d+-\d+(?:\(\d+\))?")


def _count_sets_played(score) -> int:
    """Count set-score tokens in a raw score string (e.g. "7-6(5) 6-4" -> 2).

    Retirement scores ("6-3 2-4 RET") count the partial set — real games
    were played. Walkovers ("W/O") and missing/non-string scores -> 0.
    """
    if not isinstance(score, str):
        return 0
    return len(_SET_SCORE_PATTERN.findall(score))
```

En `_clean` (ATP), agregar antes del `return`:
```python
    df["sets_played"] = df["score"].apply(_count_sets_played)
```

En `_clean_wta` (WTA), agregar antes del `return`:
```python
    w_sets = pd.to_numeric(df["Wsets"], errors="coerce") if "Wsets" in df.columns else pd.Series(0.0, index=df.index)
    l_sets = pd.to_numeric(df["Lsets"], errors="coerce") if "Lsets" in df.columns else pd.Series(0.0, index=df.index)
    df["sets_played"] = (w_sets.fillna(0) + l_sets.fillna(0)).astype(int)
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_loader.py -v`
Expected: PASS (todos)

- [ ] **Step 5: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: mismo conteo que antes + los tests nuevos, nada roto (agregar una
columna nueva a un DataFrame no debería afectar nada que no la lea).

- [ ] **Step 6: Commit**

```bash
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: normalize sets_played column in ATP/WTA loaders"
```

---

## Task 2: `FeatureBuilder` — historial de carga (`_workload_history`)

**Files:**
- Modify: `src/features/engineering.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Escribir los tests que fallan**

Agregar a `tests/test_features.py`:

```python
class TestWorkloadHistory:
    def test_empty_for_new_player(self):
        fb = FeatureBuilder()
        assert fb.workload_history("A") == []

    def test_records_sets_played_on_update(self):
        fb = FeatureBuilder()
        fb.update("A", "B", "hard", date(2023, 1, 1), sets_played=3)
        assert fb.workload_history("A") == [(date(2023, 1, 1), 3)]
        assert fb.workload_history("B") == [(date(2023, 1, 1), 3)]

    def test_sets_played_defaults_to_zero(self):
        """Existing update() call sites (H2H/form/rest tests) don't pass
        sets_played -- must not break them."""
        fb = FeatureBuilder()
        fb.update("A", "B", "hard", date(2023, 1, 1))
        assert fb.workload_history("A") == [(date(2023, 1, 1), 0)]

    def test_accumulates_across_matches_in_order(self):
        fb = FeatureBuilder()
        fb.update("A", "B", "hard", date(2023, 1, 1), sets_played=2)
        fb.update("A", "C", "hard", date(2023, 1, 10), sets_played=3)
        assert fb.workload_history("A") == [
            (date(2023, 1, 1), 2), (date(2023, 1, 10), 3),
        ]
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_features.py::TestWorkloadHistory -v`
Expected: FAIL — `AttributeError: 'FeatureBuilder' object has no attribute 'workload_history'`

- [ ] **Step 3: Implementar en `src/features/engineering.py`**

En `__init__`, agregar junto a `self._last_match_date`:
```python
        # player -> [(match_date, sets_played)] — for fatigue/workload
        self._workload_history: Dict[str, List[Tuple[date, int]]] = defaultdict(list)
```

Reemplazar `update`:
```python
    def update(self, winner: str, loser: str, surface: str, match_date: date, sets_played: int = 0) -> None:
        self._history[winner].append((match_date, surface, True))
        self._history[loser].append((match_date, surface, False))
        self._h2h_matches[(winner, loser)].append((match_date, True))
        self._h2h_matches[(loser, winner)].append((match_date, False))
        self._workload_history[winner].append((match_date, sets_played))
        self._workload_history[loser].append((match_date, sets_played))
        self._last_match_date[winner] = match_date
        self._last_match_date[loser] = match_date
```

Agregar el accessor, junto a `match_dates`:
```python
    def workload_history(self, player: str) -> List[Tuple[date, int]]:
        """This player's own (match_date, sets_played) history, oldest first."""
        return list(self._workload_history[player])
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: PASS (todos, incluyendo los ~14 tests existentes de H2H/form/rest — no
deben verse afectados ya que `sets_played` tiene default `0`)

- [ ] **Step 5: Commit**

```bash
git add src/features/engineering.py tests/test_features.py
git commit -m "feat: track per-player workload history in FeatureBuilder"
```

---

## Task 3: `fatigue_multiplier` — función pura en `src/features/decay.py`

**Files:**
- Modify: `src/features/decay.py`
- Modify: `tests/test_decay.py`

- [ ] **Step 1: Escribir los tests que fallan**

Agregar a `tests/test_decay.py` (agregar `fatigue_multiplier` al import de
`src.features.decay` existente):

```python
# ── fatigue_multiplier ────────────────────────────────────────────────────────
# Hand-computed values — see docs/superpowers/specs/2026-07-28-fatigue-workload-design.md

class TestFatigueMultiplier:
    def test_no_history_returns_neutral(self):
        assert fatigue_multiplier([], date(2023, 6, 1)) == 1.0

    def test_single_light_match_barely_reduces_multiplier(self):
        # 1 match/2 sets, 3 days ago (inside both windows).
        # load_7d  = 0.5*min(1,1/3) + 0.5*min(1,2/6)  = 1/3
        # load_14d = 0.5*min(1,1/5) + 0.5*min(1,2/10) = 0.2
        # fatigue_load = 0.6*(1/3) + 0.4*0.2 = 0.28
        # multiplier = 1 - 0.15*0.28 = 0.958
        history = [(date(2023, 5, 29), 2)]
        assert fatigue_multiplier(history, date(2023, 6, 1)) == pytest.approx(0.958)

    def test_max_load_in_both_windows_hits_penalty_cap(self):
        # 3 matches/6 sets inside the last 7 days (saturates the 7d window),
        # plus 2 more matches/4 sets between day 8-14 (saturates the 14d
        # window too: 5 matches/10 sets total). Both windows load=1.0.
        # fatigue_load = 0.6*1.0 + 0.4*1.0 = 1.0
        # multiplier = 1 - 0.15*1.0 = 0.85 (the maximum possible penalty)
        history = [
            (date(2023, 5, 20), 2), (date(2023, 5, 22), 2),  # 8-10 days ago
            (date(2023, 5, 27), 1), (date(2023, 5, 28), 2), (date(2023, 5, 30), 3),  # <=7 days ago
        ]
        assert fatigue_multiplier(history, date(2023, 6, 1)) == pytest.approx(0.85)

    def test_match_outside_7d_window_still_counts_in_14d(self):
        # 1 match/2 sets, 10 days ago: excluded from the 7d window (>7),
        # included in the 14d window (<=14).
        # load_7d = 0; load_14d = 0.5*min(1,1/5) + 0.5*min(1,2/10) = 0.2
        # fatigue_load = 0.6*0 + 0.4*0.2 = 0.08
        # multiplier = 1 - 0.15*0.08 = 0.988
        history = [(date(2023, 5, 22), 2)]
        assert fatigue_multiplier(history, date(2023, 6, 1)) == pytest.approx(0.988)
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_decay.py::TestFatigueMultiplier -v`
Expected: FAIL — `ImportError: cannot import name 'fatigue_multiplier'`

- [ ] **Step 3: Implementar en `src/features/decay.py`**

Agregar las constantes junto a las existentes (después de `AGE_PENALTY_FLOOR`):
```python
FATIGUE_WINDOW_SHORT_DAYS = 7
FATIGUE_WINDOW_LONG_DAYS  = 14
FATIGUE_MATCH_TARGET_7D   = 3
FATIGUE_SET_TARGET_7D     = 6
FATIGUE_MATCH_TARGET_14D  = 5
FATIGUE_SET_TARGET_14D    = 10
FATIGUE_RECENT_WEIGHT     = 0.6
FATIGUE_PENALTY_MAX       = 0.15
```

Agregar las funciones, después de `rust_factor` y antes de `calculate_decay_features`:
```python
def _count_window(
    workload_history: List[Tuple[date, int]], current_date: date, window_days: int,
) -> Tuple[int, int]:
    """(matches, sets) played strictly before current_date, within the
    trailing window_days — same cutoff convention as rust_factor
    (cutoff <= d < current_date, never includes the current match)."""
    cutoff = current_date - timedelta(days=window_days)
    sets_in_window = [sets for d, sets in workload_history if cutoff <= d < current_date]
    return len(sets_in_window), sum(sets_in_window)


def _window_load(matches: int, sets: int, match_target: int, set_target: int) -> float:
    match_load = min(1.0, matches / match_target)
    set_load = min(1.0, sets / set_target)
    return 0.5 * match_load + 0.5 * set_load


def fatigue_multiplier(workload_history: List[Tuple[date, int]], current_date: date) -> float:
    """1.0 = fresh, down to (1 - FATIGUE_PENALTY_MAX) at maximum load in
    both the 7-day and 14-day windows. A player with no history returns
    1.0 (nothing to judge overload against — same convention as
    rust_factor for brand-new players)."""
    matches_7d, sets_7d = _count_window(workload_history, current_date, FATIGUE_WINDOW_SHORT_DAYS)
    matches_14d, sets_14d = _count_window(workload_history, current_date, FATIGUE_WINDOW_LONG_DAYS)

    load_7d = _window_load(matches_7d, sets_7d, FATIGUE_MATCH_TARGET_7D, FATIGUE_SET_TARGET_7D)
    load_14d = _window_load(matches_14d, sets_14d, FATIGUE_MATCH_TARGET_14D, FATIGUE_SET_TARGET_14D)

    fatigue_load = FATIGUE_RECENT_WEIGHT * load_7d + (1 - FATIGUE_RECENT_WEIGHT) * load_14d
    return 1.0 - FATIGUE_PENALTY_MAX * fatigue_load
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_decay.py -v`
Expected: PASS (todos, incluyendo `TestFatigueMultiplier`)

- [ ] **Step 5: Commit**

```bash
git add src/features/decay.py tests/test_decay.py
git commit -m "feat: add fatigue_multiplier (7d/14d workload) to src/features/decay.py"
```

---

## Task 4: Integrar `fatigue_multiplier` en `calculate_decay_features`

**Files:**
- Modify: `src/features/decay.py`
- Modify: `tests/test_decay.py`

- [ ] **Step 1: Escribir el test que falla**

Agregar a `tests/test_decay.py`:

```python
class TestCalculateDecayFeaturesFatigue:
    def test_includes_fatigue_multiplier_and_folds_into_adjusted_elo(self):
        tracker = EloHistoryTracker()
        heavy_workload = [
            (date(2023, 5, 27), 1), (date(2023, 5, 28), 2), (date(2023, 5, 30), 3),
        ]  # matches Task 3's max-load-7d case in spirit (not exact — just needs < 1.0)
        result = calculate_decay_features(
            historical_elo_surface=1700.0, tracker=tracker, player="P",
            surface="clay", match_dates=[], current_date=date(2023, 6, 1),
            player_age=None, workload_history=heavy_workload,
        )
        assert result["fatigue_multiplier"] < 1.0
        assert result["adjusted_elo_surface"] == pytest.approx(
            1700.0 * result["age_multiplier"] * result["rust_factor"] * result["fatigue_multiplier"]
        )

    def test_workload_history_defaults_to_empty_and_stays_neutral(self):
        """Backward compatibility: existing callers that don't pass
        workload_history (this file's other tests, walkforward/value_analysis
        before Tasks 6-7 wire it up) must keep working with neutral fatigue."""
        tracker = EloHistoryTracker()
        result = calculate_decay_features(
            historical_elo_surface=1700.0, tracker=tracker, player="P",
            surface="clay", match_dates=[], current_date=date(2023, 6, 1),
            player_age=None,
        )
        assert result["fatigue_multiplier"] == 1.0
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_decay.py::TestCalculateDecayFeaturesFatigue -v`
Expected: FAIL — `KeyError: 'fatigue_multiplier'` (or `TypeError` on the first
test if `workload_history` isn't accepted yet)

- [ ] **Step 3: Implementar en `src/features/decay.py`**

Reemplazar la firma y cuerpo de `calculate_decay_features`:

```python
def calculate_decay_features(
    historical_elo_surface: float,
    tracker: EloHistoryTracker,
    player: str,
    surface: str,
    match_dates: List[date],
    current_date: date,
    player_age: Optional[float],
    workload_history: Optional[List[Tuple[date, int]]] = None,
) -> dict:
    """Per-player decay-adjustment features for one side of a matchup.

    historical_elo_surface: the player's pre-match effective surface Elo
        (elo.get_effective_rating(player, surface)) — passed in rather than
        recomputed here since callers already have it for elo_diff.
    match_dates: this player's own past match dates (any surface), used
        only for the rust_factor recency window.
    workload_history: this player's own (match_date, sets_played) history,
        used for fatigue_multiplier — defaults to empty (neutral fatigue)
        for callers that don't track it yet.
    """
    rolling = tracker.rolling_elo(player, surface)
    rolling_elo_diff = 0.0 if rolling is None else historical_elo_surface - rolling

    mult = age_multiplier(player_age)
    rust = rust_factor(match_dates, current_date)
    fatigue = fatigue_multiplier(workload_history or [], current_date)
    adjusted_elo_surface = historical_elo_surface * mult * rust * fatigue

    return {
        "rolling_elo_diff":     rolling_elo_diff,
        "age_multiplier":       mult,
        "rust_factor":          rust,
        "fatigue_multiplier":   fatigue,
        "adjusted_elo_surface": adjusted_elo_surface,
    }
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_decay.py -v`
Expected: PASS (todos)

- [ ] **Step 5: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: revisar con cuidado cualquier fallo en `tests/test_build_match_features.py`
o `tests/test_value_analysis.py` — no deberían fallar (llaman
`calculate_decay_features` sin `workload_history`, que ahora es opcional con
default neutro), pero confirmar antes de continuar.

- [ ] **Step 6: Commit**

```bash
git add src/features/decay.py tests/test_decay.py
git commit -m "feat: fold fatigue_multiplier into calculate_decay_features/adjusted_elo_surface"
```

---

## Task 5: Agregar `fatigue_multiplier_diff` a `FEATURE_COLS`

**Files:**
- Modify: `src/features/__init__.py`

- [ ] **Step 1: Actualizar la constante**

Gracias a la consolidación de 3a, este es un cambio en un solo archivo:

```python
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff", "adjusted_elo_diff",
]
```

- [ ] **Step 2: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: **fallos esperados aquí** — `_MIRROR_FLIP_COLS` y las filas
construidas en `walkforward.py`/`value_analysis.py` todavía no producen
`fatigue_multiplier_diff`, así que cualquier código que itere `FEATURE_COLS`
esperando encontrar esa clave en un dict de features (`build_prediction_features`,
`build_match_features`) va a fallar con `KeyError`. Esto es esperado y se
arregla en las Tasks 6-7 — no revertir este cambio, seguir adelante.

- [ ] **Step 3: Commit**

```bash
git add src/features/__init__.py
git commit -m "feat: add fatigue_multiplier_diff to FEATURE_COLS"
```

---

## Task 6: Conectar en `src/backtest/walkforward.py` (entrenamiento)

**Files:**
- Modify: `src/backtest/walkforward.py`
- Modify: `tests/test_build_match_features.py`

- [ ] **Step 1: Escribir el test que falla**

Agregar `sets_played: int = 0` como parámetro al helper `_row` existente en
`tests/test_build_match_features.py` (agregarlo a la firma y al dict que
retorna, con clave `"sets_played"`), y agregar:

```python
def test_mirror_row_negates_fatigue_multiplier_diff():
    """fatigue_multiplier_diff is a plain signed a-b difference (unlike
    h2h_rate) -- simple negation on the mirror row, same as age/rust."""
    df = pd.DataFrame(
        [
            _row("A", "B", "clay", date(2023, 1, 1), sets_played=3),
            _row("A", "C", "clay", date(2023, 1, 3), sets_played=3),
            _row("A", "D", "clay", date(2023, 1, 5), sets_played=3),
        ]
    )
    match_df = build_match_features(df, EloSystem(), FeatureBuilder())
    original = match_df[~match_df["is_mirror"]]
    mirror = match_df[match_df["is_mirror"]]

    # A racked up 3 heavy matches in 5 days -> A's own fatigue_multiplier_diff
    # (vs each fresh opponent) should be negative by the 3rd match.
    assert original.iloc[-1]["fatigue_multiplier_diff"] < 0
    for orig, mir in zip(original["fatigue_multiplier_diff"], mirror["fatigue_multiplier_diff"]):
        assert mir == pytest.approx(-orig)
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_build_match_features.py::test_mirror_row_negates_fatigue_multiplier_diff -v`
Expected: FAIL — `KeyError: 'fatigue_multiplier_diff'`

- [ ] **Step 3: Implementar en `src/backtest/walkforward.py`**

En el loop de `build_match_features`, extraer `sets_played` junto a los demás
campos de la fila (cerca de `w_age = _age_or_none(row.get("winner_age"))`):
```python
        sets_played = int(row.get("sets_played", 0))
```

Actualizar las dos llamadas a `calculate_decay_features` para pasar el
historial de carga:
```python
        w_decay = calculate_decay_features(
            elo_w, elo_tracker, winner, surface,
            feature_builder.match_dates(winner), match_date, w_age,
            feature_builder.workload_history(winner),
        )
        l_decay = calculate_decay_features(
            elo_l, elo_tracker, loser, surface,
            feature_builder.match_dates(loser), match_date, l_age,
            feature_builder.workload_history(loser),
        )
```

Agregar la columna a la fila (junto a `"adjusted_elo_diff"`):
```python
                "fatigue_multiplier_diff": w_decay["fatigue_multiplier"] - l_decay["fatigue_multiplier"],
```

Actualizar la llamada a `feature_builder.update` (post-match) para pasar
`sets_played`:
```python
        feature_builder.update(winner, loser, surface, match_date, sets_played=sets_played)
```

Agregar `"fatigue_multiplier_diff"` a `_MIRROR_FLIP_COLS`:
```python
_MIRROR_FLIP_COLS = (
    "elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff", "adjusted_elo_diff",
)
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_build_match_features.py -v`
Expected: PASS (todos)

- [ ] **Step 5: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: revisar `tests/test_value_analysis.py` con cuidado — todavía no
está conectado (Task 7), así que cualquier test ahí que dependa de
`FEATURE_COLS` completo (ej. `TestTrainLRPipeline`, que fitea un `Pipeline`
sobre un CSV sintético con las columnas viejas) puede fallar por
desalineamiento de columnas — si eso pasa, es esperado, se arregla en Task 7.

- [ ] **Step 6: Commit**

```bash
git add src/backtest/walkforward.py tests/test_build_match_features.py
git commit -m "feat: wire fatigue_multiplier_diff into build_match_features"
```

---

## Task 7: Conectar en `src/value_analysis.py` (predicción)

**Files:**
- Modify: `src/value_analysis.py`
- Modify: `tests/test_value_analysis.py`

- [ ] **Step 1: Escribir el test que falla**

Agregar a `tests/test_value_analysis.py`, cerca de `TestPredictMatchH2hWiring`:

```python
class TestPredictMatchFatigueWiring:
    def test_fatigue_multiplier_diff_reaches_lr_input_vector(self):
        elo = _fake_elo({"A": 1500.0, "B": 1500.0})
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.5, "h2h_matches": 0, "rest_days": 14.0,
            },
            match_dates=lambda p: [],
            workload_history=lambda p: (
                [(date(2026, 7, 18), 3), (date(2026, 7, 19), 2), (date(2026, 7, 20), 3)]
                if p == "A" else []
            ),
        )

        class _StubClf:
            def predict_proba(self, X):
                import numpy as _np
                idx = FEATURE_COLS.index("fatigue_multiplier_diff")
                diff = X[0][idx]
                # A is heavily loaded (fatigue_multiplier_diff < 0), B is fresh
                # -> lower P(A wins) than a neutral coin flip would give.
                p = 0.5 + diff  # arbitrary monotonic mapping, just needs to move away from 0.5
                return _np.array([[1 - p, p]])

        pred = predict_match(
            elo, fb, _StubClf(),
            "A", "B", "clay", date(2026, 7, 21),
            rank_lookup=None, age_lookup=None,
        )
        assert pred["p_a_raw"] < 0.5  # A's heavier recent workload should pull this below neutral
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestPredictMatchFatigueWiring -v`
Expected: FAIL — `AttributeError` (stub `fb` has no `workload_history` consumer
yet) o `KeyError: 'fatigue_multiplier_diff'`

- [ ] **Step 3: Implementar en `src/value_analysis.py`**

En `build_prediction_features`, actualizar las dos llamadas a
`calculate_decay_features`:
```python
    decay_a = calculate_decay_features(
        elo_a, tracker, pa_key, surface, fb.match_dates(pa_key), match_date, age_a,
        fb.workload_history(pa_key),
    )
    decay_b = calculate_decay_features(
        elo_b, tracker, pb_key, surface, fb.match_dates(pb_key), match_date, age_b,
        fb.workload_history(pb_key),
    )
```

Agregar la clave al dict retornado (junto a `"adjusted_elo_diff"`):
```python
        "fatigue_multiplier_diff": decay_a["fatigue_multiplier"] - decay_b["fatigue_multiplier"],
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py -v`

`TestTrainLRPipeline`/`TestLoadModelSnapshot` build a synthetic CSV fixture
hardcoding the 11-column schema — this will now fail with a shape/column
mismatch against `FEATURE_COLS`'s 12 columns. Find that fixture (grep for
`elo_diff.*elo_prob.*rank_diff` in `tests/test_value_analysis.py`) and add
`fatigue_multiplier_diff` to it with value `0.0` in every row (same neutral
value already used for the other decay-diff columns in that fixture, e.g.
`rolling_elo_diff`), in the exact same column position `FEATURE_COLS` now has
it (between `rust_factor_diff` and `adjusted_elo_diff`).

Expected after the fix: PASS (all tests in the file).

- [ ] **Step 5: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`

Apply the same fixture fix (add `fatigue_multiplier_diff: 0.0` at the same
position) to any other file with the same synthetic 11-column schema that
now fails: `tests/test_backtest.py`'s `_synthetic_features()` and
`tests/test_pipeline.py`'s `fake_snapshot` fixture are the known candidates
(both were flagged during 3a's research as places that independently
hardcode this schema). Expected after fixing all of them: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py
git commit -m "feat: wire fatigue_multiplier_diff into build_prediction_features"
```

(Si Step 4/5 revela fixtures sintéticas desactualizadas en otros archivos de
test, incluir esos cambios en este mismo commit — son parte de completar el
wiring, no un task aparte.)

---

## Task 8: Regenerar datasets, correr backtest, comparar contra baseline

**Files:**
- Regenerated (gitignored, no commit): `data/processed/atp_features.csv`, `data/processed/wta_features.csv`

- [ ] **Step 1: Regenerar y correr el backtest para ambos tours**

Run: `./tenis-env/Scripts/python.exe -m src.pipeline atp`
Run: `./tenis-env/Scripts/python.exe -m src.pipeline wta`

- [ ] **Step 2: Comparar contra el baseline (reutilizado de 3a, ver arriba)**

```
ATP baseline: Accuracy 0.6688, Log-Loss 0.6024
WTA baseline: Accuracy 0.6443, Log-Loss 0.6264
```

Reportar la diferencia exacta para cada tour.

- [ ] **Step 3: Decisión**

Igual que en 3a — el usuario decide si mantener, ajustar las constantes
(`FATIGUE_MATCH_TARGET_*`, `FATIGUE_SET_TARGET_*`, `FATIGUE_RECENT_WEIGHT`,
`FATIGUE_PENALTY_MAX`) y volver a validar, o revertir. No requiere commit de
código — solo verificación y reporte.

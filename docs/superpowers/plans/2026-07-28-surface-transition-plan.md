# Módulo 3, Sub-proyecto 3c: Transición de Superficie — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar `surface_transition_multiplier_diff` (columna #13) a `FEATURE_COLS`, penalizando temporalmente a un jugador que cambió de superficie respecto a su partido anterior, con severidad según qué tan distintas son las superficies, integrada como multiplicador de `adjusted_elo_surface`.

**Architecture:** El más liviano de los 3 sub-proyectos de Módulo 3 — no toca `src/data/loader.py` (todo el dato necesario ya vive en `FeatureBuilder._history`). Un accessor nuevo (`last_surface_and_date`) → función pura `surface_transition_multiplier` en `decay.py` → integrar en `calculate_decay_features` → conectar en los dos puntos de consumo (`walkforward.py`, `value_analysis.py`) → validar con backtest. A diferencia de Fatiga, esta vez `_LOG_FIELDS` y las fixtures sintéticas desactualizadas se incluyen desde el plan (ya sabemos que hace falta tocarlas, no son un descubrimiento tardío).

**Tech Stack:** Python, numpy, pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-07-28-surface-transition-design.md` — leer primero para las fórmulas exactas y el razonamiento.

**Baseline reutilizado de 3b** (mismo estado de `master`, no hace falta re-correr):
```
ATP: Accuracy 0.6692, Log-Loss 0.6019
WTA: Accuracy 0.6445, Log-Loss 0.6261
```

---

## Task 1: Confirmar baseline (sin cambios de código)

**Files:** ninguno.

- [ ] **Step 1: Confirmar que el baseline de 3b sigue vigente**

Run: `git log --oneline -1` — confirmar que `HEAD` es el commit de merge de 3b
(`eb0ba30` o su equivalente si hubo más commits desde entonces). Si `HEAD`
coincide, el baseline de arriba es válido tal cual, sin re-correr el
pipeline. Si hay commits nuevos desde `eb0ba30` que pudieran afectar el
modelo (no solo docs), correr `python -m src.pipeline atp`/`wta` de nuevo
para capturar un baseline fresco antes de continuar.

No requiere commit.

---

## Task 2: `FeatureBuilder.last_surface_and_date` — accessor nuevo, sin storage nuevo

**Files:**
- Modify: `src/features/engineering.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Escribir los tests que fallan**

Agregar a `tests/test_features.py`:

```python
class TestLastSurfaceAndDate:
    def test_none_for_new_player(self):
        fb = FeatureBuilder()
        assert fb.last_surface_and_date("A") is None

    def test_returns_most_recent_match(self):
        fb = FeatureBuilder()
        fb.update("A", "B", "clay", date(2023, 1, 1))
        fb.update("A", "C", "grass", date(2023, 1, 10))
        assert fb.last_surface_and_date("A") == (date(2023, 1, 10), "grass")

    def test_tracks_both_winner_and_loser(self):
        fb = FeatureBuilder()
        fb.update("A", "B", "hard", date(2023, 1, 1))
        assert fb.last_surface_and_date("A") == (date(2023, 1, 1), "hard")
        assert fb.last_surface_and_date("B") == (date(2023, 1, 1), "hard")
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_features.py::TestLastSurfaceAndDate -v`
Expected: FAIL — `AttributeError: 'FeatureBuilder' object has no attribute 'last_surface_and_date'`

- [ ] **Step 3: Implementar en `src/features/engineering.py`**

Agregar el método, junto a `match_dates`/`workload_history` (no requiere
tocar `__init__` ni `update` — se lee de `self._history`, que ya existe):

```python
    def last_surface_and_date(self, player: str) -> Optional[Tuple[date, str]]:
        """This player's most recent match's (date, surface), or None if
        they have no history yet."""
        history = self._history[player]
        if not history:
            return None
        d, s, _ = history[-1]
        return (d, s)
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: PASS (todos, incluyendo los ~18 tests existentes — este accessor
no toca ningún estado existente, solo lee)

- [ ] **Step 5: Commit**

```bash
git add src/features/engineering.py tests/test_features.py
git commit -m "feat: add FeatureBuilder.last_surface_and_date accessor"
```

---

## Task 3: `surface_transition_multiplier` — función pura en `src/features/decay.py`

**Files:**
- Modify: `src/features/decay.py`
- Modify: `tests/test_decay.py`

- [ ] **Step 1: Escribir los tests que fallan**

Agregar a `tests/test_decay.py` (agregar `surface_transition_multiplier` al
import de `src.features.decay` existente):

```python
# ── surface_transition_multiplier ────────────────────────────────────────────
# Hand-computed values — see docs/superpowers/specs/2026-07-28-surface-transition-design.md

class TestSurfaceTransitionMultiplier:
    def test_no_history_returns_neutral(self):
        assert surface_transition_multiplier(None, "clay", date(2023, 6, 1)) == 1.0

    def test_same_surface_returns_neutral_regardless_of_days(self):
        last = (date(2023, 5, 20), "clay")
        assert surface_transition_multiplier(last, "clay", date(2023, 6, 1)) == 1.0

    def test_immediate_clay_to_grass_transition_is_max_penalty(self):
        # days_since=0, severity=1.0 (clay<->grass), recency=(10-0)/10=1.0
        # multiplier = 1 - 0.10*1.0*1.0 = 0.90
        last = (date(2023, 6, 1), "clay")
        assert surface_transition_multiplier(last, "grass", date(2023, 6, 1)) == pytest.approx(0.90)

    def test_hard_transition_is_less_severe(self):
        # days_since=0, severity=0.5 (hard<->clay), recency=1.0
        # multiplier = 1 - 0.10*0.5*1.0 = 0.95
        last = (date(2023, 6, 1), "hard")
        assert surface_transition_multiplier(last, "clay", date(2023, 6, 1)) == pytest.approx(0.95)

    def test_decays_linearly_at_midpoint(self):
        # days_since=5, severity=1.0 (clay<->grass), recency=(10-5)/10=0.5
        # multiplier = 1 - 0.10*1.0*0.5 = 0.95
        last = (date(2023, 5, 27), "clay")
        assert surface_transition_multiplier(last, "grass", date(2023, 6, 1)) == pytest.approx(0.95)

    def test_boundary_day_10_is_neutral(self):
        # days_since=10 -> outside [0,10), fully decayed
        last = (date(2023, 5, 22), "clay")
        assert surface_transition_multiplier(last, "grass", date(2023, 6, 1)) == 1.0

    def test_day_9_still_has_small_penalty(self):
        # days_since=9, severity=1.0, recency=(10-9)/10=0.1
        # multiplier = 1 - 0.10*1.0*0.1 = 0.99
        last = (date(2023, 5, 23), "clay")
        assert surface_transition_multiplier(last, "grass", date(2023, 6, 1)) == pytest.approx(0.99)
```

- [ ] **Step 2: Correr los tests para verificar que fallan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_decay.py::TestSurfaceTransitionMultiplier -v`
Expected: FAIL — `ImportError: cannot import name 'surface_transition_multiplier'`

- [ ] **Step 3: Implementar en `src/features/decay.py`**

Agregar las constantes junto a las existentes (después de las `FATIGUE_*`):
```python
SURFACE_TRANSITION_WINDOW_DAYS  = 10
SURFACE_TRANSITION_PENALTY_MAX  = 0.10
SURFACE_TRANSITION_SEVERITY = {
    frozenset({"clay", "grass"}): 1.0,
    frozenset({"clay", "hard"}): 0.5,
    frozenset({"grass", "hard"}): 0.5,
}
_DEFAULT_TRANSITION_SEVERITY = 0.5   # carpet/unknown pairs — rare, safe default
```

Agregar la función, después de `fatigue_multiplier` y antes de `calculate_decay_features`:
```python
def surface_transition_multiplier(
    last_surface_and_date: Optional[Tuple[date, str]],
    current_surface: str,
    current_date: date,
) -> float:
    """1.0 = no recent surface change (no history, same surface, or the
    switch happened more than SURFACE_TRANSITION_WINDOW_DAYS ago). Down to
    (1 - SURFACE_TRANSITION_PENALTY_MAX * severity) immediately after
    switching, linearly decaying back to 1.0 over the window."""
    if last_surface_and_date is None:
        return 1.0
    last_date, last_surface = last_surface_and_date
    if last_surface == current_surface:
        return 1.0
    days_since = (current_date - last_date).days
    if not (0 <= days_since < SURFACE_TRANSITION_WINDOW_DAYS):
        return 1.0
    severity = SURFACE_TRANSITION_SEVERITY.get(
        frozenset({last_surface, current_surface}), _DEFAULT_TRANSITION_SEVERITY
    )
    recency = (SURFACE_TRANSITION_WINDOW_DAYS - days_since) / SURFACE_TRANSITION_WINDOW_DAYS
    return 1.0 - SURFACE_TRANSITION_PENALTY_MAX * severity * recency
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_decay.py -v`
Expected: PASS (todos, incluyendo `TestSurfaceTransitionMultiplier`)

- [ ] **Step 5: Commit**

```bash
git add src/features/decay.py tests/test_decay.py
git commit -m "feat: add surface_transition_multiplier to src/features/decay.py"
```

---

## Task 4: Integrar `surface_transition_multiplier` en `calculate_decay_features`

**Files:**
- Modify: `src/features/decay.py`
- Modify: `tests/test_decay.py`

- [ ] **Step 1: Escribir el test que falla**

Agregar a `tests/test_decay.py`:

```python
class TestCalculateDecayFeaturesSurfaceTransition:
    def test_includes_surface_transition_multiplier_and_folds_into_adjusted_elo(self):
        tracker = EloHistoryTracker()
        result = calculate_decay_features(
            historical_elo_surface=1700.0, tracker=tracker, player="P",
            surface="grass", match_dates=[], current_date=date(2023, 6, 1),
            player_age=None, last_surface_and_date=(date(2023, 5, 29), "clay"),
        )
        assert result["surface_transition_multiplier"] < 1.0
        assert result["adjusted_elo_surface"] == pytest.approx(
            1700.0 * result["age_multiplier"] * result["rust_factor"]
            * result["fatigue_multiplier"] * result["surface_transition_multiplier"]
        )

    def test_last_surface_and_date_defaults_to_none_and_stays_neutral(self):
        """Backward compatibility: existing callers (this file's other
        tests) don't pass last_surface_and_date -- must keep working."""
        tracker = EloHistoryTracker()
        result = calculate_decay_features(
            historical_elo_surface=1700.0, tracker=tracker, player="P",
            surface="clay", match_dates=[], current_date=date(2023, 6, 1),
            player_age=None,
        )
        assert result["surface_transition_multiplier"] == 1.0
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_decay.py::TestCalculateDecayFeaturesSurfaceTransition -v`
Expected: FAIL — `TypeError`/`KeyError`

- [ ] **Step 3: Implementar en `src/features/decay.py`**

Actualizar la firma y cuerpo de `calculate_decay_features`:

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
    last_surface_and_date: Optional[Tuple[date, str]] = None,
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
    last_surface_and_date: this player's most recent (match_date, surface),
        used for surface_transition_multiplier — defaults to None (neutral,
        no recent transition) for callers that don't track it yet.
    """
    rolling = tracker.rolling_elo(player, surface)
    rolling_elo_diff = 0.0 if rolling is None else historical_elo_surface - rolling

    mult = age_multiplier(player_age)
    rust = rust_factor(match_dates, current_date)
    fatigue = fatigue_multiplier(workload_history or [], current_date)
    transition = surface_transition_multiplier(last_surface_and_date, surface, current_date)
    adjusted_elo_surface = historical_elo_surface * mult * rust * fatigue * transition

    return {
        "rolling_elo_diff":                rolling_elo_diff,
        "age_multiplier":                  mult,
        "rust_factor":                     rust,
        "fatigue_multiplier":              fatigue,
        "surface_transition_multiplier":   transition,
        "adjusted_elo_surface":            adjusted_elo_surface,
    }
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_decay.py -v`
Expected: PASS (todos)

- [ ] **Step 5: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: sin fallos nuevos (parámetro opcional, backward compatible —
`tests/test_build_match_features.py`/`tests/test_value_analysis.py` no
deberían verse afectados todavía).

- [ ] **Step 6: Commit**

```bash
git add src/features/decay.py tests/test_decay.py
git commit -m "feat: fold surface_transition_multiplier into calculate_decay_features/adjusted_elo_surface"
```

---

## Task 5: Agregar `surface_transition_multiplier_diff` a `FEATURE_COLS` (columna #13)

**Files:**
- Modify: `src/features/__init__.py`

- [ ] **Step 1: Actualizar la constante**

```python
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff", "surface_transition_multiplier_diff",
    "adjusted_elo_diff",
]
```

- [ ] **Step 2: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`

**Fallos esperados aquí** — mismo patrón que en Fatiga (Sub-proyecto 3b,
Task 5): cualquier código que construya un dict/fila esperando las 13
columnas o lea `data/processed/*.csv`/fixtures sintéticas del formato de
12 columnas va a fallar (`KeyError`/`ValueError` de `csv.DictWriter`). Esto
es esperado — se arregla en las Tasks 6-7. No revertir este cambio.

Reportar cuántos tests fallan y en qué archivos, como referencia (debería
ser una lista similar a la de 3b: `test_backtest.py`, `test_pipeline.py`,
`test_value_analysis.py`).

- [ ] **Step 3: Commit**

```bash
git add src/features/__init__.py
git commit -m "feat: add surface_transition_multiplier_diff to FEATURE_COLS"
```

---

## Task 6: Conectar en `src/backtest/walkforward.py` (entrenamiento)

**Files:**
- Modify: `src/backtest/walkforward.py`
- Modify: `tests/test_build_match_features.py`

- [ ] **Step 1: Escribir el test que falla**

Agregar a `tests/test_build_match_features.py`:

```python
def test_mirror_row_negates_surface_transition_multiplier_diff():
    """surface_transition_multiplier_diff is a plain signed a-b difference
    (like age/rust/fatigue) -- simple negation on the mirror row."""
    df = pd.DataFrame(
        [
            _row("A", "B", "clay", date(2023, 1, 1)),
            _row("A", "C", "grass", date(2023, 1, 5)),
        ]
    )
    match_df = build_match_features(df, EloSystem(), FeatureBuilder())
    original = match_df[~match_df["is_mirror"]]
    mirror = match_df[match_df["is_mirror"]]

    # A switched clay (1/1) -> grass (1/5), 4 days later; C is a fresh
    # opponent -> A's diff should be negative on the 2nd match.
    assert original.iloc[-1]["surface_transition_multiplier_diff"] < 0
    for orig, mir in zip(
        original["surface_transition_multiplier_diff"], mirror["surface_transition_multiplier_diff"]
    ):
        assert mir == pytest.approx(-orig)
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_build_match_features.py::test_mirror_row_negates_surface_transition_multiplier_diff -v`
Expected: FAIL — `KeyError: 'surface_transition_multiplier_diff'`

- [ ] **Step 3: Implementar en `src/backtest/walkforward.py`**

Actualizar las dos llamadas a `calculate_decay_features` para pasar el
último surface+date de cada lado — **leído ANTES de `feature_builder.update()`
para este partido, preservando no-lookahead (mismo lugar donde ya se leen
`match_dates`/`workload_history`)**:
```python
        w_decay = calculate_decay_features(
            elo_w, elo_tracker, winner, surface,
            feature_builder.match_dates(winner), match_date, w_age,
            feature_builder.workload_history(winner),
            feature_builder.last_surface_and_date(winner),
        )
        l_decay = calculate_decay_features(
            elo_l, elo_tracker, loser, surface,
            feature_builder.match_dates(loser), match_date, l_age,
            feature_builder.workload_history(loser),
            feature_builder.last_surface_and_date(loser),
        )
```

Agregar la columna a la fila (junto a `"fatigue_multiplier_diff"`, antes de
`"adjusted_elo_diff"`):
```python
                "surface_transition_multiplier_diff": w_decay["surface_transition_multiplier"] - l_decay["surface_transition_multiplier"],
```

Agregar `"surface_transition_multiplier_diff"` a `_MIRROR_FLIP_COLS`:
```python
_MIRROR_FLIP_COLS = (
    "elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff", "surface_transition_multiplier_diff",
    "adjusted_elo_diff",
)
```

(`feature_builder.update()` — no changes needed, `last_surface_and_date`
reads from `_history`, which `update()` already populates.)

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_build_match_features.py -v`
Expected: PASS (todos)

- [ ] **Step 5: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: los fallos de Task 5 en `test_backtest.py`/`test_pipeline.py`/
`test_value_analysis.py` siguen ahí (se arreglan en Task 7) — confirmar que
`test_build_match_features.py` pasa completo y que el conteo total de
fallos no creció más allá de lo esperado por agregar la columna a
`_MIRROR_FLIP_COLS` (mismo patrón que pasó en 3b Task 6 con
`test_backtest.py` — si eso vuelve a pasar acá, es igual de esperado, no
hace falta investigarlo de nuevo).

- [ ] **Step 6: Commit**

```bash
git add src/backtest/walkforward.py tests/test_build_match_features.py
git commit -m "feat: wire surface_transition_multiplier_diff into build_match_features"
```

---

## Task 7: Conectar en `src/value_analysis.py` (predicción) + cerrar todo el schema drift

**Files:**
- Modify: `src/value_analysis.py`
- Modify: `tests/test_value_analysis.py`, `tests/test_backtest.py`, `tests/test_pipeline.py`

Esta task agrupa TODO lo necesario para volver a dejar la suite 100% verde
(mismo alcance que tuvo que cubrir 3b Task 7, pero esta vez todo va desde
el principio del task en vez de descubrirse a mitad de camino).

- [ ] **Step 1: Escribir el test que falla — wiring de predicción**

Agregar a `tests/test_value_analysis.py`, cerca de `TestPredictMatchFatigueWiring`:

```python
class TestPredictMatchSurfaceTransitionWiring:
    def test_surface_transition_multiplier_diff_reaches_lr_input_vector(self):
        elo = _fake_elo({"A": 1500.0, "B": 1500.0})
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.5, "h2h_matches": 0, "rest_days": 14.0,
            },
            match_dates=lambda p: [],
            workload_history=lambda p: [],
            last_surface_and_date=lambda p: (
                (date(2026, 7, 18), "clay") if p == "A" else None
            ),
        )

        class _StubClf:
            def predict_proba(self, X):
                import numpy as _np
                idx = FEATURE_COLS.index("surface_transition_multiplier_diff")
                diff = X[0][idx]
                p = 0.5 + diff
                return _np.array([[1 - p, p]])

        pred = predict_match(
            elo, fb, _StubClf(),
            "A", "B", "grass", date(2026, 7, 21),
            rank_lookup=None, age_lookup=None,
        )
        assert pred["p_a_raw"] < 0.5  # A just switched clay->grass 3 days ago
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestPredictMatchSurfaceTransitionWiring -v`
Expected: FAIL

- [ ] **Step 3: Implementar el wiring en `src/value_analysis.py`**

En `build_prediction_features`, actualizar las dos llamadas a
`calculate_decay_features`:
```python
    decay_a = calculate_decay_features(
        elo_a, tracker, pa_key, surface, fb.match_dates(pa_key), match_date, age_a,
        fb.workload_history(pa_key), fb.last_surface_and_date(pa_key),
    )
    decay_b = calculate_decay_features(
        elo_b, tracker, pb_key, surface, fb.match_dates(pb_key), match_date, age_b,
        fb.workload_history(pb_key), fb.last_surface_and_date(pb_key),
    )
```

Agregar al dict retornado (junto a `"fatigue_multiplier_diff"`, antes de
`"adjusted_elo_diff"`):
```python
        "surface_transition_multiplier_diff": decay_a["surface_transition_multiplier"] - decay_b["surface_transition_multiplier"],
```

- [ ] **Step 4: `_LOG_FIELDS` — tercera copia del schema (ya conocido, no es un descubrimiento nuevo esta vez)**

`_LOG_FIELDS` en `src/value_analysis.py` (~línea 682) es una lista de
encabezado CSV separada, consumida por `csv.DictWriter` con
`extrasaction="raise"` estricto. Agregar `"surface_transition_multiplier_diff"`
en la misma posición relativa que en `FEATURE_COLS` (entre
`"fatigue_multiplier_diff"` y `"adjusted_elo_diff"`).

- [ ] **Step 5: Fixtures sintéticas desactualizadas (5 ubicaciones conocidas)**

Cada una necesita `"surface_transition_multiplier_diff"` insertada entre
`"fatigue_multiplier_diff"` y `"adjusted_elo_diff"` (o el índice
posicional equivalente si es un array crudo, no un dict):

1. **`tests/test_backtest.py::_synthetic_features()`** — agregar
   `"surface_transition_multiplier_diff": rng.uniform(-0.10, 0.10, n),` al
   dict `original` (rango `±0.10` = `SURFACE_TRANSITION_PENALTY_MAX`), y
   agregar el mismo nombre de columna a la tupla de mirror-flip inline
   unas líneas más abajo.
2. **`tests/test_value_analysis.py::TestTrainLRPipeline._write_synthetic_features_csv`**
   — agregar `"surface_transition_multiplier_diff": rng.uniform(-0.10, 0.10, n),`.
3. **Tres arrays `np.array([[...]])` de 12 elementos en `TestTrainLRPipeline`**
   (ya tienen 12 desde 3b, no 11 — insertar un 13er elemento `0.0` (neutro)
   entre la posición de `fatigue_multiplier_diff` y `adjusted_elo_diff`,
   es decir, como nuevo índice 11, empujando el viejo índice 11 a 12):
   - `test_predict_proba_returns_valid_probabilities`
   - `test_predictions_are_deterministic_across_calls`
   - `test_scaling_actually_applied_matches_manual_pipeline` (variable `X_query`)
4. **`tests/test_value_analysis.py::TestLoadModelSnapshot._write_features_csv`**
   — agregar la columna igual que en (2).
5. **`tests/test_pipeline.py::fake_snapshot`** — agregar la columna igual
   que en (2).

Antes de editar, releer cada sección primero para confirmar el contenido
actual exacto (puede haber cambiado levemente) — no asumir ciegamente que
los números de línea de tareas anteriores siguen siendo válidos.

- [ ] **Step 6: `SimpleNamespace` stubs — agregar `last_surface_and_date`**

Grepear `tests/test_value_analysis.py` por `workload_history=lambda` (todas
las instancias que ya tienen ese kwarg desde 3b también necesitan
`last_surface_and_date=lambda p: None` agregado, mismo motivo: una vez que
`build_prediction_features` llama `fb.last_surface_and_date(...)`
incondicionalmente, cualquier stub sin ese método rompe con
`AttributeError`).

- [ ] **Step 7: Correr la suite completa y confirmar 0 fallos**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`

**Criterio de aceptación de esta task: 100% verde, cero fallos.** Si algo
sigue fallando después de los Steps 1-6, investigar antes de continuar — no
es aceptable cerrar esta task con la suite en rojo.

- [ ] **Step 8: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py tests/test_backtest.py tests/test_pipeline.py
git commit -m "feat: wire surface_transition_multiplier_diff into build_prediction_features and fix stale FEATURE_COLS fixtures"
```

---

## Task 8: Regenerar datasets, correr backtest, comparar contra baseline

**Files:**
- Regenerated (gitignored, no commit): `data/processed/atp_features.csv`, `data/processed/wta_features.csv`

- [x] **Step 1: Regenerar y correr el backtest para ambos tours**

Run: `./tenis-env/Scripts/python.exe -m src.pipeline atp`
Run: `./tenis-env/Scripts/python.exe -m src.pipeline wta`

- [x] **Step 2: Comparar contra el baseline (Task 1, reutilizado de 3b)**

**Resultados 2026-07-28:**

```
ATP: Accuracy 0.6692 -> 0.6694  (+0.02pp)
     Log-Loss 0.6019 -> 0.6017  (-0.0002, mejora marginal)

WTA: Accuracy 0.6445 -> 0.6450  (+0.05pp)
     Log-Loss 0.6261 -> 0.6262  (+0.0001, empeora marginal)
```

Resultado esencialmente neutro (deltas dentro del ruido), similar en
magnitud al de H2H (3a) — a diferencia de Fatiga (3b), que mejoró las 4
métricas de forma consistente.

- [x] **Step 3: Decisión**

Igual que en 3a/3b — el usuario decide si mantener, ajustar las constantes
(`SURFACE_TRANSITION_PENALTY_MAX`, la tabla `SURFACE_TRANSITION_SEVERITY`,
`SURFACE_TRANSITION_WINDOW_DAYS`) y volver a validar, o revertir. No
requiere commit de código — solo verificación y reporte.

Este es también el cierre de Módulo 3 completo (3 de 3 features) si se
decide mantener — vale la pena un resumen final de las 3 features juntas
(H2H neutro/mantenido, Fatiga mejora/mantenido, Transición de Superficie
según este resultado) en el reporte al usuario.

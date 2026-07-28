# Módulo 3, Sub-proyecto 3a: FEATURE_COLS + H2H Suavizado — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidar `FEATURE_COLS` en una única fuente de verdad, y mejorar `h2h_rate` de un win-rate plano a uno ponderado por recencia + suavizado por tamaño de muestra (`shrinkage`), validando el impacto real vía walk-forward backtest antes/después.

**Architecture:** Refactor mecánico (Task 2) seguido de un cambio de fórmula puro y aislado en `FeatureBuilder` (Task 3), verificado end-to-end vía los mismos puntos de integración que ya prueban `age_multiplier_diff`/`rust_factor_diff` (Tasks 4-5), y validado empíricamente con el backtest existente (Tasks 1 y 6).

**Tech Stack:** Python, numpy, pandas, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-07-28-h2h-smoothing-design.md` — leer primero para las fórmulas exactas y el razonamiento (por qué NO se pliega en `elo_prob`/`adjusted_elo`).

---

## Task 1: Capturar baseline (sin cambios de código)

**Files:** ninguno — solo verificación, guardar los números impresos para comparar en Task 6.

- [ ] **Step 1: Correr el pipeline para ambos tours y guardar el reporte**

Run: `./tenis-env/Scripts/python.exe -m src.pipeline atp`
Run: `./tenis-env/Scripts/python.exe -m src.pipeline wta`

Cada corrida tarda varios minutos (reconstruye Elo/features desde datos crudos
1990-2023, sin caché). Al final de cada una, copiar textualmente las dos
líneas finales:
```
  Mean Accuracy : X.XXXX  (Elo-only: X.XXXX)
  Mean Log-Loss : X.XXXX  (Elo-only: X.XXXX)
```

- [ ] **Step 2: Guardar el baseline en el propio plan**

Pegar los 4 números (ATP accuracy/log-loss, WTA accuracy/log-loss) como
comentario en este archivo de plan, junto a este task, para referencia en
Task 6. No requiere commit — es solo registro de referencia.

---

## Task 2: Consolidar `FEATURE_COLS` en `src/features/__init__.py`

**Files:**
- Create: `src/features/__init__.py` (existe pero está vacío)
- Modify: `src/value_analysis.py:54-58`
- Modify: `src/backtest/walkforward.py:13-25`
- Test: correr la suite completa (sin tests nuevos — es un refactor puro)

- [ ] **Step 1: Escribir la constante centralizada**

`src/features/__init__.py` (reemplaza el archivo vacío):

```python
"""Shared feature-column schema — single source of truth for the exact list
and order of columns fed into the LogisticRegression, used identically at
training time (src/backtest/walkforward.py) and prediction time
(src/value_analysis.py). Order matters: it's positional in the model's
input vector (np.array([[feats[c] for c in FEATURE_COLS]])).
"""
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff",
]
```

- [ ] **Step 2: Actualizar `src/value_analysis.py`**

Reemplazar (línea 53-59):
```python
# ── model constants ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff",
]
```
por:
```python
# ── model constants ──────────────────────────────────────────────────────────
from src.features import FEATURE_COLS
```
(mover este import junto a los demás imports de `src.*` al inicio del archivo,
en vez de dejarlo a mitad del archivo donde estaba la lista literal — seguir
el orden de imports ya establecido arriba en el archivo).

- [ ] **Step 3: Actualizar `src/backtest/walkforward.py`**

Reemplazar (líneas 13-25):
```python
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
```
por:
```python
from src.features import FEATURE_COLS as _FEATURE_COLS
```
(agregar a los imports existentes en la parte superior del archivo, junto a
`from src.features.decay import ...` / `from src.features.engineering import ...`;
el nombre local `_FEATURE_COLS` se mantiene sin cambios ya que se usa así en
el resto del archivo).

- [ ] **Step 4: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: mismo número de tests pasando que antes de este cambio (refactor
puro, no debe romper ni cambiar el comportamiento de nada).

- [ ] **Step 5: Commit**

```bash
git add src/features/__init__.py src/value_analysis.py src/backtest/walkforward.py
git commit -m "refactor: consolidate FEATURE_COLS into a single shared constant"
```

---

## Task 3: `_weighted_h2h_rate` — ponderación por recencia + shrinkage

**Files:**
- Modify: `src/features/engineering.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Escribir los tests que fallan**

Reemplazar el contenido de `tests/test_features.py` completo (agrega imports
nuevos y una nueva sección de tests para `_weighted_h2h_rate`, además de
reemplazar `test_h2h_tracks_wins_per_direction` que hardcodea la fórmula
vieja):

```python
import pytest
from datetime import date
from src.features.engineering import FeatureBuilder, _weighted_h2h_rate


def test_new_player_returns_defaults():
    fb = FeatureBuilder()
    f = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    assert f["recent_win_rate"] == 0.5
    assert f["h2h_win_rate"] == 0.5
    assert f["h2h_matches"] == 0
    assert f["rest_days"] == 14  # default when no history


def test_recent_win_rate_after_all_wins():
    fb = FeatureBuilder()
    for _ in range(10):
        fb.update("A", "B", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 1.0


def test_recent_win_rate_after_all_losses():
    fb = FeatureBuilder()
    for _ in range(10):
        fb.update("B", "A", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 0.0


def test_recent_win_rate_uses_only_last_n():
    fb = FeatureBuilder(recent_n=5)
    # 10 losses then 5 wins — should see only the 5 wins
    for _ in range(10):
        fb.update("B", "A", "hard", date(2023, 1, 1))
    for _ in range(5):
        fb.update("A", "B", "hard", date(2023, 2, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 1.0


def test_h2h_matches_counts_both_directions():
    """h2h_matches (total count) is unaffected by the weighted/shrunk rate
    formula change — still a plain count of meetings either direction."""
    fb = FeatureBuilder()
    fb.update("A", "B", "hard", date(2023, 1, 1))
    fb.update("A", "B", "hard", date(2023, 2, 1))
    fb.update("B", "A", "hard", date(2023, 3, 1))

    fa = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    fb_ = fb.get_features("B", "A", "hard", date(2023, 6, 1))

    assert fa["h2h_matches"] == 3
    assert fb_["h2h_matches"] == 3


def test_rest_days_calculated_correctly():
    fb = FeatureBuilder()
    fb.update("A", "B", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 1, 8))
    assert f["rest_days"] == 7


def test_surface_form_is_surface_specific():
    fb = FeatureBuilder(surface_n=5)
    fb.update("A", "B", "clay", date(2023, 1, 1))
    fb.update("A", "B", "clay", date(2023, 1, 2))
    fb.update("B", "A", "hard", date(2023, 1, 3))
    fb.update("B", "A", "hard", date(2023, 1, 4))

    f_clay = fb.get_features("A", "B", "clay", date(2023, 6, 1))
    f_hard = fb.get_features("A", "B", "hard", date(2023, 6, 1))

    assert f_clay["recent_win_rate_surface"] == 1.0
    assert f_hard["recent_win_rate_surface"] == 0.0


def test_update_does_not_affect_features_for_current_match():
    """Features must use only pre-match information."""
    fb = FeatureBuilder()
    f_before = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    fb.update("A", "B", "hard", date(2023, 1, 1))
    f_after = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    # h2h before update should be 0, after should be 1
    assert f_before["h2h_matches"] == 0
    assert f_after["h2h_matches"] == 1


# ── _weighted_h2h_rate ────────────────────────────────────────────────────────
# Hand-computed expected values — see docs/superpowers/specs/2026-07-28-h2h-smoothing-design.md
# for the formula: shrink = n/(n+4), rate = shrink*weighted_rate + (1-shrink)*0.5,
# weighted_rate = weighted_average(wins, weights=linspace(0.5, 1.0, n)).

class TestWeightedH2hRate:
    def test_no_matches_returns_neutral(self):
        assert _weighted_h2h_rate([]) == 0.5

    def test_single_win_shrinks_toward_neutral(self):
        # n=1: weighted_rate=1.0 (single point, weight cancels), shrink=1/5=0.2
        # rate = 0.2*1.0 + 0.8*0.5 = 0.6 (not 1.0 — one win isn't strong evidence)
        matches = [(date(2023, 1, 1), True)]
        assert _weighted_h2h_rate(matches) == pytest.approx(0.6)

    def test_single_loss_shrinks_toward_neutral(self):
        matches = [(date(2023, 1, 1), False)]
        assert _weighted_h2h_rate(matches) == pytest.approx(0.4)

    def test_recency_weighting_favors_more_recent_result(self):
        # Same 1-1 record, opposite chronological order -> different rate.
        # n=2, weights=[0.5,1.0]: recent win -> weighted_rate=1.0/1.5=2/3,
        # shrink=2/6=1/3, rate=1/3*2/3 + 2/3*0.5 = 2/9+1/3 = 5/9.
        lost_old_won_recent = [(date(2023, 1, 1), False), (date(2023, 2, 1), True)]
        won_old_lost_recent = [(date(2023, 1, 1), True), (date(2023, 2, 1), False)]

        recent_win_rate = _weighted_h2h_rate(lost_old_won_recent)
        recent_loss_rate = _weighted_h2h_rate(won_old_lost_recent)

        assert recent_win_rate == pytest.approx(5 / 9)
        assert recent_loss_rate == pytest.approx(4 / 9)
        assert recent_win_rate > recent_loss_rate

    def test_opponent_perspective_is_exact_complement(self):
        # h2h_rate(B,A) must equal 1 - h2h_rate(A,B) for any history — the
        # mirror-row rule (h2h_rate -> 1-h2h_rate) depends on this holding
        # exactly, not approximately.
        player_view = [(date(2023, 1, 1), True), (date(2023, 2, 1), False), (date(2023, 3, 1), True)]
        opponent_view = [(date(2023, 1, 1), False), (date(2023, 2, 1), True), (date(2023, 3, 1), False)]

        assert _weighted_h2h_rate(opponent_view) == pytest.approx(
            1 - _weighted_h2h_rate(player_view)
        )

    def test_larger_sample_still_shrinks_but_less(self):
        # n=10, all wins: weighted_rate=1.0 (uniform result regardless of
        # weights), shrink=10/14=5/7, rate = 5/7*1.0 + 2/7*0.5 = 6/7.
        # Still not 1.0 even with 10 straight wins, but much closer than n=1's 0.6.
        matches = [(date(2023, 1, i + 1), True) for i in range(10)]
        assert _weighted_h2h_rate(matches) == pytest.approx(6 / 7)
```

- [ ] **Step 2: Correr los tests nuevos para verificar que fallan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: FAIL — `ImportError: cannot import name '_weighted_h2h_rate'`

- [ ] **Step 3: Implementar en `src/features/engineering.py`**

Reemplazar el archivo completo:

```python
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

H2H_SHRINKAGE_K = 4


def _weighted_h2h_rate(matches: List[Tuple[date, bool]]) -> float:
    """Recency-weighted, sample-size-shrunk head-to-head win rate.

    matches: chronological (oldest first) list of (date, player_won) for
    this specific player-vs-opponent pairing.

    Recency weighting reuses the same linspace(0.5, 1.0, n) shape as
    EloHistoryTracker.rolling_elo, for consistency across the codebase.
    Shrinkage toward 0.5 (K=4) keeps a single meeting from swinging the
    rate as hard as a real sample would — see design doc for the exact
    algebraic proof that this still satisfies rate(B,A) == 1 - rate(A,B).
    """
    n = len(matches)
    if n == 0:
        return 0.5
    weights = np.linspace(0.5, 1.0, n)
    wins = np.array([1.0 if won else 0.0 for _, won in matches])
    weighted_rate = float(np.average(wins, weights=weights))
    shrink = n / (n + H2H_SHRINKAGE_K)
    return shrink * weighted_rate + (1 - shrink) * 0.5


class FeatureBuilder:
    def __init__(self, recent_n: int = 10, surface_n: int = 10):
        self.recent_n = recent_n
        self.surface_n = surface_n
        # player -> [(match_date, surface, won)]
        self._history: Dict[str, List[Tuple[date, str, bool]]] = defaultdict(list)
        # (player, opponent) -> [(match_date, player_won)], both directions stored
        self._h2h_matches: Dict[Tuple[str, str], List[Tuple[date, bool]]] = defaultdict(list)
        self._last_match_date: Dict[str, Optional[date]] = {}

    def _win_rate(self, results: List[bool]) -> float:
        if not results:
            return 0.5
        return sum(results) / len(results)

    def get_features(self, player: str, opponent: str, surface: str, match_date: date) -> dict:
        history = self._history[player]

        recent = [won for _, _, won in history[-self.recent_n :]]
        recent_surface = [won for _, s, won in history if s == surface][-self.surface_n :]

        h2h_matches = self._h2h_matches[(player, opponent)]
        h2h_rate = _weighted_h2h_rate(h2h_matches)

        last = self._last_match_date.get(player)
        rest_days = (match_date - last).days if last is not None else 14

        return {
            "recent_win_rate": self._win_rate(recent),
            "recent_win_rate_surface": self._win_rate(recent_surface) if recent_surface else 0.5,
            "h2h_win_rate": h2h_rate,
            "h2h_matches": len(h2h_matches),
            "rest_days": float(rest_days),
        }

    def match_dates(self, player: str) -> List[date]:
        """This player's own past match dates (any surface), oldest first."""
        return [d for d, _, _ in self._history[player]]

    def update(self, winner: str, loser: str, surface: str, match_date: date) -> None:
        self._history[winner].append((match_date, surface, True))
        self._history[loser].append((match_date, surface, False))
        self._h2h_matches[(winner, loser)].append((match_date, True))
        self._h2h_matches[(loser, winner)].append((match_date, False))
        self._last_match_date[winner] = match_date
        self._last_match_date[loser] = match_date
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: PASS (todos, incluyendo la nueva clase `TestWeightedH2hRate`)

- [ ] **Step 5: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: revisar cuidadosamente cualquier fallo en `tests/test_build_match_features.py`
o `tests/test_value_analysis.py` — es esperable que NINGUNO falle todavía en
este punto, ya que `h2h_rate` no está en `_MIRROR_FLIP_COLS` (se maneja aparte)
y ningún test existente hardcodea el valor exacto de h2h_rate salvo el que ya
se reemplazó arriba. Si algo falla inesperadamente, investigar antes de continuar
(no asumir que es "solo la fórmula nueva", podría ser un uso no documentado
del viejo `_h2h_wins`).

- [ ] **Step 6: Commit**

```bash
git add src/features/engineering.py tests/test_features.py
git commit -m "feat: weight h2h_rate by recency and shrink toward 0.5 for small samples"
```

---

## Task 4: Test de simetría de fila espejo en `build_match_features`

**Files:**
- Modify: `tests/test_build_match_features.py`

- [ ] **Step 1: Escribir el test que falla (probablemente ya pasa — es una verificación, no una feature nueva)**

Agregar a `tests/test_build_match_features.py`:

```python
def test_mirror_row_negates_h2h_rate_as_one_minus_rate():
    """h2h_rate isn't in _MIRROR_FLIP_COLS (it's a bounded [0,1] rate, not a
    signed diff) — the mirror row must instead show 1 - original, and this
    must still hold exactly under the new weighted+shrunk formula."""
    df = pd.DataFrame(
        [
            _row("A", "B", "clay", date(2023, 1, 1)),
            _row("B", "A", "clay", date(2023, 2, 1)),
            _row("A", "B", "clay", date(2023, 3, 1)),
        ]
    )
    match_df = build_match_features(df, EloSystem(), FeatureBuilder())
    original = match_df[~match_df["is_mirror"]]
    mirror = match_df[match_df["is_mirror"]]

    for orig_rate, mirror_rate in zip(original["h2h_rate"], mirror["h2h_rate"]):
        assert mirror_rate == pytest.approx(1 - orig_rate)
```

Añadir `import pytest` al inicio del archivo si no está ya presente.

- [ ] **Step 2: Correr el test**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_build_match_features.py -v`
Expected: PASS. Si falla, es una señal real de que la propiedad algebraica
del diseño no se cumple en la implementación — no continuar sin entender por qué.

- [ ] **Step 3: Commit**

```bash
git add tests/test_build_match_features.py
git commit -m "test: verify mirror-row h2h_rate symmetry holds under the weighted formula"
```

---

## Task 5: Test de wiring posicional en `predict_match`

**Files:**
- Modify: `tests/test_value_analysis.py`

- [ ] **Step 1: Escribir el test que falla (verificación de wiring, no feature nueva)**

Agregar a `tests/test_value_analysis.py`, cerca de `TestBuildPredictionFeaturesDecay`
(reutiliza el patrón de stub-classifier ya establecido en
`TestPredictMatchAgeLookup`):

```python
class TestPredictMatchH2hWiring:
    def test_h2h_rate_reaches_lr_input_vector(self):
        """Proves h2h_rate is positionally wired into the LR input, the same
        way TestPredictMatchAgeLookup proves it for adjusted_elo_diff."""
        elo = _fake_elo({"A": 1500.0, "B": 1500.0})
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.9, "h2h_matches": 5, "rest_days": 14.0,
            },
            match_dates=lambda p: [],
        )

        class _StubClf:
            def predict_proba(self, X):
                import numpy as _np
                idx = FEATURE_COLS.index("h2h_rate")
                h2h = X[0][idx]
                return _np.array([[1 - h2h, h2h]])

        pred = predict_match(
            elo, fb, _StubClf(),
            "A", "B", "clay", date(2026, 7, 21),
            rank_lookup=None, age_lookup=None,
        )
        assert pred["p_a_raw"] == pytest.approx(0.9)
```

- [ ] **Step 2: Correr el test**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_value_analysis.py::TestPredictMatchH2hWiring -v`
Expected: PASS. Si falla, `build_prediction_features` no está mapeando
`h2h_win_rate` a `h2h_rate` correctamente, o `FEATURE_COLS` cambió de orden —
investigar antes de continuar.

- [ ] **Step 3: Correr la suite completa**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: todos los tests pasan.

- [ ] **Step 4: Commit**

```bash
git add tests/test_value_analysis.py
git commit -m "test: verify h2h_rate reaches the LR input vector positionally"
```

---

## Task 6: Regenerar datasets, correr backtest, comparar contra baseline

**Files:**
- Regenerated (gitignored, no commit): `data/processed/atp_features.csv`, `data/processed/wta_features.csv`

- [ ] **Step 1: Regenerar y correr el backtest para ambos tours**

Run: `./tenis-env/Scripts/python.exe -m src.pipeline atp`
Run: `./tenis-env/Scripts/python.exe -m src.pipeline wta`

- [ ] **Step 2: Comparar contra el baseline de Task 1**

Para cada tour, comparar `Mean Accuracy` y `Mean Log-Loss` contra los números
guardados en Task 1. Reportar la diferencia exacta (ej. "ATP accuracy
0.6423 → 0.6431, +0.08pp; log-loss 0.6011 → 0.5998, -0.0013").

- [ ] **Step 3: Decisión**

Si accuracy sube y/o log-loss baja en ambos tours (o al menos no empeora
significativamente en ninguno): la feature se considera validada, queda tal
cual. Si empeora en algún tour: reportar los números exactos al usuario antes
de decidir si ajustar `H2H_SHRINKAGE_K`, revertir, o mantenerla igual (el
usuario decide — no es una decisión puramente mecánica).

No requiere commit de código en este task — es un paso de verificación y
reporte. Si se decide ajustar `H2H_SHRINKAGE_K`, eso es una vuelta a Task 3
(nuevo ciclo TDD con el valor de K distinto), no parte de este task.

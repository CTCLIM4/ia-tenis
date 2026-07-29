# Módulo 3, Sub-proyecto 3c: Transición de Superficie — Diseño

**Fecha:** 2026-07-28
**Estado:** Aprobado, pendiente de implementación
**Contexto:** Tercero y último de los 3 features de Módulo 3 (H2H — neutro,
mantenido; Fatiga — mejora las 4 métricas, mantenido). Este es el más
liviano de los tres en cuanto a plomería de datos: no requiere tocar
`src/data/loader.py` en absoluto.

## Objetivo

Agregar una feature `surface_transition_multiplier_diff` que penalice
temporalmente a un jugador que acaba de cambiar de superficie respecto a su
partido anterior, con severidad distinta según qué tan diferentes son las
dos superficies (Arcilla↔Césped es la transición más brusca; Dura actúa
como terreno intermedio), integrada al mismo patrón que
`age_multiplier`/`rust_factor`/`fatigue_multiplier` ya usan.

## 1. Datos disponibles (verificado contra el código actual)

`FeatureBuilder._history[player]` (`src/features/engineering.py`) ya
almacena `(match_date, surface, won)` por cada partido, en orden
cronológico, para H2H/form/rest. **La última entrada de esa lista ya
contiene todo lo necesario** (fecha y superficie del partido anterior del
jugador) — a diferencia de Fatiga, este feature **no requiere ningún
almacenamiento nuevo** en `FeatureBuilder`, ni tocar `update()`, ni tocar
`src/data/loader.py`. Solo hace falta un accessor nuevo que exponga la
última entrada.

Distribución real de superficies verificada (`load_atp_matches`/
`load_wta_matches`, 2015-2024): ATP `hard=16339, clay=8352, grass=2932,
unknown=135, carpet=15`; WTA `hard=14264, clay=6340, grass=2732` (sin
carpet/unknown en esta ventana). `carpet`/`unknown` son casos raros —
reciben un valor de severidad por defecto, no una tabla dedicada.

## 2. Decisiones confirmadas

1. **Tabla de severidad**: Arcilla↔Césped = 1.0 (máxima diferencia de
   rebote/velocidad); Dura↔Arcilla = Dura↔Césped = 0.5 (Dura es terreno
   intermedio). Cualquier par no listado explícitamente (carpet/unknown) usa
   un valor por defecto de 0.5.
2. **Decaimiento lineal** dentro de la ventana de 10 días: penalización
   máxima el día 0 (transición muy reciente), decreciendo linealmente hasta
   0 en el día 10 — mismo estilo de normalización que `fatigue_multiplier`/
   `rust_factor` ya usan, no decaimiento exponencial.
3. **`SURFACE_TRANSITION_PENALTY_MAX = 0.10`** — valor inicial sin
   calibrar, mismo enfoque que `H2H_SHRINKAGE_K`/`FATIGUE_PENALTY_MAX`:
   implementar, validar con el backtest, ajustar si hace falta.

## 3. Modelo matemático

**Nuevo accessor** en `FeatureBuilder` (sin nuevo almacenamiento):
```python
def last_surface_and_date(self, player: str) -> Optional[Tuple[date, str]]:
    """This player's most recent match's (date, surface), or None if no history."""
    history = self._history[player]
    if not history:
        return None
    d, s, _ = history[-1]
    return (d, s)
```

**Función pura** en `src/features/decay.py`:
```python
SURFACE_TRANSITION_WINDOW_DAYS = 10
SURFACE_TRANSITION_PENALTY_MAX = 0.10
SURFACE_TRANSITION_SEVERITY = {
    frozenset({"clay", "grass"}): 1.0,
    frozenset({"clay", "hard"}): 0.5,
    frozenset({"grass", "hard"}): 0.5,
}
_DEFAULT_TRANSITION_SEVERITY = 0.5   # carpet/unknown pairs — rare, safe default

def surface_transition_multiplier(
    last_surface_and_date: Optional[Tuple[date, str]],
    current_surface: str,
    current_date: date,
) -> float:
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

`1.0` (neutro) cuando: sin historial, misma superficie que el partido
anterior, o `days_since >= 10` (transición ya "absorbida"). Nota:
`days_since < 0` (fecha del partido anterior posterior a la actual)
tampoco debería ocurrir dado el reemplazo cronológico estricto del
pipeline, pero el chequeo `0 <=` lo trata como neutro por seguridad, sin
lanzar excepción.

**Integración en `adjusted_elo_surface`** (`calculate_decay_features`), un
4to multiplicador:
```python
adjusted_elo_surface = historical_elo_surface * age_mult * rust * fatigue * surface_transition
```

## 4. Simetría A↔B

Igual de simple que fatiga — cada jugador se calcula de forma
independiente a partir de su propio último partido, sin información
cruzada. `surface_transition_multiplier_diff = mult_a - mult_b`, una
diferencia con signo plana, agregada a `_MIRROR_FLIP_COLS` (negación
simple, sin necesidad de prueba algebraica especial como `h2h_rate`).

## 5. `FEATURE_COLS` — columna #13

```python
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff", "surface_transition_multiplier_diff",
    "adjusted_elo_diff",
]
```

## 6. Archivos afectados (5 — el más liviano de los 3 sub-proyectos)

1. `src/features/engineering.py` — `last_surface_and_date()`.
2. `src/features/decay.py` — `surface_transition_multiplier()`, constantes,
   integración en `calculate_decay_features`.
3. `src/features/__init__.py` — agregar la columna a `FEATURE_COLS`.
4. `src/backtest/walkforward.py` — wiring + `_MIRROR_FLIP_COLS`.
5. `src/value_analysis.py` — wiring + `_LOG_FIELDS` (ya sabemos, por
   Fatiga, que esta es una tercera copia del schema que también necesita
   actualizarse — se incluye desde el plan esta vez, no como
   descubrimiento tardío).

No toca `src/data/loader.py` (a diferencia de Fatiga) — no hay nuevo dato
crudo que parsear, todo ya está en `_history`.

## 7. Baseline para el backtest final

Se reutilizan los números finales de 3b (mismo estado de `master` en el que
arranca este sub-proyecto, commit `eb0ba30`):
```
ATP: Accuracy 0.6692, Log-Loss 0.6019
WTA: Accuracy 0.6445, Log-Loss 0.6261
```

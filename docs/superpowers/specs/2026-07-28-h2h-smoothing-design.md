# Módulo 3, Sub-proyecto 3a: Refactorización FEATURE_COLS + H2H Suavizado — Diseño

**Fecha:** 2026-07-28
**Estado:** Aprobado, pendiente de implementación
**Contexto:** Primero de 3 features de Módulo 3 (H2H, fatiga, transición de superficie),
implementadas y validadas una a la vez vía backtest walk-forward antes de avanzar
a la siguiente.

## Objetivo

1. Eliminar la duplicación manual de `FEATURE_COLS` entre `src/value_analysis.py`
   y `src/backtest/walkforward.py` (dos copias independientes de la misma lista
   de 11 columnas, sin nada que las mantenga sincronizadas).
2. Mejorar `h2h_rate` (columna ya existente en `FEATURE_COLS`) de un win-rate
   plano sin ponderar a una versión ponderada por recencia + suavizada
   ("shrinkage") por tamaño de muestra chico.

## Decisión de diseño explícita: NO se pliega en `elo_prob`/`adjusted_elo`

El pedido original describía esto como un "ajuste a la probabilidad Elo". Se
decidió **no** implementarlo como un blend directo sobre `elo_prob` (como sí
ocurre con `age_multiplier`/`rust_factor` sobre `adjusted_elo_surface`), por
una razón técnica: esos multiplicadores representan fuerza-de-rating (mismas
unidades que Elo, neutro=1.0); H2H es información específica del enfrentamiento
(acotada [0,1], sin forma multiplicativa natural sin inventar una transformación
arbitraria). H2H se queda como el resto de las señales no-Elo del proyecto
(`rank_diff`, `form_diff`, `surface_form_diff`): una columna propia que el
`LogisticRegression` ya entrenado aprende a ponderar junto a `elo_prob`.

## Cambio 1: `FEATURE_COLS` — fuente única de verdad

Nuevo: `src/features/__init__.py` (hoy vacío) exporta:

```python
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff",
]
```

- `src/value_analysis.py:54-58` — reemplaza la definición local por
  `from src.features import FEATURE_COLS`.
- `src/backtest/walkforward.py:13-25` — reemplaza `_FEATURE_COLS = [...]` por
  `from src.features import FEATURE_COLS as _FEATURE_COLS` (mantiene el nombre
  local `_FEATURE_COLS` usado en el resto del archivo, para minimizar el diff).

Cambio puramente mecánico — no debe alterar ningún valor calculado. Se valida
corriendo el baseline del backtest walk-forward antes y después de este cambio
(deben coincidir exactamente).

`_MIRROR_FLIP_COLS` (`walkforward.py:27-30`) no cambia — `h2h_rate` ya está
excluida de esa lista (se maneja aparte como `1 - h2h_rate`, ver Cambio 2).

## Cambio 2: `h2h_rate` ponderado + suavizado

**Estado actual** (`src/features/engineering.py`):
```python
_h2h_wins: Dict[Tuple[str, str], int]   # (winner, loser) -> conteo
...
p_wins = self._h2h_wins[(player, opponent)]
o_wins = self._h2h_wins[(opponent, player)]
total_h2h = p_wins + o_wins
h2h_rate = p_wins / total_h2h if total_h2h > 0 else 0.5   # win-rate plano
```

**Nuevo estado**: `_h2h_wins` (conteo) se reemplaza por
`_h2h_matches: Dict[Tuple[str, str], List[Tuple[date, bool]]]` — por cada
partido, se guardan ambas direcciones: `_h2h_matches[(winner, loser)].append((date, True))`
y `_h2h_matches[(loser, winner)].append((date, False))`. Sin límite de tamaño
(a diferencia de `EloHistoryTracker`'s `deque(maxlen=15)`) — los enfrentamientos
H2H entre dos jugadores específicos son raros a lo largo de una carrera, no
hace falta cap.

**Nueva función pura y testeable** `_weighted_h2h_rate(matches: List[Tuple[date, bool]]) -> float`:

```python
H2H_SHRINKAGE_K = 4

def _weighted_h2h_rate(matches: List[Tuple[date, bool]]) -> float:
    n = len(matches)
    if n == 0:
        return 0.5
    weights = np.linspace(0.5, 1.0, n)   # mismo esquema que rolling_elo, por consistencia
    wins = np.array([1.0 if won else 0.0 for _, won in matches])
    weighted_rate = np.average(wins, weights=weights)
    shrink = n / (n + H2H_SHRINKAGE_K)
    return shrink * weighted_rate + (1 - shrink) * 0.5
```

`K=4` es un valor inicial razonable, no calibrado — se valida/ajusta empíricamente
vía el backtest (Cambio 3), no analíticamente.

**Propiedad verificada algebraicamente**: `_weighted_h2h_rate` de A vs B sigue
cumpliendo `h2h_rate(B,A) == 1 - h2h_rate(A,B)` exactamente (misma n, mismos
pesos posicionales, wins complementarios en cada partido) — la regla de mirroreo
existente `h2h_rate → 1 - h2h_rate` (`walkforward.py:51,157`) sigue siendo
correcta sin cambios.

`get_features()` cambia solo la línea que computa `h2h_rate` (dict key sigue
siendo `h2h_win_rate`, sin cambios en el nombre — solo el valor). `h2h_matches`
(conteo, ya no usado por el modelo, solo debug/display) se deriva de
`len(matches)` en vez de la suma de los dos contadores viejos.

**Tests existentes que se rompen intencionalmente**: `test_h2h_tracks_wins_per_direction`
(`tests/test_features.py:42-53`) hardcodea el resultado del win-rate plano viejo
(`2/3`, `1/3`) — se reemplaza con valores calculados a mano para la fórmula nueva.
El resto de los tests de H2H en ese archivo (`test_new_player_returns_defaults`,
`test_update_does_not_affect_features_for_current_match`) no dependen del valor
exacto del rate y siguen pasando sin cambios.

## Cambio 3: regenerar datasets + validar con backtest

Comando único por tour (regenera `data/processed/{tour}_features.csv` Y corre
`walk_forward_backtest` con reporte de accuracy/log-loss en la misma corrida —
no existe un `python -m src.walkforward` separado, es `src/pipeline.py` quien
orquesta ambos pasos):

```
python -m src.pipeline atp
python -m src.pipeline wta
```

Se corre **dos veces**: una vez después del Cambio 1 (refactor puro — debe dar
resultados idénticos al baseline actual, confirmando que no se alteró
comportamiento), y otra vez después del Cambio 2 (fórmula H2H nueva — aquí sí
se espera que cambien accuracy/log-loss, y es lo que se compara contra el
baseline para decidir si la feature mejora el modelo).

## Testing (TDD)

1. `_weighted_h2h_rate` — función pura, aislada: `n=0`→0.5; `n=1` (ganado) da
   un valor sustancialmente encogido hacia 0.5 (no 1.0); registro simétrico
   1-1 → exactamente 0.5 sin importar el shrink; caso asimétrico con recencia
   (ej. 2 derrotas antiguas + 1 victoria reciente) con valor esperado calculado
   a mano; muestra grande (n=10+) converge cerca del rate real ponderado.
2. `FeatureBuilder` wiring — reemplazar `test_h2h_tracks_wins_per_direction`
   con la fórmula nueva; agregar caso de simetría A↔B explícito.
3. `build_match_features` — fila espejo (`tests/test_build_match_features.py`)
   sigue negando correctamente `h2h_rate` vía `1 - h2h_rate`.
4. `predict_match`/`FEATURE_COLS` — test de wiring posicional con el patrón de
   stub-classifier ya establecido en `tests/test_value_analysis.py`
   (`TestBuildPredictionFeaturesDecay`), confirmando que `h2h_rate` llega
   correctamente al vector de entrada del LR.
5. Consolidación `FEATURE_COLS`: correr la suite completa tras el Cambio 1,
   sin tests nuevos (es un refactor, no una feature).

# Módulo 3, Sub-proyecto 3b: Feature de Carga de Partidos / Fatiga — Diseño

**Fecha:** 2026-07-28
**Estado:** Aprobado, pendiente de implementación
**Contexto:** Segundo de 3 features de Módulo 3 (H2H ya mergeado, resultado
neutro en backtest pero mantenido por robustez). Fatiga es genuinamente
nueva — a diferencia de H2H, no existe hoy ningún tracking de sets jugados
en el pipeline.

## Objetivo

Agregar una feature `fatigue_multiplier_diff` que penalice a un jugador que
viene de una carga de partidos/sets alta en los últimos 7-14 días, integrada
al mismo patrón que `age_multiplier`/`rust_factor` ya usan: columna propia en
`FEATURE_COLS` **y** multiplicador dentro de `adjusted_elo_surface`.

## 1. Datos disponibles (verificado contra los archivos crudos reales)

- **ATP** (`data/raw/tennis_atp_tml/*.csv`): columna `score` (string, ej.
  `"7-6(5) 6-4"`). Sin columna de "sets jugados" directa — hay que parsear.
  Casos reales encontrados: `"6-3 2-4 RET"` (retiro a mitad de set),
  `"W/O"` (walkover, cero sets jugados).
- **WTA** (`data/raw/tennis_wta_tduk/*.xlsx`): columnas `Wsets`/`Lsets`
  directas — sets jugados por cualquiera de los dos jugadores = `Wsets + Lsets`
  (ambos están en cancha en cada set del partido).
- `src/data/loader.py`: `_clean`/`_clean_wta` hoy pasan `score`/`Wsets`/`Lsets`
  sin tocar — no existe una columna normalizada `sets_played` compartida
  entre tours. Se agrega en este sub-proyecto (mismo lugar donde ya se
  normalizan `surface`/`match_date`/ranks).
- `FeatureBuilder._history` (`src/features/engineering.py`) ya trackea
  `(date, surface, won)` por partido — sin sets. `rust_factor`
  (`src/features/decay.py`) ya cuenta *cantidad de partidos* en ventana de
  60 días, pero para un propósito distinto (inactividad de largo plazo, no
  sobrecarga de corto plazo) — señal complementaria, no duplicada.

## 2. Decisiones confirmadas

1. **Sets con RET se cuentan como jugados.** Parsing simple: cualquier token
   con forma `\d+-\d+(\(\d+\))?` cuenta como un set (incluye el set
   incompleto de un retiro — "2-4 RET" sí sumó desgaste físico real,
   6 games jugados). `"RET"`/`"W/O"`/texto no numérico se ignora. `"W/O"`
   → 0 sets. Score faltante/no parseable → 0 sets (default conservador, no
   crashea).
2. **Constantes de calibración sin calibrar**, mismo enfoque que
   `H2H_SHRINKAGE_K=4` en 3a: valores iniciales razonables, implementar,
   validar con el backtest walk-forward, ajustar si hace falta — no
   investigación empírica previa de distribución real.
3. **`FATIGUE_PENALTY_MAX = 0.15`** (reducción máxima de Elo efectivo por
   fatiga) — más chico que `AGE_PENALTY_FLOOR` (hasta 40% por edad), porque
   fatiga de 2 semanas debería pesar menos que declive de años/década.

## 3. Modelo matemático

**Almacenamiento nuevo** (paralelo a `_history`, no lo toca — cero riesgo
para H2H/form/rest ya construidos): `FeatureBuilder._workload_history:
Dict[str, List[Tuple[date, int]]]` — `(match_date, sets_played)` por
jugador, poblado en `update()`.

**Carga por ventana**, misma normalización `min(1.0, n/target)` que
`rust_factor` ya usa:

```python
FATIGUE_WINDOW_SHORT_DAYS = 7
FATIGUE_WINDOW_LONG_DAYS  = 14
FATIGUE_MATCH_TARGET_7D   = 3
FATIGUE_SET_TARGET_7D     = 6
FATIGUE_MATCH_TARGET_14D  = 5
FATIGUE_SET_TARGET_14D    = 10
FATIGUE_RECENT_WEIGHT     = 0.6   # peso de la ventana de 7d sobre la de 14d
FATIGUE_PENALTY_MAX       = 0.15

def _window_load(matches: int, sets: int, match_target: int, set_target: int) -> float:
    match_load = min(1.0, matches / match_target)
    set_load   = min(1.0, sets / set_target)
    return 0.5 * match_load + 0.5 * set_load

def fatigue_multiplier(workload_history: List[Tuple[date, int]], current_date: date) -> float:
    matches_7d, sets_7d   = _count_window(workload_history, current_date, FATIGUE_WINDOW_SHORT_DAYS)
    matches_14d, sets_14d = _count_window(workload_history, current_date, FATIGUE_WINDOW_LONG_DAYS)

    load_7d  = _window_load(matches_7d,  sets_7d,  FATIGUE_MATCH_TARGET_7D,  FATIGUE_SET_TARGET_7D)
    load_14d = _window_load(matches_14d, sets_14d, FATIGUE_MATCH_TARGET_14D, FATIGUE_SET_TARGET_14D)

    fatigue_load = FATIGUE_RECENT_WEIGHT * load_7d + (1 - FATIGUE_RECENT_WEIGHT) * load_14d
    return 1.0 - FATIGUE_PENALTY_MAX * fatigue_load
```

`_count_window` usa la misma convención de límites que `rust_factor` ya
establece (`cutoff <= d < current_date`, nunca incluye el partido actual —
no-lookahead). Un jugador sin historial (`workload_history == []`) da
`matches=sets=0` en ambas ventanas → `fatigue_load=0` → `fatigue_multiplier=1.0`
(neutro), igual que el resto de las features de decay para jugadores nuevos.

**"¿Tiene decaimiento temporal?"** Sí, pero como blend de dos ventanas
discretas (7d pesa 60%, 14d pesa 40%) en vez de decaimiento exponencial
continuo — mismo estilo que el resto del archivo (`rust_factor`'s corte
duro a 60 días, `rolling_elo`'s `linspace` por posición, no por fecha).
Consistente con la filosofía del proyecto: constantes nombradas y simples,
validadas empíricamente, no fórmulas de decaimiento continuo sin motivo
claro para la complejidad extra.

**Integración con `adjusted_elo_surface`** (`calculate_decay_features`,
`src/features/decay.py`), agregando un factor más a la fórmula existente:
```python
adjusted_elo_surface = historical_elo_surface * age_mult * rust * fatigue_mult
```

## 4. Simetría A↔B

Mucho más simple que H2H — `fatigue_multiplier` se calcula de forma
independiente por jugador, a partir únicamente de su propio historial (sin
información cruzada entre jugadores, a diferencia de H2H). La columna final
es `fatigue_multiplier_diff = fatigue_a - fatigue_b`, una diferencia con
signo plana. Se agrega a `_MIRROR_FLIP_COLS` (negación simple:
`diff_BA = fatiga_b - fatiga_a = -(fatiga_a - fatiga_b) = -diff_AB`, cierto
algebraicamente para cualquier resta A-B sin necesidad de una prueba
especial como la de `h2h_rate` acotado [0,1]).

## 5. `FEATURE_COLS` — un nuevo elemento (a diferencia de H2H, que no agregó ninguno)

```python
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff",   # NUEVO
    "adjusted_elo_diff",
]
```
Gracias a la consolidación de 3a, este es un cambio en **un solo lugar**
(`src/features/__init__.py`) en vez de dos copias a mantener sincronizadas.

## 6. Archivos afectados (6, más que los 3 de H2H — refleja el alcance
"completo" elegido: partidos + sets reales, no solo partidos)

1. `src/data/loader.py` — normalizar `sets_played` (parser ATP, suma WTA).
2. `src/features/engineering.py` — `_workload_history`, `update()` gana
   parámetro `sets_played`, nuevo accessor `workload_history(player)`.
3. `src/features/decay.py` — `fatigue_multiplier()`, constantes, integración
   en `calculate_decay_features`.
4. `src/backtest/walkforward.py` — pasar `sets_played` a `update()`,
   agregar `fatigue_multiplier_diff` a la fila y a `_MIRROR_FLIP_COLS`.
5. `src/features/__init__.py` — agregar la columna a `FEATURE_COLS`.
6. `src/value_analysis.py` — `build_prediction_features` calcula
   `fatigue_a`/`fatigue_b` vía `fb.workload_history(player)` +
   `calculate_decay_features`, igual que ya hace con `age`/`rust`.

## 7. Baseline para el backtest final

Se reutilizan los números finales de 3a (capturados en el mismo estado de
`master` en el que arranca este sub-proyecto, commit `207b34f`) — no hace
falta correr el pipeline de nuevo solo para tener un "antes":
```
ATP: Accuracy 0.6688, Log-Loss 0.6024
WTA: Accuracy 0.6443, Log-Loss 0.6264
```

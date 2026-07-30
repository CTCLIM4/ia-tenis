# Módulo 4: Simulación Monte Carlo de Banca (Kelly) — Diseño

**Fecha:** 2026-07-29
**Módulo:** `src/bankroll_simulation.py`
**Estado:** Aprobado, pendiente de implementación

## Objetivo

El Módulo 2 (`backtest_analytics.py`) mide, de forma retrospectiva, cómo le
fue a la banca con las apuestas ya resueltas. El Módulo 4 responde una
pregunta distinta y prospectiva: **si seguimos apostando con el perfil de
edge/odds observado hasta ahora, ¿qué tan probable es que la banca crezca o
caiga X% en las próximas N apuestas, y qué fracción de Kelly (completo,
1/2, 1/4) minimiza el riesgo de ruina sin sacrificar demasiado el
crecimiento esperado?**

Para eso corre una simulación Monte Carlo: miles de secuencias sintéticas
de apuestas, generadas a partir de un perfil estadístico simple (edge y
odds promedio ± dispersión) calibrado con las apuestas reales ya resueltas
en `data/value_bets_log.csv`, comparando varias fracciones de Kelly en la
misma corrida.

## Arquitectura

Un solo módulo autocontenido (`src/bankroll_simulation.py`), siguiendo el
mismo patrón que `backtest_analytics.py` y `daily_scanner.py`. Reutiliza
`backtest_analytics.load_resolved_bets()` (no duplica el filtrado ni la
derivación de `bet_side`/`odds_taken`) y `value_analysis.KELLY_CAP` (mismo
tope de 5% de banca por apuesta que usa el resto del proyecto). Dependencia
nueva: ninguna — `numpy` ya es dependencia dura (`requirements.txt`).

Funciones puras y testeables, sin I/O salvo en la carga del CSV (heredada
de `load_resolved_bets`):

```
estimate_bet_profile(df: pd.DataFrame) -> dict
sample_bet(rng: np.random.Generator, profile: dict) -> tuple[float, float]
kelly_stake(p_win: float, odds: float, kelly_multiplier: float) -> float
simulate_path(rng, profile: dict, kelly_multiplier: float, n_bets: int, initial_bankroll: float) -> np.ndarray
run_monte_carlo(profile: dict, kelly_multipliers: list[float], n_simulations: int, n_bets: int, initial_bankroll: float, ruin_threshold: float, seed: Optional[int]) -> dict
print_report(mc_result: dict) -> None
run_simulation(bets_path, tour, bankroll, n_bets, n_simulations, kelly_multipliers, ruin_threshold, seed, mean_edge, std_edge, mean_odds, std_odds) -> None   # orquesta + CLI entry point
main() -> None
```

## Modelo de datos: del historial real al perfil de apuesta simulada

### `estimate_bet_profile(df)`

Recibe el DataFrame que devuelve `load_resolved_bets` (ya trae `bet_side`,
`odds_taken`, y las columnas crudas `edge_a`/`edge_b`). Deriva:

- `edge_taken = df["edge_a"].where(df["bet_side"] == "a", df["edge_b"])`
- `mean_edge = edge_taken.mean()`, `std_edge = edge_taken.std(ddof=1)` si
  `len(df) > 1`, si no `0.0` (con una sola fila no hay dispersión que
  estimar — se simula sin variar el edge apuesta a apuesta).
- `mean_odds = df["odds_taken"].mean()`, `std_odds = df["odds_taken"].std(ddof=1)`
  con el mismo fallback a `0.0` para `len(df) <= 1`.
- `n = len(df)` — se reporta y se usa para la advertencia de muestra
  pequeña (mismo umbral que Módulo 2, `SMALL_SAMPLE_THRESHOLD = 30`,
  importado de `backtest_analytics`).

Si `df` está vacío, `estimate_bet_profile` lanza `ValueError` — la
orquestación (`run_simulation`) es responsable de exigir overrides
manuales antes de llegar a este punto (ver sección CLI).

Perfil resultante: `{"mean_edge", "std_edge", "mean_odds", "std_odds", "n"}`.
`n` puede ser un entero (estimado del historial) o el string `"manual"`
cuando el usuario provee las 4 cifras a mano vía CLI — en ese caso no hay
`SMALL_SAMPLE_THRESHOLD` que aplicar y el reporte lo indica explícitamente
en vez de imprimir un número de N confuso.

### `sample_bet(rng, profile)`

Genera una apuesta sintética:

1. `edge = rng.normal(profile["mean_edge"], profile["std_edge"])`, luego
   `edge = max(edge, 0.001)` — un edge no positivo no generaría una apuesta
   real (el histórico solo contiene apuestas con edge > 0 by construcción
   de `calculate_value`), así que se recorta a un piso pequeño en vez de
   permitir simular "no apostar" (eso complicaría la comparación entre
   fracciones de Kelly sin aportar señal nueva).
2. `odds`: se muestrea de una lognormal ajustada por método de momentos
   para que su media/desviación coincidan con `mean_odds`/`std_odds`:
   - Si `std_odds == 0`: `odds = mean_odds` (determinístico).
   - Si no: `sigma2 = ln(1 + (std_odds/mean_odds)^2)`,
     `mu = ln(mean_odds) - sigma2/2`,
     `odds = rng.lognormal(mu, sqrt(sigma2))`.
   - Recorte final: `odds = max(odds, 1.01)` (cuota mínima operable).
3. `implied_prob = 1 / odds`; `p_win = clip(implied_prob + edge, 0.01, 0.99)`
   — recortado lejos de 0/1 para que Kelly no degenere.

Devuelve `(p_win, odds)`.

### `kelly_stake(p_win, odds, kelly_multiplier)`

Misma fórmula que `value_analysis.calculate_value` (no se reinventa el
criterio de Kelly del proyecto, se reutiliza su convención exacta):

```
implied_prob = 1 / odds
edge = p_win - implied_prob
raw_kelly = edge / (odds - 1) if edge > 0 else 0.0
stake = min(kelly_multiplier * raw_kelly, KELLY_CAP)   # KELLY_CAP = 0.05
```

El multiplicador (1.0 / 0.5 / 0.25) se aplica **antes** del cap — replica
la pregunta real ("si aplico medio Kelly, ¿mi stake real baja de 5% a
2.5% cuando el edge es grande, o el cap ya lo estaba limitando de todos
modos?").

### `simulate_path(rng, profile, kelly_multiplier, n_bets, initial_bankroll)`

A diferencia del Módulo 2 (que usa una curva de banca *lineal*,
`bankroll + cumsum(profit)*bankroll`, apropiada para reportar el pasado
tal cual ocurrió), esta simulación **compone** — cada apuesta arriesga una
fracción de la banca *actual*, no de la banca inicial fija. Esto es
deliberado: componer es el supuesto central del criterio de Kelly (crecimiento
geométrico), y es la razón por la que half-Kelly reduce tanto el riesgo de
ruina en la práctica. Se documenta explícitamente para que no se confunda
con una regresión respecto a la convención de Módulo 2.

```python
bankroll = initial_bankroll
path = [bankroll]
for _ in range(n_bets):
    p_win, odds = sample_bet(rng, profile)
    stake_frac = kelly_stake(p_win, odds, kelly_multiplier)
    stake_usd = stake_frac * bankroll
    if rng.random() < p_win:
        bankroll += stake_usd * (odds - 1)
    else:
        bankroll -= stake_usd
    bankroll = max(bankroll, 0.0)
    path.append(bankroll)
return np.array(path)
```

`stake_frac == 0.0` (edge recortado a 0.001 con odds muy altas puede dar un
`raw_kelly` minúsculo pero nunca exactamente 0 salvo por el piso — no hace
falta un caso especial de "no apostar").

### `run_monte_carlo(profile, kelly_multipliers, n_simulations, n_bets, initial_bankroll, ruin_threshold, seed)`

```python
rng = np.random.default_rng(seed)
results = {}
for km in kelly_multipliers:
    finals, max_drawdowns_pct, ruined = [], [], 0
    for _ in range(n_simulations):
        path = simulate_path(rng, profile, km, n_bets, initial_bankroll)
        finals.append(path[-1])
        running_max = np.maximum.accumulate(path)
        drawdown_pct = np.where(running_max > 0, (path - running_max) / running_max, 0.0)
        max_drawdowns_pct.append(drawdown_pct.min())
        if path.min() < ruin_threshold * initial_bankroll:
            ruined += 1
    finals = np.array(finals)
    results[km] = {
        "p10": np.percentile(finals, 10),
        "p50": np.percentile(finals, 50),
        "p90": np.percentile(finals, 90),
        "ruin_probability": ruined / n_simulations,
        "median_max_drawdown_pct": np.median(max_drawdowns_pct) * 100,
    }
return {
    "profile": profile, "results": results,
    "n_simulations": n_simulations, "n_bets": n_bets,
    "initial_bankroll": initial_bankroll, "ruin_threshold": ruin_threshold,
}
```

Un único `rng` (no uno por fracción de Kelly ni por simulación) para que
cada fracción de Kelly comparta la misma secuencia de partidos sintéticos
en la medida de lo posible dado el orden de iteración — no es un
requisito estricto de varianza-reducida (no se implementa common random
numbers de forma estricta por fracción, ver "No haremos" más abajo), solo
evita crear N objetos `Generator` sin necesidad.

## Reporte de consola (mockup)

```
============================================================
 SIMULACION MONTE CARLO DE BANCA - Modulo 4
 Banca inicial: $1,000.00 | 10,000 simulaciones x 50 apuestas
 Perfil: edge medio 5.8% (sd 4.1%) | odds medias 2.03 (sd 0.60) | N=7
============================================================

  *** Perfil estimado de muestra chica (N=7 < 30) - usar con cautela ***

                          Kelly completo    1/2 Kelly     1/4 Kelly
  Banca final P10               $612.34       $780.11       $890.02
  Banca final P50 (mediana)   $1,340.55     $1,190.20     $1,080.15
  Banca final P90             $2,850.90     $1,950.44     $1,420.33
  Prob. de ruina (<50% banca)     18.4%          6.2%          1.1%
  Drawdown maximo esperado       -34.2%        -19.8%        -10.5%
```

Si `profile["n"] == "manual"`, la línea de "Perfil:" omite `| N=...` y no
se imprime la advertencia de muestra chica.

## Manejo de errores (defensivo, nunca crashea)

- `bets_path` no existe y no se dieron los 4 overrides (`--mean-edge`,
  `--std-edge`, `--mean-odds`, `--std-odds`) completos: mensaje descriptivo
  pidiendo overrides manuales o que el archivo exista, `return`.
- `load_resolved_bets` devuelve 0 filas y no hay overrides completos:
  mismo mensaje que el caso anterior.
- Si los 4 overrides están presentes, se usa un perfil manual
  (`n = "manual"`) sin tocar el CSV en absoluto — permite simular
  supuestos hipotéticos sin depender de tener historial.
- Overrides parciales (1-3 de los 4): error explícito — "o los 4 o
  ninguno" — evita mezclar un mean_edge inventado con un std_odds
  estimado del historial de forma confusa.
- Validación de parámetros: `n_bets > 0`, `n_simulations > 0`,
  `0 < ruin_threshold < 1`, todos los `kelly_multipliers > 0` — si no,
  `ValueError` con mensaje claro (son errores de uso de CLI, no de datos).

## CLI

```
python -m src.bankroll_simulation
  --bankroll FLOAT            (default 1000)
  --tour {atp,wta,both}       (default both)
  --bets-file PATH             (default data/value_bets_log.csv)
  --bets INT                   (default 50)     # n_bets simulados por trayectoria
  --simulations INT            (default 10000)
  --kelly-fractions "1.0,0.5,0.25"  (default, lista separada por comas)
  --ruin-threshold FLOAT        (default 0.5)   # fracción de banca inicial
  --seed INT                    (default None -> no determinístico)
  --mean-edge FLOAT / --std-edge FLOAT / --mean-odds FLOAT / --std-odds FLOAT
      (opcionales, override manual; deben darse los 4 juntos o ninguno)
```

## Testing

`tests/test_bankroll_simulation.py`, siguiendo el patrón de
`test_backtest_analytics.py`:

- `estimate_bet_profile`: valores conocidos calculados a mano sobre un
  DataFrame sintético (reutilizando `load_resolved_bets` de
  `backtest_analytics` sobre un CSV de prueba), caso `n == 1` (std → 0.0),
  caso `df` vacío (`ValueError`).
- `sample_bet`: perfil con `std_edge == 0` y `std_odds == 0` → resultado
  exactamente determinístico (`p_win`/`odds` calculables a mano), sin
  necesidad de mockear `rng` gracias a la dispersión cero.
- `kelly_stake`: casos de valores fijos (incluye el caso del cap activo,
  edge negativo → stake 0.0, multiplicador 0.5).
- `simulate_path`: `rng = np.random.default_rng(42)` fijo, resultado
  reproducible byte a byte entre dos llamadas con la misma semilla;
  también un caso de perfil determinístico (std 0 en edge y odds) donde
  solo varía si gana o pierde cada apuesta, permitiendo verificar el
  balance a mano paso a paso.
- `run_monte_carlo`: `seed` fijo, `n_simulations` pequeño (100-1000) —
  smoke test de estructura de salida (claves presentes, un resultado por
  cada `kelly_multiplier`), sanity checks: banca nunca negativa en ningún
  percentil, `ruin_probability` monótona no-creciente al bajar el
  multiplicador de Kelly (con el mismo perfil y semilla).
- `print_report`: `capsys`, smoke test de que aparecen las secciones
  esperadas y los 3 multiplicadores en la tabla.
- `run_simulation`: casos de error (archivo inexistente sin overrides,
  overrides parciales, overrides completos sin archivo) y un caso normal
  (capsys, contiene "SIMULACION MONTE CARLO").

## Fuera de alcance (no haremos)

- Common random numbers estrictos por fracción de Kelly (cada fracción
  vería la *misma* secuencia exacta de partidos sintéticos) — añadiría
  complejidad de diseño (necesita clonar el estado del `rng` por
  fracción) sin cambiar las conclusiones prácticas del reporte.
- Gráficos/PNG de las trayectorias — decidido explícitamente en el
  brainstorming: solo reporte de consola, sin nueva dependencia
  (`matplotlib` ya está en `requirements.txt` para otros usos, pero no se
  usa aquí).
- Bootstrap del historial real como fuente de muestreo — decidido
  explícitamente: con N=7 el bootstrap subestimaría la variedad real de
  escenarios; se usa el modelo paramétrico calibrado.

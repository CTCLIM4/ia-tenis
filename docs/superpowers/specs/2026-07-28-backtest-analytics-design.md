# Módulo 2: Dashboard de Rendimiento y Análisis de Banca — Diseño

**Fecha:** 2026-07-28
**Módulo:** `src/backtest_analytics.py`
**Estado:** Aprobado, pendiente de implementación

## Objetivo

Leer el histórico acumulado de `data/value_bets_log.csv` (apuestas logueadas) y
`data/prediction_audit_log.csv` (población completa de predicciones evaluadas)
para generar un reporte de consola con métricas financieras, de riesgo, y una
comparación de calibración (EV teórico vs beneficio real).

## Arquitectura

Un solo módulo autocontenido (`src/backtest_analytics.py`), siguiendo el
patrón existente del proyecto (`daily_scanner.py`, `calibration_audit.py`,
`value_analysis.py`). Usa `pandas` — ya es dependencia dura, no se agrega
nada nuevo.

Funciones puras y testeables, sin I/O salvo en `load_*`:

```
load_resolved_bets(path: str, tour: Optional[str] = None) -> pd.DataFrame
load_audit_log(path: str) -> pd.DataFrame

compute_financial_metrics(df: pd.DataFrame, bankroll: float) -> dict
compute_risk_metrics(df: pd.DataFrame, bankroll: float) -> dict
compute_calibration_metrics(df: pd.DataFrame) -> dict
compute_audit_coverage(audit_df: pd.DataFrame) -> dict

print_report(financial, risk, calibration, coverage, bankroll, n) -> None
run_report(bets_path, audit_path, bankroll, tour) -> None   # orquesta + CLI entry point
main() -> None
```

## Modelo de datos: fila cruda → apuesta resuelta

`load_resolved_bets`:
1. Lee `value_bets_log.csv`.
2. Filtra a `status == 'ok'` y `result in {'A_win', 'B_win'}` — excluye filas
   de test (`test_never_played`), Elo faltante (`invalid_missing_elo`),
   pendientes y descartadas. Nunca deben contaminar las métricas de banca real
   (ver memoria "no sample contamination").
3. Filtro opcional `tour` (`'atp'` / `'wta'`; `None`/`'both'` = sin filtro).
4. Deriva por fila:
   - `bet_side` ('a'/'b'): el lado con `kelly_<side> > 0`. Determinístico —
     `calculate_value()` pone `kelly_fraction=0.0` cuando `edge <= 0`, y
     `edge_a + edge_b <= 0` siempre por el vig del bookmaker, así que como
     mucho un lado tiene kelly > 0 por fila.
   - `stake` = `kelly_<bet_side>` (fracción de banca)
   - `odds_taken` = `odds_<bet_side>`
   - `ev_theoretical` = `ev_<bet_side>` (ya es "por unidad apostada")
   - `win` = `profit > 0`

`load_audit_log` simplemente lee `prediction_audit_log.csv` sin transformar
(no tiene resultado/profit — no puede cruzarse con desenlaces reales todavía).

## Fórmulas

**Financieras** (banca base configurable vía `--bankroll`, default $1000):
- `ROI% = sum(profit) / sum(stake) * 100` (ratio adimensional, la banca se cancela)
- `Win Rate% = mean(win) * 100`
- `Beneficio Neto $ = sum(profit) * bankroll`
- `Stake Promedio $ = mean(stake) * bankroll`
- `Apuestas procesadas = len(df)`

**Riesgo** (orden cronológico por `match_date`):
- Curva de banca: `equity[i] = bankroll + cumsum(profit)[i] * bankroll`
- `Drawdown máximo $ = min(equity - running_max(equity))`
- `Drawdown máximo % = ese valor / running_max(equity) en ese punto * 100`
- `Varianza = var(profit, ddof=1)` — unidades de banca (fracción), no de ROI
  relativo, para ser consistente con Beneficio Neto y Drawdown
- Rachas: longitud máxima de `True`/`False` consecutivos en la serie `win`
  (orden cronológico)

**Calibración** — `ev_theoretical` es "retorno esperado por unidad apostada"
(adimensional); `profit` está escalado por Kelly (fracción de banca) y no es
directamente comparable. Se normaliza a retorno por unidad:
- `EV teórico promedio = mean(ev_theoretical)`
- `Retorno real promedio (por unidad) = mean(profit / stake)`
- `Diferencia = real - teórico`

**Cobertura del audit log** (de `prediction_audit_log.csv`, sin cruzar con
resultados):
- Total de predicciones evaluadas
- Conteo por `decision`: `logged`, `passed_low_edge`, `passed_user_declined`,
  `blocked_suspicious`, `invalid_missing_elo`
- Línea resumen: `logged` (apostadas) vs. resto (pasadas), como agregado
  simple de la fila anterior

## Layout de consola (mockup)

```
============================================================
 REPORTE DE RENDIMIENTO Y BANCA — Modulo 2
 Banca base: $1,000.00 | Periodo: 2026-07-02 a 2026-07-26 | Apuestas: 4
============================================================

  *** Muestra pequena (N=4 < 30) - metricas poco confiables todavia. ***

MÉTRICAS FINANCIERAS
  ROI acumulado:            -22.50%
  Win Rate:                   50.0%   (2W / 2L)
  Beneficio Neto:           -$92.50
  Stake promedio:            $42.50   (4.25% banca)
  Apuestas procesadas:            4

MÉTRICAS DE RIESGO
  Drawdown máximo:         -$100.00   (-9.5%)
  Varianza (profit):        0.00187
  Racha ganadora máxima:          1
  Racha perdedora máxima:         2

CALIBRACIÓN: EV TEÓRICO vs BENEFICIO REAL
  EV teórico promedio (por unidad):    +8.4%
  Retorno real promedio (por unidad): -18.2%
  Diferencia (real - teórico):        -26.6 pp

COBERTURA DEL AUDIT LOG (prediction_audit_log.csv)
  Predicciones evaluadas totales:   2
    logged:                  2 (100.0%)
    passed_low_edge:         0
    passed_user_declined:    0
    blocked_suspicious:      0
    invalid_missing_elo:     0
  Logged (apostadas) vs resto:  2 vs 0
```

## Manejo de errores (defensivo, nunca crashea)

- CSV no existe (`bets_path` o `audit_path`): `run_report` lo detecta,
  imprime un mensaje descriptivo por archivo y continúa con lo que sí
  esté disponible; si `value_bets_log.csv` falta o no tiene filas resueltas,
  no hay reporte financiero/riesgo/calibración que mostrar y se corta ahí
  (mensaje claro, sin traceback).
- Cero apuestas resueltas tras el filtro: mensaje
  "No hay apuestas resueltas todavía (status='ok' y result en A_win/B_win)."
  y `return` — evita división por cero en ROI (`sum(stake) == 0`).
- `N < 30` apuestas resueltas: advertencia explícita al inicio del reporte
  (no bloquea, solo avisa).
- `prediction_audit_log.csv` faltante: la sección de cobertura se omite con
  un aviso de una línea; el resto del reporte (financiero/riesgo/calibración)
  se muestra igual — son fuentes independientes.

## CLI

```
python -m src.backtest_analytics
  --bankroll FLOAT      (default 1000)
  --tour {atp,wta,both} (default both)
  --bets-file PATH       (default data/value_bets_log.csv)
  --audit-file PATH      (default data/prediction_audit_log.csv)
```

## Testing

`tests/test_backtest_analytics.py`, siguiendo el patrón de
`test_calibration_audit.py` (DataFrames/CSVs sintéticos vía `tmp_path`, sin
tocar los CSV reales del repo):
- `load_resolved_bets`: filtra status/result correctamente, filtro de tour,
  deriva `bet_side`/`stake`/`win` bien en casos ganados y perdidos.
- `compute_financial_metrics` / `compute_risk_metrics` /
  `compute_calibration_metrics`: valores conocidos calculados a mano sobre
  DataFrames pequeños construidos en el test (2-5 filas), incluyendo racha y
  drawdown con un caso no trivial (ganar-ganar-perder-perder).
- `compute_audit_coverage`: conteo por `decision` sobre un DataFrame sintético.
- `run_report`: casos de error — archivo inexistente, cero filas resueltas,
  N < 30 (warning presente), reporte normal (capsys, smoke test de que no
  crashea y contiene las secciones esperadas).

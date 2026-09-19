# Feature engineering — Fase 5 (2026-09-18, sesión 6)

4 features nuevas propuestas: momentum (N=3,5,10), surface win rate trend,
rest days (ya existía), H2H trend. Implementadas las 4 en
`FeatureBuilder.get_features()` (`src/features/engineering.py`) y
propagadas a `build_match_features` (backtest) y `build_prediction_features`
(predicción en vivo) — pero **solo 2 de las 4 se agregaron a
`FEATURE_COLS`**, con evidencia empírica de por medio.

## Implementación

- `FeatureBuilder._trend(results, min_n=4)`: win rate de la mitad reciente
  menos la mitad anterior de una lista de resultados — reutilizado para
  `surface_win_rate_trend` y `h2h_trend`. Neutral (0.0) con menos de 4
  resultados para comparar.
- `momentum_3`/`momentum_5`: win rate de los últimos 3/5 partidos (cualquier
  superficie). `momentum_10` ya existía como `recent_win_rate` (default
  `recent_n=10`), no se duplicó.
- `h2h_trend`: a diferencia de `h2h_rate`, es antisimétrico por
  construcción (`fa["h2h_trend"] == -fb["h2h_trend"]`) — se agregó a
  `_MIRROR_FLIP_COLS` (negación simple), no al tratamiento especial de
  `h2h_rate` (`1 - rate`).
- Todo mantiene el patrón no-lookahead existente: `get_features()` sigue
  llamándose antes que `update()`, sin cambios en ese orden.

## Validación empírica (regla: backtest no debe empeorar)

A/B walk-forward completo (warmup=10 años, años evaluados 2020-2026),
ATP (160K predicciones) y WTA (51K):

| Config | ATP Acc | ATP LogLoss | ATP Brier |
|---|---|---|---|
| Baseline (13 features) | 0.64940 | 0.61810 | 0.21504 |
| + h2h_trend solo | 0.64909 | 0.61809 | 0.21504 |
| + surface_win_rate_trend_diff solo | 0.64991 | 0.61810 | 0.21504 |
| + momentum_3_diff solo | 0.64913 | 0.61820 | 0.21509 |
| + momentum_5_diff solo | 0.64875 | 0.61817 | 0.21507 |
| **+ h2h_trend + surface_win_rate_trend_diff** | **0.64975** | **0.61810** | **0.21503** |
| + las 4 juntas | 0.64844 | 0.61819 | 0.21508 |

| Config | WTA Acc | WTA LogLoss | WTA Brier |
|---|---|---|---|
| Baseline (13 features) | 0.65142 | 0.61697 | 0.21451 |
| + h2h_trend + surface_win_rate_trend_diff | 0.65197 | 0.61701 | 0.21452 |

**Decisión**: se agregaron `h2h_trend` y `surface_win_rate_trend_diff` a
`FEATURE_COLS` (13→15). Mejora pequeña pero consistente en ambos tours, sin
costo de calibración/log-loss medible. `momentum_3_diff`/`momentum_5_diff`
**no se agregaron**: sin beneficio individual claro, y empeoran (leve) el
resultado cuando se combinan con las otras dos — quedan implementadas en
`FeatureBuilder.get_features()` por si se quiere revisitar con más datos,
pero no llegan al modelo. Mismo patrón que Módulo 3 (2026-07-28): solo se
adopta lo que mejora o es genuinamente neutral, no lo que empeora en
conjunto aunque un componente individual se vea bien aislado.

## Migración de esquema

Cambiar `FEATURE_COLS` (13→15) tocó más superficie de la esperada — no solo
el modelo, sino:
- `_MIRROR_FLIP_COLS` (backtest): 2 columnas nuevas para negar en mirror rows.
- `_LOG_FIELDS` + `_migrate_log_header_if_needed()` (`src/value_analysis.py`):
  el CSV log (`data/value_bets_log.csv`) tiene su propio header fijo,
  separado de `FEATURE_COLS` — se agregaron las 2 columnas ahí también, con
  default `"0.0"` para filas viejas vía el mecanismo de migración ya
  existente (mismo patrón usado para `odds_a_source`/`bookmaker_a` en su
  momento).
- 6 fixtures de test con CSVs/dicts hardcodeados de 13 columnas (loader
  stubs de `FeatureBuilder.get_features`, CSVs sintéticos para
  `_train_lr`/snapshots) necesitaron las 2 columnas nuevas para no romper.

Reentrenados los 3 modelos (ATP/WTA/Davis) con el esquema de 15 features y
verificado `predict_match()` end-to-end contra el modelo real.

## Tests

31 tests en `tests/test_features.py` (9 nuevos: momentum, surface trend,
h2h trend, rest_days pin), 3 nuevos en `tests/test_build_match_features.py`,
3 nuevos en `tests/test_features_schema.py` (nuevo archivo, fija qué
features llegan realmente al modelo y por qué). 663/663 tests pasando.

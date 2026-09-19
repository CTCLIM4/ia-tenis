# Investigación del calibration gap — 2026-09-18 (sesión 5, Fase 1)

Checkpoint: investigar el gap de -23.5pp (EV-teórico vs retorno-real,
diagnosticado 2026-08-27) con `_SHRINK_RATE` nuevo candidato. **Conclusión:
no se tocó `_SHRINK_RATE` — la evidencia dice que no hace falta.**

## El -23.5pp no es un problema de calibración

Ese número viene de `src/backtest_analytics.py`, comparando EV teórico vs
retorno real sobre los **108 apuestas reales** de `value_bets_log.csv`.
Con odds media 3.28 y **sd 4.52** (Módulo 4, sesión anterior), es una
muestra chica y de altísima varianza — perfectamente capaz de mostrar un
-23.5pp por pura mala suerte en unas pocas apuestas de odds altas, sin que
el modelo esté mal calibrado. Confundir "retorno real bajo" con
"probabilidades mal calibradas" es el error que este checkpoint evita.

## Medición correcta: calibración contra el backtest (N grande)

`src/calibration_metrics.py` (nuevo) mide el error de calibración real:
por cada bin de probabilidad predicha, compara la probabilidad media
predicha contra la frecuencia real de victorias, ponderado por tamaño de
bin. Requirió un fix en `src/backtest/walkforward.py`
(`return_predictions=True`, `probs_calibration`/`y_true_calibration`):
el walk-forward original solo evalúa la fila "perspectiva del ganador"
(outcome siempre 1 por la convención de mirror rows) — una curva de
calibración sobre eso da basura (frecuencia real "100%" en cada bin, sin
importar la probabilidad predicha). Sumando las mirror rows (perspectiva
del perdedor, outcome 0) se recupera una base balanceada de verdad
(base_rate exacto 0.5 en los tres tours).

**Resultado sobre el backtest completo (warmup=10 años, walk-forward
1990-2026 ATP / 2007-2026 WTA / 1981-2026 Davis):**

| Tour  | N predicciones | Error calibración [0.20,0.80] |
|-------|-----------------|-------------------------------|
| ATP   | 160,424          | 0.90 pp |
| WTA   | 50,790           | 0.77 pp |
| Davis | 14,828           | 2.01 pp |

Los tres muy por debajo del umbral de 3pp. El modelo **sí está bien
calibrado** en el rango de probabilidad donde se toman la mayoría de las
decisiones de apuesta.

## Por qué no se tocó `_SHRINK_RATE`

`apply_shrinkage` solo transforma predicciones fuera de
`[_SHRINK_LO, _SHRINK_HI] = [0.10, 0.90]` — por construcción, **no puede
afectar la calibración en [0.20, 0.80]** (confirmado: el error ahí es
idéntico bit a bit en rate=0.0, 0.4, 0.5, 0.6, 0.7). Probando rates
0.0-0.7 sobre el rango completo (0-1), las diferencias son de orden de
ruido (ATP 0.94-1.00pp, WTA 0.59-0.81pp, Davis 1.84-2.19pp) — el valor
actual (0.6) ya está dentro de ese rango sin ganancia clara de ningún
otro candidato. Forzar un cambio de hyperparámetro sobre diferencias de
este tamaño sería sobreajustar ruido — el mismo motivo por el que la
decisión del 2026-08-27 tampoco tocó `apply_shrinkage`
([[project_kelly_quarter_2026-08-27]] en memoria: "N=28... arriesgaría
sobreajustar ruido de muestra chica"). Con N=108 reales seguimos sin
tener la potencia estadística para superar esa barrera vía el gap
financiero, pero el backtest (N=160K/51K/15K) sí la tiene, y dice que no
hace falta tocar nada.

## Qué haría falta para mover la aguja en el -23.5pp real

Nada de esto es un problema de modelo: es varianza de muestra chica y
odds altas. Las palancas reales ya están aplicadas (1/4 Kelly desde
2026-08-27, justo para recortar el impacto de esa varianza). Más datos
(más apuestas reales acumuladas) es lo único que reduce genuinamente el
ruido de esa medición específica — no hay atajo de hyperparámetro.

## Tests nuevos

`tests/test_calibration_metrics.py` (9 tests, datos sintéticos —
portable, no depende de `data/processed/` que está gitignored) +
3 tests nuevos en `tests/test_backtest.py::TestReturnPredictions` para
la instrumentación `return_predictions`/`probs_calibration`. 612/612
tests pasando.

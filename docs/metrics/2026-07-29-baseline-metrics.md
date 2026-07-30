# Baseline de métricas — 2026-07-29

Resumen financiero acumulado (Módulo 2, `src/backtest_analytics.py`) y
simulación Monte Carlo de banca a 50 apuestas (Módulo 4,
`src/bankroll_simulation.py`), ambos corridos sobre el estado actual de
`data/value_bets_log.csv` (7 apuestas resueltas).

## Módulo 2 — Rendimiento y banca (histórico)

Comando: `python -m src.backtest_analytics --bankroll 1000`

```
============================================================
 REPORTE DE RENDIMIENTO Y BANCA - Modulo 2
 Banca base: $1,000.00 | Periodo: 2026-07-02 a 2026-07-29 | Apuestas: 7
============================================================

  *** Muestra pequena (N=7 < 30) - metricas poco confiables todavia. ***

METRICAS FINANCIERAS
  ROI acumulado:            -14.40%
  Win Rate:                   57.1%   (4W / 3L)
  Beneficio Neto:           -45.70$
  Stake promedio:            45.34$   (4.53% banca)
  Apuestas procesadas:          7

METRICAS DE RIESGO
  Drawdown maximo:          -150.00$   (-14.8%)
  Varianza (profit):        0.00172
  Racha ganadora maxima:        3
  Racha perdedora maxima:       3

CALIBRACION: EV TEORICO vs BENEFICIO REAL
  EV teorico promedio (por unidad):    +11.5%
  Retorno real promedio (por unidad):  +5.0%
  Diferencia (real - teorico):         -6.5 pp

COBERTURA DEL AUDIT LOG (prediction_audit_log.csv)
  Predicciones evaluadas totales:   16
    logged                   5 (31.2%)
    passed_low_edge          3 (18.8%)
    passed_user_declined     7 (43.8%)
    blocked_suspicious       1 (6.2%)
    invalid_missing_elo      0 (0.0%)
  Logged (apostadas) vs resto:  5 vs 11
```

## Módulo 4 — Simulación Monte Carlo de banca (50 apuestas)

Comando: `python -m src.bankroll_simulation --bets 50 --simulations 10000 --seed 42`

```
============================================================
 SIMULACION MONTE CARLO DE BANCA - Modulo 4
 Banca inicial: $1,000.00 | 10000 simulaciones x 50 apuestas
============================================================

 Perfil: edge medio 6.8% (sd 3.0%) | odds medias 1.85 (sd 0.56) | N=7

  *** Perfil estimado de muestra chica (N=7 < 30) - usar con cautela. ***

                              Kelly completo       1/2 Kelly       1/4 Kelly
  Banca final P10                    $897.84         $932.35         $970.47
  Banca final P50 (mediana)        $1,281.89       $1,237.81       $1,155.48
  Banca final P90                  $1,836.48       $1,618.13       $1,360.46
  Prob. de ruina                        0.1%            0.0%            0.0%
  Drawdown maximo esperado            -18.8%          -14.3%           -9.1%
```

## Lectura

El histórico real (N=7) está en rojo (-14.40% ROI, -$45.70), con una
diferencia de calibración de -6.5 pp entre el EV teórico y el retorno real
observado. La simulación paramétrica (calibrada con ese mismo perfil de
edge/odds) proyecta crecimiento esperado a 50 apuestas en las tres
fracciones de Kelly porque el edge promedio observado es positivo (+6.8%);
Kelly completo da el mayor upside (P90 $1,836) pero también el mayor
drawdown esperado (-18.8%), mientras que 1/4 Kelly recorta ambos extremos.

Ambos reportes llevan la misma advertencia: N=7 está muy por debajo del
umbral de 30 que ambos módulos usan para considerar las métricas
confiables — esta es una baseline de referencia, no una conclusión sobre
la estrategia de staking todavía.

# Calibración y análisis de valor — 2026-09-18

Checkpoint Día 4 del plan de descarga/procesamiento. Corrido sobre los
datos refrescados el mismo día (ver
`2026-09-18-pipeline-refresh-walkforward.md`) y sobre el estado actual
de `data/value_bets_log.csv` / `prediction_audit_log.csv`.

## Módulo 2 — Rendimiento y banca (histórico completo, N=108)

Comando: `python -m src.backtest_analytics --bankroll 1000`

```
 Periodo: 2026-07-02 a 2026-09-03 | Apuestas: 108

METRICAS FINANCIERAS
  ROI acumulado:            +6.92%
  Win Rate:                   49.1%   (53W / 55L)
  Beneficio Neto:           +202.63$
  Stake promedio:            27.10$   (2.71% banca)

METRICAS DE RIESGO
  Drawdown maximo:          -188.00$   (-14.0%)
  Racha ganadora maxima:        6
  Racha perdedora maxima:       8

CALIBRACION: EV TEORICO vs BENEFICIO REAL
  EV teorico promedio (por unidad):    +19.0%
  Retorno real promedio (por unidad):  -4.4%
  Diferencia (real - teorico):         -23.5 pp

COBERTURA DEL AUDIT LOG (625 predicciones evaluadas)
  logged                 119 (19.0%)
  passed_low_edge         72 (11.5%)
  passed_user_declined   216 (34.6%)
  blocked_low_sample      88 (14.1%)
  blocked_suspicious_edge 123 (19.7%)
```

Ya con N=108 (> umbral de 30), el ROI acumulado sigue positivo
(+6.92%) pero la brecha de calibración EV-teórico vs retorno-real
sigue siendo grande y negativa (-23.5 pp) — consistente con el sesgo
de sobreconfianza que ya motivó el recorte a 1/4 Kelly el 2026-08-27
(ver `project_kelly_quarter_2026-08-27` en memoria). No se tocó
`apply_shrinkage` en esta sesión por la misma razón documentada
entonces: con la muestra actual el ajuste de shrinkage sigue siendo
territorio de sobreajuste, no de conclusión estadística firme.

## Módulo 4 — Monte Carlo de banca (50 apuestas, perfil real N=108)

Comando: `python -m src.bankroll_simulation --bets 50 --simulations 10000 --seed 42`

```
 Perfil: edge medio 6.0% (sd 2.3%) | odds medias 3.28 (sd 4.52) | N=108

                              Kelly completo       1/2 Kelly       1/4 Kelly
  Banca final P50 (mediana)        $1,173.40       $1,112.52       $1,072.96
  Banca final P90                  $1,561.74       $1,334.14       $1,184.40
  Prob. de ruina                        0.0%            0.0%            0.0%
  Drawdown maximo esperado            -14.9%           -9.7%           -5.8%
```

Perfil de riesgo similar al del baseline de 2026-07-29 pero con muestra
mucho más grande y confiable. 1/4 Kelly (la fracción vigente desde
2026-08-27) sigue siendo la opción de menor drawdown esperado sin
aumentar el riesgo de ruina, que ya era 0% en las tres fracciones.

## Análisis de value-bets del día (scan en vivo, `--dry-run`)

`src/value_analysis.py` es un REPL interactivo (pide odds manuales por
stdin) — no corre en modo no interactivo. El equivalente de producción
es `scripts/daily_workflow.py`, que sí se corrió (`--dry-run`, sin
efectos: no email, no push, no log):

```
*** ADVERTENCIA: dataset ATP desactualizado (regla: regular_week) ***
*** Ultimo partido en los datos: 2026-09-13 (5 dias atras).
*** ADVERTENCIA: dataset WTA desactualizado (regla: regular_week) ***
*** Ultimo partido en los datos: 2026-09-12 (6 dias atras).
  ATP desactualizado: se ignoran picks ATP de esta jornada.
  WTA desactualizado: se ignoran picks WTA de esta jornada.
0 pick(s) seleccionados (edge >= 3%, 1/4 Kelly)
```

Ambos tours quedaron excluidos por el gate de staleness simétrico
(commit 2026-08-28) — no es un problema de descarga (los datos están
al día al 2026-09-18, ver Día 1-2), sino que no hay partidos más
recientes que el 13/12 de septiembre en las fuentes crudas todavía.
Confirmado con el usuario: se deja en dry-run, sin enviar el correo
informativo de "0 picks" ni tocar el repositorio.

## Verificación de tests

`pytest` completo: **554/554 passed** (548 originales + 6 nuevos de la
corrección de descarga WTA del Día 1-2).

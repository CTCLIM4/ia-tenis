# Jornada 2026-08-17 — ATP Cincinnati Open (Round of 32)

3 de 4 picks pendientes cerrados en `data/value_bets_log.csv` (filas match_date 2026-08-17). El pick Paul vs Vallejo queda **pendiente** — ver nota al final.

| Partido | Pick (edge+) | Odds | Kelly stake | Resultado real | Profit (fracción banca) |
|---|---|---|---|---|---|
| Blockx vs Cobolli | Cobolli (B) | 2.08 | 0.0250 | B_win (Cobolli, 7-5, 4-6, 7-5) | +0.0270 |
| Tirante vs Landaluce | Tirante (A) | 1.75 | 0.0250 | A_win (Tirante, 7-6(5), 7-6(4)) | +0.01875 |
| Jodar vs Tabilo | Jodar (A) | 1.46 | 0.0250 | A_win (Jodar, 6-2, 6-1) | +0.0115 |
| Paul vs Vallejo | Vallejo (B) | 4.55 | 0.0140 | **pendiente** (ver nota) | — |

**Resumen de la jornada** (banca base $1,000, sobre los 3 picks resueltos; 1 sigue pendiente):
- Apuestas resueltas: 3, Win rate: 100% (3W/0L)
- Stake total: 7.5% banca ($75.00)
- Beneficio de la jornada: +5.725% banca (+$57.25)
- ROI de la jornada (profit/stake): +76.33%

**Acumulado del proyecto tras esta jornada** (`python -m src.backtest_analytics --bankroll 1000`, N=65, 2026-07-02 a 2026-08-17):
- ROI acumulado: +10.49% | Win rate: 60.0% (39W/26L)
- Beneficio neto: +$263.83 | Stake promedio: $38.68 (3.87% banca)
- Drawdown máximo: -$178.30 (-17.6%) | Varianza (profit): 0.00131
- Racha ganadora máxima: 4 | Racha perdedora máxima: 4
- Calibración: EV teórico +12.9% vs retorno real +7.1% (diff -5.8 pp)

## Nota: Paul vs Vallejo no se liquidó

Se consultaron múltiples fuentes (WebSearch + WebFetch a tennismajors.com, tennis.com,
perfect-tennis.com, tennisuptodate.com) y los resultados fueron **contradictorios y no
confiables**: una fuente reportó a Paul ganador citando un marcador que en realidad
correspondía a su partido de ronda anterior contra Hurkacz (6-4, 6-7(3), 6-3), y otra
reportó a Vallejo ganador con un "marcador" que resultó ser una mala lectura de números
de la URL, no datos reales del partido. Ninguna fuente de recap/resultados confirmado
(ATP Tour results, recaps de día) mostró el partido como finalizado. Dado el principio
de no usar resultados no verificados para liquidar picks ([[feedback-no-sample-contamination]]),
la fila correspondiente en `data/value_bets_log.csv` se deja en `status=pending` para
liquidar en una jornada posterior cuando el resultado pueda confirmarse con una fuente fiable.

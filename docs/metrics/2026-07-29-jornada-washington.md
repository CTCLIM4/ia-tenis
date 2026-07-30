# Jornada 2026-07-29 — ATP/WTA Washington Open

3/3 aciertos. Picks resueltos en `data/value_bets_log.csv` (filas `2026-07-28T23:39:00/01`).

| Partido | Pick (edge+) | Odds | Kelly stake | Resultado | Profit (fracción banca) |
|---|---|---|---|---|---|
| Vukic vs Musetti | Musetti (B) | 1.41 | 0.05 | B_win | +0.0205 |
| de Minaur vs Tsitsipas | de Minaur (A) | 1.75 | 0.05 | A_win | +0.0375 |
| Navarro vs Cocciaretto | Cocciaretto (B) | 2.94 | 0.0174 | B_win | +0.0338 |

**Resumen de la jornada** (banca base $1,000):
- Apuestas: 3, Win rate: 100% (3/3)
- Stake total: 11.74% banca ($117.40)
- Beneficio de la jornada: +9.18% banca (+$91.80)
- ROI de la jornada (profit/stake): +78.2%

**Acumulado del proyecto tras esta jornada** (`python -m src.backtest_analytics --bankroll 1000`, N=7):
- ROI acumulado: -14.40% | Win rate: 57.1% (4W/3L)
- Beneficio neto: -$45.70 | Stake promedio: $45.34 (4.53% banca)
- Drawdown máximo: -$150.00 (-14.8%) | Varianza (profit): 0.00172
- Racha ganadora máxima: 3 | Racha perdedora máxima: 3
- Calibración: EV teórico +11.5% vs retorno real +5.0% (diff -6.5 pp)

Nota: N=7 sigue por debajo del umbral de 30 para métricas confiables (ver Módulo 2).

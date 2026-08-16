# Jornada 2026-08-16 — ATP Cincinnati Open (qualy + R1)

12 picks pendientes cerrados en `data/value_bets_log.csv` (filas match_date 2026-08-11 a 2026-08-14).
10 resueltos (5W/5L), 2 anulados (`excluded`, profit=0) por no haberse disputado el cruce apostado.

| Partido | Pick (edge+) | Odds | Kelly stake | Resultado real | Profit (fracción banca) |
|---|---|---|---|---|---|
| Lajovic vs Bonzi | Lajovic (A) | 2.72 | 0.0200 | A_win (7-6, 7-6) | +0.0344 |
| Mpetshi Perricard vs Svajda | Mpetshi Perricard (A) | 1.56 | 0.0500 | A_win (6-1, 1-6, 6-2) | +0.0280 |
| Faria vs Basilashvili | Basilashvili (B) | 2.81 | 0.0211 | A_win (Faria, 6-4 6-4) | -0.0211 |
| Trungelliti vs Ofner | Trungelliti (A) | 2.36 | 0.0224 | A_win (4-6, 6-4, 7-6) | +0.0305 |
| Gaston vs Budkov Kjaer | Budkov Kjaer (B) | 2.13 | 0.0297 | B_win (6-2, 3-6, 6-4) | +0.0336 |
| Wolf vs Samuel | Samuel (B) | 1.66 | 0.0500 | **excluded** — Samuel se retiró del torneo antes de jugar (walkover) | 0.0000 |
| Lajal vs Svrcina | Svrcina (B) | 1.89 | 0.0500 | A_win (Lajal, 5-7, 6-4, 6-3) | -0.0500 |
| Shimabukuro vs Wolf | Shimabukuro (A) | 2.14 | 0.0500 | B_win (Wolf, ronda final qualy) | -0.0500 |
| Fucsovics vs Atmane | Fucsovics (A) | 2.78 | 0.0380 | B_win (Atmane, 3-1 RET — Fucsovics se retiró por espalda) | -0.0380 |
| Hijikata vs Monfils | Monfils (B) | 1.77 | 0.0500 | A_win (Hijikata, 2-6, 7-6, 6-3) | -0.0500 |
| Van de Zandschulp vs Griekspoor | Griekspoor (B) | 2.06 | 0.0499 | **excluded** — ninguno jugó el cruce (ambos salieron del draw; Shevchenko entró como lucky loser) | 0.0000 |
| Baez vs Dimitrov | Baez (A) | 2.27 | 0.0500 | A_win (1-6, 6-3, 6-4) | +0.0635 |

**Resumen de la jornada** (banca base $1,000, sobre los 10 picks resueltos; excluye los 2 anulados):
- Apuestas resueltas: 10, Win rate: 50% (5W/5L)
- Stake total: 38.12% banca ($381.20)
- Beneficio de la jornada: -1.91% banca (-$19.10)
- ROI de la jornada (profit/stake): -5.01%

**Acumulado del proyecto tras esta jornada** (`python -m src.backtest_analytics --bankroll 1000`, N=54):
- ROI acumulado: +9.72% | Win rate: 59.3% (32W/22L)
- Beneficio neto: +$221.36 | Stake promedio: $42.17 (4.22% banca)
- Drawdown máximo: -$188.00 (-14.0%) | Varianza (profit): 0.00153
- Racha ganadora máxima: 4 | Racha perdedora máxima: 4
- Calibración: EV teórico +13.2% vs retorno real +8.2% (diff -5.0 pp)

Nota: los 2 casos `excluded` (Wolf/Samuel, Van de Zandschulp/Griekspoor) son walkovers/withdrawals previos al partido — no hay resultado real que asignar a ninguno de los dos lados apostados, así que se anulan (profit=0) en vez de forzar A_win/B_win artificialmente.

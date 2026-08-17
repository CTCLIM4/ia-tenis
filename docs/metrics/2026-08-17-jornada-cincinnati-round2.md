# Jornada 2026-08-16/17 — ATP/WTA Cincinnati Open (Round of 64)

8 picks pendientes cerrados en `data/value_bets_log.csv` (filas match_date 2026-08-16/2026-08-17). Todos resueltos, 0 anulados.

| Partido | Pick (edge+) | Odds | Kelly stake | Resultado real | Profit (fracción banca) |
|---|---|---|---|---|---|
| Fonseca vs Van de Zandschulp | Van de Zandschulp (B) | 3.40 | 0.0074 | A_win (Fonseca, 6-4, 7-6(2)) | -0.0074 |
| Altmaier vs Musetti | Altmaier (A) | 4.61 | 0.0050 | B_win (Musetti, 6-4, 6-2) | -0.0050 |
| Tsitsipas vs Auger-Aliassime | Auger-Aliassime (B) | 1.75 | 0.0245 | B_win (Auger-Aliassime, 6-3, 7-5) | +0.018375 |
| Krejcikova vs Bejlek | Krejcikova (A) | 1.62 | 0.0250 | B_win (Bejlek, 7-6(5), 6-4) | -0.0250 |
| Tjen vs Golubic | Golubic (B) | 2.04 | 0.0250 | A_win (Tjen, 7-6(8), 5-7, 7-5) | -0.0250 |
| Gauff vs Samsonova | Gauff (A) | 1.33 | 0.0250 | A_win (Gauff, 2-6, 6-4, 6-1) | +0.00825 |
| Rublev vs Carreno Busta | Rublev (A) | 1.49 | 0.0250 | A_win (Rublev, 7-6(5), 6-1) | +0.01225 |
| Valentova vs Svitolina | Svitolina (B) | 1.35 | 0.0250 | B_win (Svitolina, 3-6, 7-6(5), 6-0) | +0.00875 |

**Resumen de la jornada** (banca base $1,000, sobre los 8 picks resueltos; sin anulados):
- Apuestas resueltas: 8, Win rate: 50% (4W/4L)
- Stake total: 16.19% banca ($161.90)
- Beneficio de la jornada: -1.4775% banca (-$14.78)
- ROI de la jornada (profit/stake): -9.13%

**Acumulado del proyecto tras esta jornada** (`python -m src.backtest_analytics --bankroll 1000`, N=62):
- ROI acumulado: +8.47% | Win rate: 58.1% (36W/26L)
- Beneficio neto: +$206.58 | Stake promedio: $39.34 (3.93% banca)
- Drawdown máximo: -$178.30 (-17.6%) | Varianza (profit): 0.00136
- Racha ganadora máxima: 4 | Racha perdedora máxima: 4
- Calibración: EV teórico +12.8% vs retorno real +3.8% (diff -9.1 pp)

Nota: el resultado de Valentova vs Svitolina se verificó cruzando tres fuentes independientes (WebSearch, LiveScore, búsqueda adicional) tras detectar una discrepancia con un primer fetch de wtatennis.com que reportaba a Valentova como ganadora en un marcador inválido (6-6, 6-3); las tres fuentes coincidentes confirman a Svitolina ganadora 3-6, 7-6(5), 6-0.

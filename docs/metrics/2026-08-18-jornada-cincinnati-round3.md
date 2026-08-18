# Jornada 2026-08-17 — ATP Cincinnati Open (Round of 32)

4 de 4 picks pendientes cerrados en `data/value_bets_log.csv` (filas match_date 2026-08-17). El pick Paul vs Vallejo se liquidó en una segunda pasada tras confirmar el resultado con fuentes cruzadas — ver nota al final.

| Partido | Pick (edge+) | Odds | Kelly stake | Resultado real | Profit (fracción banca) |
|---|---|---|---|---|---|
| Blockx vs Cobolli | Cobolli (B) | 2.08 | 0.0250 | B_win (Cobolli, 7-5, 4-6, 7-5) | +0.0270 |
| Tirante vs Landaluce | Tirante (A) | 1.75 | 0.0250 | A_win (Tirante, 7-6(5), 7-6(4)) | +0.01875 |
| Jodar vs Tabilo | Jodar (A) | 1.46 | 0.0250 | A_win (Jodar, 6-2, 6-1) | +0.0115 |
| Paul vs Vallejo | Vallejo (B) | 4.55 | 0.0140 | A_win (Paul avanza a octavos vs Zverev) | -0.0140 |

**Resumen de la jornada** (banca base $1,000, sobre los 4 picks resueltos):
- Apuestas resueltas: 4, Win rate: 75% (3W/1L)
- Stake total: 8.9% banca ($89.00)
- Beneficio de la jornada: +4.325% banca (+$43.25)
- ROI de la jornada (profit/stake): +48.60%

**Acumulado del proyecto tras esta jornada** (`python -m src.backtest_analytics --bankroll 1000`, N=66, 2026-07-02 a 2026-08-17):
- ROI acumulado: +9.88% | Win rate: 59.1% (39W/27L)
- Beneficio neto: +$249.83 | Stake promedio: $38.30 (3.83% banca)
- Drawdown máximo: -$188.00 (-14.0%) | Varianza (profit): 0.00130
- Racha ganadora máxima: 4 | Racha perdedora máxima: 4
- Calibración: EV teórico +13.4% vs retorno real +5.5%

## Nota: cómo se confirmó Paul vs Vallejo

La primera pasada de liquidación (mismo día) dejó esta fila en `pending` porque las dos
primeras fuentes consultadas eran contradictorias y ambas resultaron poco fiables: una
citaba el marcador del partido anterior de Paul contra Hurkacz como si fuera contra
Vallejo, y otra "leía" dígitos de la URL de la página como si fueran el marcador real.

En una segunda pasada, a pedido del usuario, se buscó evidencia estructural independiente
en vez de depender de un único marcador textual:
- El fixture de Flashscore para Tommy Paul lista su siguiente partido como "Zverev vs.
  Paul" el 19 de agosto — solo existe si Paul avanzó tras vencer a Vallejo.
- El cuadro (draw) de Wikipedia del torneo marca a Paul en negrita como el jugador que
  avanza en esa llave.
- Un artículo de recap (tennismajors.com) y una búsqueda adicional coinciden en que
  "Paul overcame Vallejo" tras ~2h20 de partido.

Cuatro señales independientes (fixture del siguiente partido, cuadro del torneo, y dos
recaps distintos) coinciden en que ganó Paul, frente a una única fuente descartada por
fabricar el marcador. Con ese nivel de corroboración se liquida la fila como `A_win`
(pierde el pick por Vallejo). El marcador exacto del partido no pudo confirmarse con
la misma confianza que el ganador, así que no se reporta un score set-by-set aquí.

# Modelo vs mercado: el edge no se sostiene — 2026-10-07

Reproducir: `python -m scripts.market_edge_backtest`.

## Pregunta

El log real (116 apuestas liquidadas) muestra EV teórico **+19,3%** frente a
retorno real **−5,0%**. La investigación del 2026-09-18 lo atribuyó a la
varianza de una muestra chica, porque la calibración **global** del modelo
en el backtest es buena (error < 1 pp en [0,20; 0,80]).

Esa medición responde "¿P(victoria) es correcta en promedio?". No responde
la pregunta de las apuestas: **en los partidos donde el modelo discrepa del
mercado lo bastante como para apostar, ¿quién acierta?** Un modelo puede
estar bien calibrado en promedio y aun así perder sistemáticamente justo
donde discrepa, si el mercado es más informativo (selección adversa).

## Método

- Predicciones fuera de muestra del walk-forward WTA (mismo esquema que
  `src/backtest/walkforward.py`: entrenar con años anteriores, warmup de
  10 años, shrinkage de producción), 2016–2026.
- Unidas por (fecha, ganadora, perdedora) a las cuotas históricas de
  Tennis-Data: media del mercado, Bet365, máxima, y Pinnacle hasta 2025.
  25.479 de 25.589 partidos tienen cuotas.
- Se reproduce la regla de producción: 3% ≤ p − 1/cuota ≤ 10%, stake
  Kelly/4 con tope 5%. Ambos lados de cada partido son candidatos.
- **Solo WTA**: la fuente ATP (Tennismylife) no trae cuotas.

## Resultados

**1. El mercado es más preciso que el modelo, y el modelo no le añade nada.**

| | Log-loss | Brier |
|---|---|---|
| Modelo | 0,6221 | 0,2167 |
| Mercado (media, sin margen) | **0,5951** | **0,2051** |
| Pinnacle (subconjunto) | 0,5971 vs modelo 0,6240 | |

La mejor mezcla `a·modelo + (1−a)·mercado` da **a = 0**: añadir el modelo a
la probabilidad del mercado empeora la predicción.

**2. La regla de producción pierde dinero, y pierde más que apostar al azar.**

| Precio | N | p modelo | q mercado | Victorias reales | EV teórico | ROI plano (IC 95%) | ROI Kelly |
|---|---|---|---|---|---|---|---|
| Media del mercado | 9.038 | 0,474 | 0,392 | **0,376** | +19,2% | **−12,3%** (−15,0..−9,5) | −9,2% |
| Bet365 | 8.848 | 0,467 | 0,386 | 0,371 | +20,2% | −11,4% (−14,3..−8,6) | −8,7% |
| Mejor cuota (optimista) | 11.440 | 0,470 | 0,412 | 0,403 | +21,6% | −3,9% (−6,5..−1,2) | −1,2% |

Apostar a **todos** los lados a cuota media da −7,2%, que es simplemente el
margen de la casa. La selección del modelo (−12,3%) es **peor que no
seleccionar**: sus "edges" señalan justo los partidos donde se equivoca.

El EV teórico del backtest (+19,2%) coincide casi exactamente con el del log
real (+19,3%). No es mala suerte de la muestra real: es lo que el modelo
produce de forma estructural.

**3. La tasa real sigue al mercado, no al modelo, en todos los tramos.**

- Por cuota: en todas las bandas, las victorias reales quedan en o por
  debajo de la q del mercado, nunca cerca de la p del modelo. Con cuotas
  > 5 el modelo dice 20,6%, el mercado 13,7% y la realidad 9,7%
  (ROI −36%).
- Por edge: **cuanto más edge declara el modelo, peor el ROI** (3–5%:
  −10,2%; 5–7%: −13,1%; 7–10%: −14,0%). Un edge real se comportaría al revés.
- Por año: ROI negativo en los 11 años (de −1,5% a −21,4%), EV teórico
  siempre en torno a +17–22%.
- Ningún filtro probado rescata un ROI positivo (cuotas < 2: −6,3%;
  p ≥ 0,6: −5,8%; edges 1–3%: −10,1%).

## El log real

| Tour | N | p modelo | q mercado | Victorias reales | ROI plano |
|---|---|---|---|---|---|
| ATP | 83 | 0,505 | 0,435 | 0,482 | +1,6% |
| WTA | 33 | 0,542 | 0,473 | 0,485 | −25,7% |

La WTA real cuadra con el backtest. La ATP real queda entre modelo y
mercado con un ROI ligeramente positivo, pero con N=83 y cuotas de alta
varianza eso no distingue un edge real de la suerte, y **la ATP no se
puede validar con este backtest** porque su fuente no trae cuotas.

## Conclusiones

1. La investigación del 2026-09-18 era correcta en lo que medía (la
   calibración global es buena) pero no medía lo que importa para apostar.
   El gap de −24 pp **no es solo varianza**: en WTA es la selección adversa
   del modelo frente a un mercado más informado.
2. Con la regla actual, el modelo WTA no tiene ventaja demostrable sobre el
   mercado; la evidencia (N=9.038, 11 años, IC claramente negativo) apunta a
   lo contrario. Tocar `_SHRINK_RATE` o `MIN_EDGE` no lo arregla: ningún
   umbral de edge da ROI positivo.
3. El EV teórico que reportan el scanner y el workflow (+15–20% típico) no
   es una estimación creíble del retorno esperado y no debería usarse para
   dimensionar stakes.

## Siguientes pasos posibles (no aplicados)

- **Validar ATP**: Tennis-Data publica también ficheros ATP con las mismas
  columnas de cuotas; descargarlos permitiría repetir este análisis donde
  está la mayoría de las apuestas reales.
- **Pausar o reducir el auto-registro WTA** mientras no haya evidencia de
  edge: la tarea programada registra picks WTA cada día.
- **Usar el mercado como base**: modelar la desviación respecto de la
  probabilidad del mercado (p. ej. features + logit(q) como input) en vez de
  predecir desde cero, y validar con este mismo script.

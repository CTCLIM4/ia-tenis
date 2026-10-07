# Modelo vs mercado: el edge no se sostiene — 2026-10-07

Reproducir: `python -m scripts.market_edge_backtest [--tour wta|atp]`.

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
- ATP: la fuente del modelo (Tennismylife) no trae cuotas, así que se
  descargaron los ficheros ATP de Tennis-Data (2000–2026) en
  `data/raw/tennis_atp_tduk_odds/`. Ver sección ATP.

## WTA

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
mercado con un ROI ligeramente positivo (N=83, no concluyente por sí solo);
ver la sección ATP.

## ATP

**Cruce de datos.** Tennismylife usa nombres completos ("Carlos Alcaraz") y
fecha el partido con el inicio del torneo; Tennis-Data usa "Alcaraz C." y
la fecha real. Se cruza por (último token del apellido, inicial) de ambos
jugadores, con la fecha de Tennis-Data entre 3 días antes y 20 después, y
solo pares 1:1. Se cruzan 25.877 de 27.210 filas de Tennis-Data 2016+ (95%);
lo que no cruza del lado Tennismylife es sobre todo Copa Davis y niveles
que Tennis-Data no cubre. Con cuota media disponible: 39.442 partidos,
2010–2026.

**1. El mercado también es más preciso, pero aquí el modelo sí aporta algo.**

| | Log-loss | Brier |
|---|---|---|
| Modelo | 0,5930 | 0,2039 |
| Mercado (media, sin margen) | **0,5789** | **0,1984** |
| Pinnacle (subconjunto) | 0,5770 vs modelo 0,5918 | |

Mejor mezcla: **a = 0,25** (log-loss 0,5773). Al contrario que en WTA, el
modelo contiene información que el mercado no incorpora del todo, aunque
el mercado sigue dominando.

**2. La regla de producción no supera el margen a precios realistas.**

| Precio | N | p modelo | q mercado | Victorias reales | EV teórico | ROI plano (IC 95%) | ROI Kelly |
|---|---|---|---|---|---|---|---|
| Media del mercado | 13.645 | 0,460 | 0,377 | 0,394 | +23,1% | **−5,0%** (−7,6..−2,5) | −1,3% |
| Bet365 | 13.544 | 0,448 | 0,367 | 0,384 | +26,2% | −3,7% (−6,4..−0,8) | −0,1% |
| Mejor cuota (optimista) | 16.792 | 0,456 | 0,399 | 0,408 | +30,4% | +3,1% (+0,2..+6,1) | +4,3% |

Apostar a todos los lados a cuota media da −7,1%: la selección del modelo
mejora unos 2 pp sobre el azar, pero no lo suficiente para cubrir el margen.
Solo con la **mejor cuota del mercado** (el máximo entre todas las casas,
al cierre; un techo que producción no alcanza siempre) sale positivo. El
log real ATP (+1,6%, N=83) cae dentro de este rango.

**3. Todo el daño está en las cuotas altas; los favoritos rozan el equilibrio.**

| Subconjunto (cuota media) | N | ROI plano (IC 95%) |
|---|---|---|
| Cuota ≤ 3 | 8.366 | +0,9% (−1,2..+2,9) |
| Cuota > 3 | 5.279 | −14,3% (−19,8..−8,6) |

Con cuotas > 5 el modelo dice 18,4%, el mercado 11,7% y la realidad 9,8%
(ROI −23,5%). Con la mejor cuota, el tramo ≤ 3 da +4,4% (IC +2,5..+6,3).

**Pero la ventaja se está erosionando.** El tramo de favoritos a cuota media
pasa de +3,0% en 2010–2017 a −0,8% en 2018–2026, con 2023 (−6,0%),
2024 (−3,8%) y 2025 (−6,6%) en negativo. A mejor cuota: +6,6% → +2,7%,
y 2023–2026 en torno a cero. Este corte por cuota sale de mirar estas
mismas tablas, así que es una hipótesis, no una regla validada.

## Conclusiones

1. La investigación del 2026-09-18 era correcta en lo que medía (la
   calibración global es buena) pero no medía lo que importa para apostar.
   El gap de −24 pp **no es solo varianza**: es la sobreestimación
   sistemática del edge frente a un mercado más informado.
2. **WTA**: sin ventaja; la selección es peor que el azar (N=9.038, 11 años,
   IC claramente negativo). Excluida del auto-registro desde 2026-10-07,
   (`AUTO_LOG_EXCLUDED_TOURS` en `scripts/daily_workflow.py`).
3. **ATP**: el modelo aporta información (peso 0,25 en la mezcla) pero la
   regla actual pierde a precio medio (−5,0%) y solo gana a la mejor cuota
   disponible (+3,1%). Las cuotas > 3 pierden con claridad; los favoritos
   rozan el equilibrio, con una ventaja que se ha ido erosionando y es ~0
   en los últimos años.
4. El EV teórico del scanner y el workflow (+19–30% de media en las
   apuestas elegidas) no es una estimación creíble del retorno esperado y
   no debería usarse para dimensionar stakes. Tocar `_SHRINK_RATE` o
   `MIN_EDGE` no lo arregla: ningún umbral de edge da ROI positivo a
   precio medio.

## Siguientes pasos posibles (no aplicados)

- ~~ATP: limitar a cuotas ≤ 3, o excluir también ATP del auto-registro.~~
  **Aplicado 2026-10-07: ATP excluida** junto con WTA
  (`AUTO_LOG_EXCLUDED_TOURS = {"wta", "atp"}`). Se descartó limitar a
  cuotas ≤ 3: ese tramo está en equilibrio a precio medio, en declive, y el
  corte salió de estas mismas tablas. Ambos tours se siguen escaneando y
  auditando; solo Davis Cup (sin cuotas con que validarla) puede seguir
  registrándose.
- **Usar el mercado como base**: modelar la desviación respecto de la
  probabilidad del mercado (p. ej. features + logit(q) como input) en vez
  de predecir desde cero. El peso 0,25 en ATP sugiere que hay algo que
  aprovechar; validarlo con este mismo script.
- **Reportar un EV realista**: sustituir la p del modelo por la mezcla con
  el mercado al calcular edge y EV.

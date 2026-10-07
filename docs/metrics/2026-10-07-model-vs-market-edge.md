# Modelo vs mercado: el edge no se sostiene — 2026-10-07

Reproducir: `python -m scripts.market_edge_backtest [--tour wta|atp] [--anchored]`.

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

- Predicciones fuera de muestra del walk-forward (mismo esquema que
  `src/backtest/walkforward.py`: entrenar con años anteriores, warmup de
  10 años, shrinkage de producción).
- Unidas a las cuotas históricas de Tennis-Data: media del mercado, Bet365,
  máxima, y Pinnacle hasta 2025.
- Se reproduce la regla de producción: 3% ≤ p − 1/cuota ≤ 10%, stake
  Kelly/4 con tope 5%. Ambos lados de cada partido son candidatos.
- **WTA**: 25.479 de 25.589 partidos 2016–2026 con cuotas, cruzados por
  (fecha, ganadora, perdedora).
- **ATP**: la fuente del modelo (Tennismylife) no trae cuotas; se
  descargaron los ficheros ATP de Tennis-Data (2000–2026) en
  `data/raw/tennis_atp_tduk_odds/` (ignorado por git). Tennismylife usa
  nombres completos ("Carlos Alcaraz") y fecha cada partido con el inicio
  del torneo; Tennis-Data usa "Alcaraz C." y la fecha real. Se cruza por
  (último token del apellido, inicial) de ambos jugadores, con la fecha de
  Tennis-Data entre 3 días antes y 20 después, solo pares 1:1: el 95% de
  las filas de Tennis-Data 2016+ cruzan (lo que no cruza del lado
  Tennismylife es sobre todo Copa Davis y niveles que Tennis-Data no
  cubre). 39.442 partidos con cuota media, 2010–2026.

## Fuga de información en los datos ATP (encontrada y corregida)

Al analizar ATP apareció una fuga: Tennismylife fecha todos los partidos
de un torneo con el mismo día, y `_clean()` en `src/data/loader.py`
ordenaba solo por fecha con un sort inestable, lo que listaba rondas
posteriores antes que las anteriores en el **99,1%** de los torneos ATP
2016–2024. Las features de un partido de primera ronda se calculaban
después de los partidos siguientes de ese jugador en el mismo torneo:
`rest_diff` era negativo para el 50,4% de los ganadores frente a un 11,3%
positivo, y por sí solo "batía" al mercado de cierre en 14,5 milésimas de
log-loss. Copa Davis comparte la misma fuente.

Corregido en el commit `4c0e919` (orden por fecha, torneo, ronda y
`match_num`); las inversiones bajan al 0,5%. Tras reconstruir:

| | Antes (con fuga) | Después |
|---|---|---|
| ATP walk-forward, accuracy | 68,49% | 66,73% |
| ATP walk-forward, log-loss | 0,5953 | 0,6037 |
| Davis, accuracy / log-loss | 68,89% / 0,5959 | 68,84% / 0,5946 |
| Ganancia de `rest_diff` sobre el mercado | 14,5 mll | 0,0 mll |

Todas las cifras ATP de este informe son **posteriores** a la corrección.
Una primera versión del análisis ATP, con la fuga, sugería que el modelo
aportaba algo al mercado y que los favoritos estaban en equilibrio; ambas
cosas eran artefactos de la fuga.

## Resultados

**1. El mercado es más preciso que el modelo, y el modelo no le añade nada.**

| | WTA log-loss | ATP log-loss |
|---|---|---|
| Modelo | 0,6221 | 0,6085 |
| Mercado (media, sin margen) | **0,5951** | **0,5789** |
| Pinnacle vs modelo (subconjunto) | 0,5971 vs 0,6240 | 0,5770 vs 0,6082 |
| Peso óptimo del modelo en `a·modelo + (1−a)·mercado` | **0** | **0** |

**2. La regla de producción pierde dinero, y pierde más que apostar al azar.**

| Tour, precio | N | p modelo | q mercado | Victorias reales | EV teórico | ROI plano (IC 95%) | ROI Kelly |
|---|---|---|---|---|---|---|---|
| WTA, media | 9.038 | 0,474 | 0,392 | **0,376** | +19,2% | **−12,3%** (−15,0..−9,5) | −9,2% |
| WTA, Bet365 | 8.848 | 0,467 | 0,386 | 0,371 | +20,2% | −11,4% (−14,3..−8,6) | −8,7% |
| WTA, mejor cuota | 11.440 | 0,470 | 0,412 | 0,403 | +21,6% | −3,9% (−6,5..−1,2) | −1,2% |
| ATP, media | 13.902 | 0,445 | 0,363 | **0,354** | +23,6% | **−10,6%** (−13,1..−8,0) | −7,8% |
| ATP, Bet365 | 13.763 | 0,433 | 0,355 | 0,346 | +26,3% | −9,1% (−11,9..−6,3) | −6,8% |
| ATP, mejor cuota | 17.517 | 0,446 | 0,390 | 0,382 | +30,6% | +1,1% (−2,0..+4,1) | +0,4% |

Apostar a **todos** los lados a cuota media da −7,2% (WTA) y −7,1% (ATP),
que es simplemente el margen de la casa. La selección del modelo es **peor
que no seleccionar** en ambos circuitos: sus "edges" señalan justo los
partidos donde se equivoca. Ni siquiera a la mejor cuota del mercado (el
máximo entre todas las casas, un techo que producción no siempre alcanza)
hay un resultado positivo distinguible de cero.

El EV teórico del backtest (+19–24% a cuota media) coincide con el del log
real (+19,3%). No es mala suerte de la muestra real: es lo que el modelo
produce de forma estructural.

**3. La tasa real sigue al mercado, no al modelo, en todos los tramos.**

- Por cuota: en todas las bandas, las victorias reales quedan en o por
  debajo de la q del mercado, nunca cerca de la p del modelo. Con cuotas
  > 5: WTA modelo 20,6% / mercado 13,7% / real 9,7% (ROI −36%); ATP
  18,5% / 11,8% / 9,3% (ROI −28%).
- Por edge: **cuanto más edge declara el modelo, peor el ROI** (WTA:
  −10,2% → −13,1% → −14,0%; ATP: −9,0% → −9,9% → −12,7% para 3–5%, 5–7%,
  7–10%). Un edge real se comportaría al revés.
- Por año: WTA negativo en los 11 años; ATP negativo en 16 de 17 (2021:
  +1,3%).
- Ningún filtro probado rescata un ROI positivo a cuota media (WTA cuotas
  < 2: −6,3%; p ≥ 0,6: −5,8%; ATP cuotas ≤ 3: −6,1%, > 3: −17,0%).

## Modelo anclado al mercado (`--anchored`)

En lugar de predecir desde cero, se parte de la probabilidad del mercado
q y se aprende solo la desviación. Walk-forward sobre los años con cuotas
(mínimo 3 de entrenamiento), sin intercepto y simétrico entre jugadores:

- **M1**: logit(q). El mercado solo, recalibrado.
- **M2**: logit(q) + logit(p del modelo actual, fuera de muestra).
- **M3**: logit(q) + las 15 features (L2, C=0,1).

| | WTA 2019–2026 | ATP 2013–2026 |
|---|---|---|
| Log-loss mercado | 0,5907 | 0,5833 |
| Log-loss M1 / M2 / M3 | 0,5901 / 0,5902 / 0,5903 | 0,5831 / 0,5829 / 0,5830 |
| Apuestas con la regla a cuota media, M1 / M2 / M3 | 2 / 6 / 190 | 5 / 6 / 387 |
| ROI M3 a cuota media | −17,0% (−30,6..−2,5) | −4,6% (−13,8..+4,7) |

Ninguna feature, sola, mejora al mercado en más de 0,3 milésimas de
log-loss (ATP, tras la corrección). Los modelos anclados son prácticamente
el mercado: a cuota media casi nunca ven un 3% de edge, que es lo que
haría un modelo honesto sin información adicional.

A la mejor cuota, ATP M1 da +8,2% (N=1.054, IC +2,1..+14,7) y M2/M3 no lo
mejoran (+8,0%, +5,6%). M1 no usa ninguna feature: esa ganancia sale de
comparar el precio máximo del mercado con su probabilidad media (buscar la
mejor cuota), no del modelo, y la cuota máxima histórica incluye casas y
precios a los que producción no tiene acceso. En WTA ni eso es
distinguible de cero.

## El log real

| Tour | N | p modelo | q mercado | Victorias reales | ROI plano |
|---|---|---|---|---|---|
| ATP | 83 | 0,505 | 0,435 | 0,482 | +1,6% |
| WTA | 33 | 0,542 | 0,473 | 0,485 | −25,7% |

La WTA real cuadra con el backtest. La ATP real (+1,6%, N=83) queda dentro
del ruido; el backtest con N=13.902 manda.

## Conclusiones

1. La investigación del 2026-09-18 era correcta en lo que medía (la
   calibración global es buena) pero no medía lo que importa para apostar.
   El gap de −24 pp **no es solo varianza**: es la sobreestimación
   sistemática del edge frente a un mercado más informado.
2. **Ni WTA ni ATP tienen ventaja**: el mercado domina, el modelo no le
   añade información (peso 0 en la mezcla) y la selección es peor que el
   azar en ambos. **Davis Cup** no se puede contrastar con el mercado, y el
   modelo ATP predice sus partidos mejor que su propio modelo (ver sección
   Davis). Los tres están excluidos del auto-registro desde 2026-10-07
   (`AUTO_LOG_EXCLUDED_TOURS = {"wta", "atp", "davis"}` en
   `scripts/daily_workflow.py`); se siguen escaneando y auditando.
3. **Anclar al mercado no crea edge** con las features actuales: el modelo
   resultante es el propio mercado. Lo único positivo (a la mejor cuota)
   viene de buscar precio, no de predecir mejor.
4. El EV teórico del scanner y el workflow (+19–30% de media en las
   apuestas elegidas) no es una estimación creíble del retorno esperado y
   no debería usarse para dimensionar stakes. Tocar `_SHRINK_RATE` o
   `MIN_EDGE` no lo arregla.

## Qué haría falta para volver a apostar

- **Información que el mercado no tenga**: las features actuales (Elo,
  ranking, forma, H2H, descanso) ya están en el precio. Haría falta otra
  fuente (estadísticas de saque/resto punto a punto, lesiones, etc.) y
  validarla con `--anchored`: una feature solo vale si mejora el log-loss
  de M1 de forma consistente fuera de muestra.
- **Davis Cup**: si algún día se reactiva, predecir sus partidos con el
  modelo ATP en vez del específico (ver sección Davis) y validarlo antes
  contra cuotas reales.

## Copa Davis

Sin cuotas históricas no hay contraste con el mercado; la revisión se
limitó a la calidad del modelo específico (`davis_features.csv`, 7.496
partidos fuera de muestra, 2000–2026).

- **Sin fuga.** Los partidos de una eliminatoria están en orden
  cronológico por `match_num`, y las métricas con el orden barajado (antes
  de la corrección del loader) y con el orden correcto son prácticamente
  iguales (68,89% vs 68,84%).
- **La ventaja de 7 pp del LR sobre el Elo se debe a un Elo débil.** El
  pipeline Davis construye su Elo solo con partidos de Copa Davis: el
  jugador mediano tiene 4, y el 53% tiene menos de 5. El Elo solo acierta
  el 61,2%; quien sostiene el modelo es el ranking ATP.
- **`rust_factor_diff` es un indicador de debutante, no de óxido.**
  `rust_factor` da 1,0 (el máximo) a quien no tiene historial, así que en un
  historial solo-Davis señala a quien juega su primera eliminatoria, y los
  debutantes pierden más (el ganador es el "más fresco" en el 9,7% de los
  partidos y el menos en el 15,7%). Es información conocida antes del
  partido.
- **Calibración** dentro de ±2 pp en todos los tramos.
- **El modelo ATP es mejor en los mismos partidos.** Los 7.496 partidos
  Davis también están en el backtest ATP (que incluye las eliminatorias y
  todo el circuito):

| Modelo | Accuracy | Log-loss |
|---|---|---|
| Davis específico | 68,9% | 0,5864 |
| **ATP** | 68,5% | **0,5798** |
| Elo solo Davis | 61,2% | 0,6661 |

La diferencia crece en los años recientes (2013–2026: 0,5871 frente a
0,5974). Como el modelo ATP no tiene ventaja frente al mercado del circuito
y es el mejor que hay para Davis, no hay base para apostar Davis: excluida
del auto-registro desde 2026-10-07. Nunca se había registrado una apuesta
Davis.

# Cerrando gaps de cobertura del scanner — 2026-09-18 (sesión 4)

Checkpoint: producción de picks reales con email/push + cierre de
exclusiones del scanner. Como en las sesiones anteriores del mismo día,
varias premisas del plan no se sostenían al verificarlas — documentado
abajo por fase.

## Fase 1 — Staleness gate en `daily_workflow.py`

A diferencia del intento anterior (declinado — ver
`2026-09-18-production-readiness.md`), esta vez el pedido era distinto y
sí tenía sentido: no cambiar los *umbrales* de `live_tournament_mode`
(que ya son más estrictos, no más laxos), sino cambiar **qué hace
`daily_workflow.py` con el resultado** — que deje de excluir un tour por
completo y en su lugar solo advierta, igual que `daily_scanner.py`, pero
**solo cuando se pide explícitamente** vía la nueva flag
`--live-tournament`.

- `_is_stale(tour, live_tournament_mode=False)` nueva función genérica en
  `scripts/daily_workflow.py`. `_wta_is_stale`/`_atp_is_stale` pasan a
  ser wrappers; se agregó `_davis_is_stale` (usa el mismo mecanismo
  genérico, lee `data/model_cache/davis.pkl` vía `get_last_match_date`,
  sin fuente separada).
- Default (`live_tournament_mode=False`, sin la flag): **comportamiento
  original sin cambios** — cualquier nivel != OK excluye el tour. Esto
  es deliberado: el gate existe específicamente para que un pick
  auto-pusheado y auto-emailado no se genere con datos de semanas de
  antigüedad sin que nadie lo revise (ver
  `docs/superpowers/specs/2026-07-13-staleness-context-aware-design.md`).
- Con `--live-tournament`: se imprime la advertencia detallada
  (`report.message`) pero el tour ya NO se excluye. Verificado en vivo:
  sin la flag, ATP/WTA se excluyen ("se ignoran picks..."); con la flag,
  ambos pasan a "Descubriendo partidos programados..." con la advertencia
  impresa arriba.
- `--dry-run` sigue funcionando sin cambios (verificado).
- Email/notifier: ya estaba cubierto end-to-end con mock
  (`tests/test_notifier.py::TestSendPicksEmail`, SMTP mockeado) y a nivel
  workflow (`test_sends_email_with_selected_picks_and_vpn_status`) — no
  hizo falta agregar nada nuevo.
- 6 tests nuevos en `tests/test_daily_workflow.py`.

## Fase 2 — Surface resolver

El feed en vivo de Odds API solo tenía Guadalajara (ya cubierto en la
sesión anterior) — cero gaps nuevos *reactivos*. Pero dado que el
objetivo es producción real, se agregó cobertura *proactiva* para el
swing asiático de cancha dura que arranca la semana que viene
(23-29 sept): Chengdu, Hangzhou, Tokyo/Japan Open, Beijing/China Open,
Wuhan — todos cancha dura, verificados vía el PDF oficial del calendario
ATP 2026 + Wikipedia antes de agregarlos (mismo rigor que Guadalajara).
7 tests nuevos, incluyendo uno que verifica que *toda* entrada existente
en `TOURNAMENT_SURFACES` resuelve correctamente (guarda de regresión).

## Fase 3 — Player matcher: investigado, casi nada que arreglar

De los 4 casos propuestos, verificados contra el código real antes de
tocar nada:

- **Nombres con acentos** ("Cristian Garín"): ya funcionaba —
  `normalize_name()` (`src/name_matching.py`) ya hace NFKD-strip de
  acentos. Ya había cobertura de esto en el test suite
  (`test_accented_name_variant`).
- **Surname + inicial** ("N. Djokovic"): ya funcionaba, mismo mecanismo
  que "A. Zverev" ya testeado.
- **Nombres de Davis Cup**: premisa falsa — Davis Cup usa el mismo feed
  ATP (`tourney_level == "D"`), mismo schema `winner_name`/`loser_name`,
  mismo formato "Nombre Apellido". Verificado contra datos reales
  cargados (`load_davis_cup_matches`). No hay convención de nombres
  separada que manejar.
- **Apodos WTA** ("Iga" → "Iga Swiatek"): **deliberadamente NO
  implementado**. The Odds API nunca envía apodos sueltos en la práctica
  (usa formatos estructurados "Nombre Apellido" / "Apellido I.", igual
  que todos los demás casos de este archivo) — y resolver por
  primer-nombre-solo significaría adivinar entre cualquier jugador que
  comparta ese nombre, exactamente el tipo de riesgo de "adivinar mal"
  que el propio docstring de `match_player_name` dice que es peor que no
  matchear. Documentado en el test en vez de construir algo especulativo.

4 tests nuevos, caracterizando el comportamiento real (2 ya pasaban sin
cambios, 1 confirma que Davis Cup no necesita nada especial, 1 documenta
por qué el caso "Iga" no se implementa).

## Fase 4 — Ejecución end-to-end

`python scripts/run_prediction.py --tour wta --days-ahead 1` (con
descarga real, sin `--no-download`): los 4 pasos corrieron limpio
(descarga, pipeline, reentrenamiento, scanner con `--auto-save`
incorporado). **0 picks** — mismo motivo que la sesión anterior, no un
bug: el único evento con cuotas en Guadalajara (Samsonova vs Stearns)
ya había arrancado, y no había cuotas nuevas publicadas para el próximo
partido al momento de la corrida (~20:37 hora Lima).

**Hallazgo importante para el checklist de Fase 4**: `run_prediction.py`
invoca `src.daily_scanner`, no `scripts.daily_workflow` — y
`daily_scanner.py` **no envía correo ni hace push** (eso es
exclusivamente de `daily_workflow.py`, confirmado en la sesión
anterior). El ítem del checklist "el email/notifier funciona" no se
puede ejercitar end-to-end vía `run_prediction.py` tal como está armado
hoy; se verificó en su lugar con el mock existente
(`tests/test_notifier.py`) y contra el `.env` real (sin credenciales
SMTP configuradas, degrada correctamente con un aviso claro, no
crashea). Si se quiere que `run_prediction.py` también notifique por
correo, haría falta un cambio de diseño explícito (¿debería invocar
`daily_workflow.py` en vez de/además de `daily_scanner.py`?) — no
asumido silenciosamente en esta sesión.

## Verificación final

- **599/599 tests** (582 al empezar esta sesión: +6 daily_workflow,
  +7 surface_resolver, +4 player_matcher = 599).
- `git status` limpio, sin tocar `.env` ni credenciales.
- `data/value_bets_log.csv` sin cambios — no se generó ningún pick real
  por el mismo motivo de timing explicado arriba, no por un bloqueo de
  staleness (que ya no bloquea con `--live-tournament`).

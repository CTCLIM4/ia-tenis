# Producción real del scanner diario — 2026-09-18

Checkpoint: llevar `daily_scanner.py` de `--dry-run` a ejecución real.
Cuatro "fixes" propuestos, investigados antes de tocar código — dos eran
reales, uno no (el gate de staleness funciona como está diseñado), y
apareció un quinto problema real no listado originalmente (superficie de
Guadalajara sin resolver).

## Fix #1 — Gate de staleness: investigado, NO es un bug, sin cambios

La hipótesis era que `lima_today()` (timezone) o `live_tournament_mode`
bloqueaban falsamente datos recientes. Verificado directamente:

- `lima_today()` devuelve `2026-09-18`, correcto — coincide con la fecha
  del sistema. Una diferencia de timezone América/Lima vs UTC es como
  mucho ±1 día, no puede explicar un gap de 5-6 días.
- `live_tournament_mode=True` es **más estricto** (WARNING >1d, CRITICAL
  >2d), no más laxo — lo contrario de la hipótesis.
- El test pedido (`yesterday + live_tournament_mode=True` → `OK`) **ya
  pasaba** antes de tocar nada — verificado por ejecución directa.
- Re-descargué WTA: `2026w.xlsx` salió byte-idéntico al de la mañana —
  tennis-data.co.uk genuinamente no ha publicado resultados nuevos en
  6 días, pese a haber un torneo en vivo (Guadalajara Open). Es latencia
  real de la fuente, no un bug de nuestro código.

**Hallazgo clave que sí cambia el plan**: `src/daily_scanner.py` **no
tiene ningún gate de staleness** — esa lógica vive solo en
`scripts/daily_workflow.py` (`_atp_is_stale`/`_wta_is_stale`). El "0
picks" original vino de `daily_workflow.py --dry-run`, no de
`daily_scanner.py`. Corriendo `daily_scanner.py` directo, la staleness
del dataset solo imprime una advertencia, nunca bloquea.

## Fix #2 — `end_year=None` en los 3 loaders (real, arreglado)

`load_atp_matches`/`load_wta_matches`/`load_davis_cup_matches` tenían
defaults hardcodeados (2024/2024/2026) que quedaban obsoletos cada año.
Ahora `end_year: Optional[int] = None` resuelve a `date.today().year`
dentro de cada función. 3 tests nuevos
(`test_{atp,wta,davis}_loader_uses_current_year_default`).

## Fix #3 — `vpn_tracker.py` no detectaba TunnelBear por WireGuard (real, arreglado)

TunnelBear v4+ puede túnelizar sobre WireGuard, que aparece como un
adaptador genérico "Wintun"/"WireGuard Tunnel", no "TunnelBear Adapter
V9" — el patrón de nombre por defecto nunca lo encontraba.
`_query_adapter_bytes` ahora reintenta con nombres de adaptador
genéricos (`Wintun`, `WireGuard`) **solo si** `_is_process_running`
confirma que `TunnelBear.exe` está corriendo — evita atribuir tráfico de
una VPN WireGuard no relacionada. 2 tests nuevos. Verificado contra la
máquina real: antes `available: False` siempre; ahora
`available: True, session_mb: 612.98` (uso real de la sesión de
descargas de este día).

## Fix #5 (no estaba en el plan original) — superficie de Guadalajara sin resolver

Al correr el scanner real por primera vez aparecería un problema
distinto: `resolve_surface` no reconocía "WTA Guadalajara Open"
(torneo relativamente nuevo, WTA 500 desde 2022) y excluía el único
partido en vivo del escaneo. Verificado con dos fuentes (wtatennis.com
+ Wikipedia, 2026-09-18): cancha dura exterior, Centro Panamericano,
Zapopan. Agregado `"guadalajara": "hard"` a `TOURNAMENT_SURFACES`
(`src/surface_resolver.py`), test-first.

## Fase B — tests de `daily_scanner.py` (6 nuevos, 1 ya cubierto)

`test_scanner_produces_picks_with_stale_data`,
`test_missing_api_key_fails_gracefully_no_crash`,
`test_scanner_handles_no_matches_found`,
`test_scanner_match_player_name_resolved` (via surname+inicial, no
string exacto), `test_scanner_with_davis_tour` — todos pasan
inmediatamente porque son tests de caracterización de comportamiento ya
correcto (no había bug que arreglar en `daily_scanner.py` mismo).
`test_scanner_filters_out_low_value_bets` no se agregó por separado:
`EvaluatedMatch.is_value_bet` delega directo en `calculate_value`'s
`has_value` (ya cubierto por `TestEvaluateMatches.test_no_value_when_edge_not_positive`
y toda la suite de `TestCalculateValue`).

## Intento de picks reales

Con la superficie resuelta, `python -m src.daily_scanner --tour wta
--days-ahead 1` (y luego `--tour both --days-ahead 2`) llegaron hasta el
descubrimiento de partidos real vía Odds API, pero **0 picks** — no por
ningún bug, sino porque el único partido del Guadalajara Open con cuotas
(Samsonova vs Stearns, `commence_time` 2026-09-19T00:53:37Z) ya había
arrancado 14 minutos antes de la consulta (ahora 01:07:57Z UTC).
`daily_scanner.py` está diseñado para cuotas pre-partido, no in-play —
exclusión correcta, no una falla. Toda la infraestructura (modelos
reentrenados, resolución de superficie, resolución de nombres, ausencia
de gate de staleness) quedó verificada end-to-end; solo faltó el timing
de una corrida antes de que arrancara el partido del día.

## Verificación final

- **580/580 tests** (569 al empezar esta tarea: +3 end_year, +2
  vpn_tracker, +5 daily_scanner, +1 surface_resolver = 580).
- 3 pipelines (atp/wta/davis) y 3 modelos reentrenados de forma no
  interactiva (`load_model(tour, retrain=True)` — deliberadamente NO
  usé `python -m src.value_analysis --retrain`, que cae en un REPL
  interactivo y se cuelga sin stdin).
- Snapshots `production-baseline` y `production-ready` creados,
  integridad verificada (`verify_snapshot_integrity` → `True` en ambos).
- `data/value_bets_log.csv` sin cambios — no se generó ningún pick real
  esta sesión (por el timing explicado arriba, no por un bloqueo).

# Dedup de value_bets_log.csv — 2026-09-19

Checkpoint: cerrar el gap encontrado en la sesión anterior — `daily_scanner.py`
no tenía protección contra volver a loguear el mismo partido pendiente en
corridas separadas, lo que hubiera duplicado la posición en el P&L una vez
liquidado.

## Corrección al brief inicial

El brief atribuía `log_query()` a `src/calibration_audit.py`. En realidad
vive en `src/value_analysis.py` (`log_prediction_audit()` sí está en
`calibration_audit.py`). Se implementó ahí, en el módulo que ya escribe
`value_bets_log.csv`.

## Fase 1 — `check_existing_log_entry()` / `update_log_entry()`

Identidad de un match: `tour + tournament + player_a + player_b +
match_date` — las cuotas nunca deciden si es "el mismo partido", solo si
hace falta actualizar. `LogMatchStatus`: `NEW` / `DUPLICATE` (cuotas
dentro del 2%) / `UPDATE` (cuotas se movieron >2%, en `odds_a` **o**
`odds_b`) / `RESOLVED` (la fila existente ya tiene un resultado real,
nunca se toca). `log_query()` se refactorizó para compartir
`_build_log_row()` con la nueva `update_log_entry()`, que reescribe el
archivo completo reemplazando la fila que matchea (mismo tradeoff que
`_migrate_log_header_if_needed()` — el archivo es chico, solo picks
reales). 12 tests nuevos.

## Fase 2 — integración en `daily_scanner.py`

Antes de `log_query()`, se llama `check_existing_log_entry()` y se
bifurca: `DUPLICATE`/`RESOLVED` → aviso impreso, no se loguea nada;
`UPDATE` → `update_log_entry()`; nuevo → `log_query()` como antes. Los 5
tests existentes que mockeaban `log_query()` directamente quedaban
leyendo el `LOG_PATH` real sin querer (funcionaban porque sus fixtures
sintéticas nunca coincidían con nada del archivo real, pero no estaban
aislados) — se corrigieron para redirigir `LOG_PATH` a un `tmp_path`
explícitamente. 4 tests nuevos de dedup + 5 corregidos.

## Fase 3 — verificación end-to-end real

Simulación con una copia real de `value_bets_log.csv` (120 filas, dos
filas preexistentes de Stearns/Jovic — una del incidente original antes
de que existiera esta protección) y una corrida real de `run_scan()`
contra la API en vivo: el partido seguía con cuotas (3.90), el mecanismo
detectó el `UPDATE` correctamente (`"Actualizado en ... (odds
cambiaron)"`), reemplazó la primera fila existente in-place, y el
archivo terminó con la **misma cantidad de filas** (120) — no se agregó
una tercera. La segunda fila preexistente (artefacto de antes del fix)
se dejó intacta, sin tocar histórico según la regla del brief.

Nota aparte: la primera versión de esta verificación manual usó una ruta
`/tmp/...` que Python nativo de Windows resolvió como `D:\tmp\...`
(distinto de lo que Git Bash resuelve para `/tmp`) — un bug de mi propio
script de verificación, no del código bajo prueba. Los tests de pytest
ya usaban `tmp_path` correctamente y nunca tuvieron este problema.

## Verificación final

692/692 tests (era 676 al empezar esta sesión). `scripts/backup_logs.py`
sigue funcionando con el nuevo flujo. 3 entradas reales nuevas en
`prediction_audit_log.csv` (de las corridas de verificación de esta
sesión, legítimas — el audit log registra toda evaluación, no solo las
guardadas).

# Protección de logs + investigación del incidente — 2026-09-19

Checkpoint: proteger `data/prediction_audit_log.csv` en git y agregar un
mecanismo de backup, después de que una limpieza de duplicados propia
borrara 26 filas del archivo (que en ese momento no estaba trackeado).

## El incidente, corregido

Al reportarlo inicialmente dije que había "perdido 26 filas históricas
sin forma de recuperarlas". Eso fue más alarmista de lo que la situación
realmente ameritaba. El script de deduplicación que corrí usaba como
clave el contenido completo de la fila **excepto el timestamp**
(`key = tuple(row[1:])`) y solo descartaba una fila si esa clave **ya
había aparecido antes** en el archivo. Por construcción, eso garantiza
que **cada una de las 26 filas borradas tiene una gemela idéntica
(mismo contenido, timestamp anterior) que sigue en el archivo hoy** —
no se perdió ningún dato único de predicción. Lo único que se perdió es
el registro de que el bug de doble-logueo disparó exactamente en esos
26 timestamps adicionales (información forense sobre el bug, no datos
de predicción/apuestas).

Confirmado con una muestra de 5 de las 26 (Krueger A., Brandon
Nakashima, Ugo Humbert, Swiatek I., Stearns P.): todas tienen múltiples
filas sobrevivientes con el mismo jugador/tour en fechas/timestamps
cercanos, consistente con la garantía del algoritmo.

## Por qué la Fase 3 (reconstrucción desde value_bets_log.csv) no aplica

La Fase 3 original asumía que había filas realmente perdidas que
reconstruir. Como no las hay (ver arriba), no hace falta el script de
reconstrucción — no hay ningún hueco que llenar. Se deja documentado acá
en vez de construir algo que resuelve un problema inexistente.

## Causa raíz del doble-logueo, todavía sin explicar del todo

El patrón (mismo timestamp exacto para varios partidos evaluados juntos,
y un segundo cluster ~1 minuto después con un subconjunto distinto de
partidos) sugiere que `run_scan()` se ejecutó dos veces seguidas en
varias jornadas distintas desde el 28 de julio, no una sola vez. No se
encontró una causa concluyente: no hay tarea programada registrada
(`schtasks`, verificado), no quedaron procesos python colgados después
del incidente de hoy. Sigue abierto — si vuelve a pasar, la protección
de Fase 1-2 (git tracking + backup automático) ahora limita el daño a
"hay una fila duplicada para arreglar", no "se perdió información sin
poder recuperarla".

## Fase 1 — git tracking

`data/prediction_audit_log.csv` ahora trackeado (igual que
`value_bets_log.csv` ya lo estaba): `.gitignore` con
`data/prediction_audit_log.csv` seguido de
`!data/prediction_audit_log.csv`. 2 tests nuevos
(`tests/test_gitignore_protection.py`) que llaman `git ls-files` de
verdad contra el repo — no tiene sentido mockearlo, la propiedad bajo
test es específica de este repo.

## Fase 2 — backup automático

`scripts/backup_logs.py`: `backup_file()`/`backup_logs()`, timestamp
inyectable (`now=`) para tests deterministas, rotación a los últimos 7
backups por archivo, `shutil.copy2` para preservar metadata. Integrado
en `src/daily_scanner.py`: `backup_logs()` se llama una vez antes del
loop que escribe `log_query()`, solo si hay value bets que guardar (no
dispara backups en escaneos vacíos). 15 tests nuevos.

## Verificación

676/676 tests pasando (663 antes de esta sesión + 2 gitignore + 9
backup_logs + 2 daily_scanner integration).

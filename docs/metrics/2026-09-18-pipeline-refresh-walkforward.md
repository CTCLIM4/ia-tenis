# Refresco de datos + walk-forward — 2026-09-18

Checkpoint Día 1-3 del plan de descarga/procesamiento. Datos ATP/WTA
estaban desactualizados desde el 2026-09-03 (última corrida de
`daily_workflow.py`/`settle_workflow.py`); esta sesión los refrescó y
corrió el pipeline completo (`src/pipeline.py`) sobre los datos en vivo.

## Día 1-2 — Descarga

- **ATP** (`stats.tennismylife.org`): sin cambios de código, funcionó
  igual que siempre. `ongoing_tourneys.csv` mezclado en `2026.csv`
  (2258 filas).
- **WTA** (`tennis-data.co.uk`): **bloqueado por un cambio real del
  sitio**, no por VPN. El sitio movió sus archivos bajo un prefijo de
  ruta opaco nuevo (`.../hrjk-85HytOjkhth76j_ygh4jf7/{year}w/{year}.xlsx`),
  así que los 6 patrones de URL hardcodeados en
  `scripts/download_data.py` fallaban con 404 para los 20 años. Fix:
  `_fetch_wta_links()` ahora scrapea `data.php` en vivo por el href real
  en vez de adivinar el patrón — sobrevive el próximo cambio de
  estructura del sitio. Ver commit `c1cd723`.
  - 19/20 años se re-verificaron o re-descargaron sin problema.
  - 2020 falló en la primera corrida completa (probable rate-limit
    transitorio de Cloudflare tras ~13 requests seguidos), pero al
    reintentarlo individualmente confirmó que ya estaba al día
    (168,741 bytes, sin pérdida de datos).
  - `2026w.xlsx` pasó de 306,902 a 325,424 bytes (partidos nuevos).

## Día 3 — Pipeline y walk-forward backtest

Comandos:
```
python -m src.pipeline atp 1990 --eval-start 2020
python -m src.pipeline wta 2007 --eval-start 2020
```

### ATP (117,364 partidos, 1990-2026, 37 temporadas)

```
  Year |     Acc |  LogLoss |   Brier |  EloAcc |    EloLL |      N
-------------------------------------------------------------------
  2020 |  0.6542 |   0.6131 |  0.2133 |  0.6323 |   0.6337 |   1466
  2021 |  0.6545 |   0.6125 |  0.2128 |  0.6453 |   0.6309 |   2735
  2022 |  0.6607 |   0.6088 |  0.2108 |  0.6443 |   0.6285 |   2918
  2023 |  0.6391 |   0.6274 |  0.2192 |  0.6301 |   0.6382 |   2995
  2024 |  0.6480 |   0.6204 |  0.2158 |  0.6309 |   0.6322 |   3159
  2025 |  0.6421 |   0.6266 |  0.2186 |  0.6264 |   0.6371 |   2861
  2026 |  0.6475 |   0.6178 |  0.2147 |  0.6422 |   0.6366 |   2258

  Mean Accuracy : 0.6494  (Elo-only: 0.6359)
  Mean Log-Loss : 0.6181  (Elo-only: 0.6339)
```

### WTA (47,483 partidos, 2007-2026, 20 temporadas)

```
  Year |     Acc |  LogLoss |   Brier |  EloAcc |    EloLL |      N
-------------------------------------------------------------------
  2020 |  0.6427 |   0.6281 |  0.2190 |  0.6417 |   0.6368 |   1055
  2021 |  0.6633 |   0.6104 |  0.2116 |  0.6465 |   0.6318 |   2447
  2022 |  0.6471 |   0.6264 |  0.2188 |  0.6273 |   0.6409 |   2369
  2023 |  0.6536 |   0.6179 |  0.2149 |  0.6332 |   0.6362 |   2500
  2024 |  0.6459 |   0.6163 |  0.2144 |  0.6372 |   0.6342 |   2522
  2025 |  0.6481 |   0.6188 |  0.2148 |  0.6384 |   0.6357 |   2464
  2026 |  0.6593 |   0.6010 |  0.2082 |  0.6482 |   0.6196 |   2075

  Mean Accuracy : 0.6514  (Elo-only: 0.6389)
  Mean Log-Loss : 0.6170  (Elo-only: 0.6336)
```

## Lectura

Ambos tours se mantienen en el rango histórico esperado (~64-66%
accuracy, LR consistentemente por encima del baseline Elo-solo en
accuracy y log-loss todos los años 2020-2026, incluyendo 2026 con datos
parciales de temporada). No hay señal de degradación ni de lookahead
tras el refresco de datos — resultados en línea con corridas previas
documentadas en este directorio. `data/processed/{atp,wta}_features.csv`
quedaron regenerados con datos hasta 2026-09-18.

## Nota aparte (no bloqueante)

`load_atp_matches()`/`load_wta_matches()` en `src/data/loader.py` tienen
default `end_year=2024` hardcodeado (y `scripts/export_figures.py` llama
`load_atp_matches(1990, 2023)` hardcodeado). Ninguno de los dos afecta el
pipeline real: `src/pipeline.py` y `src/value_analysis.py` siempre pasan
`end_year` resuelto dinámicamente a `date.today().year` (fix previo,
commit `6f18e15`). Es un defecto latente solo para quien llame al loader
directamente sin argumentos — no se tocó en este checkpoint por ser
tangencial al plan del día.

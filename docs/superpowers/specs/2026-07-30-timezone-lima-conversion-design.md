# Conversión explícita de timestamps de la API a America/Lima — Diseño

**Fecha:** 2026-07-30
**Módulos:** `src/daily_scanner.py`, `src/data/staleness.py`, `src/value_analysis.py`
**Estado:** Aprobado, pendiente de implementación

## Objetivo

The Odds API reporta `commence_time` en UTC (ISO8601, sufijo `Z`). Hoy
`src/daily_scanner.py` toma `.date()` directamente sobre ese datetime UTC
sin convertir, así que un partido a las `02:00Z` queda registrado como si
fuera un día después del que realmente es en Lima (UTC-5) — por ejemplo
`2026-07-27T02:00:00Z` es en realidad `2026-07-26 21:00` en Lima, pero hoy
se guarda como `match_date = 2026-07-27`.

Ese `match_date` mal calculado se propaga a: las features de
rest/fatigue/workload (`src/features/engineering.py`, `decay.py`, que
reciben `match_date` como parámetro), y las columnas `match_date` de
`data/value_bets_log.csv` y `data/prediction_audit_log.csv`. También afecta
la coherencia del chequeo de staleness (`src/data/staleness.py`), cuyo
`reference_date` por defecto hoy es `date.today()` — hora local del
sistema, ambigua, no necesariamente Lima — lo que puede desalinearse contra
un `match_date` que sí está correctamente en Lima una vez aplicado este fix.

**Fuera de alcance (decidido explícitamente):** la columna `timestamp` de
ambos logs (cuándo se registró el pick, vía `datetime.now()`) no viene de
la API — queda sin tocar.

## Arquitectura

Módulo nuevo y pequeño, mismo patrón que `surface_resolver.py` /
`player_matcher.py`: `src/data/timezone_utils.py`.

```python
from datetime import date, datetime
from zoneinfo import ZoneInfo

LIMA_TZ = ZoneInfo("America/Lima")

def to_lima(dt: datetime) -> datetime:
    """Convert an aware datetime to America/Lima. Raises on naive input —
    a naive datetime's origin tz is ambiguous and must never be guessed."""
    if dt.tzinfo is None:
        raise ValueError("to_lima() requires a timezone-aware datetime")
    return dt.astimezone(LIMA_TZ)

def lima_today() -> date:
    return datetime.now(LIMA_TZ).date()
```

`to_lima` rechaza datetimes naive en vez de asumir su origen (mismo
principio de "nunca adivinar" que ya sigue `player_matcher` con nombres no
emparejables y `surface_resolver` con superficies no resolubles: se
excluye/falla explícito en vez de inferir en silencio).

Dependencia nueva: `tzdata` en `requirements.txt`. `zoneinfo` es stdlib
(Python 3.9+), pero en Windows y en imágenes mínimas de Linux no hay base
de datos IANA de zonas horarias instalada en el sistema — el paquete
`tzdata` la provee. Ya está presente en esta máquina, pero debe quedar
declarado como dependencia para que el entorno sea reproducible en otra
máquina/CI.

## Cambios por archivo

### `src/daily_scanner.py`

Línea 182, dentro de `discover_matches`:

```python
# antes
match_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).date()
# después
match_date = to_lima(datetime.fromisoformat(commence_time.replace("Z", "+00:00"))).date()
```

`_now()` y `_is_within_window` **no se tocan**: comparan datetimes aware
como instantes en el tiempo (`now <= ts <= now + timedelta(...)`), y esa
comparación es correcta sin importar en qué zona horaria estén expresados
los operandos — Python normaliza internamente. Se deja un comentario en el
código explicando por qué, para que no se lea como un descuido.

### `src/data/staleness.py`

`check_dataset_staleness`: el default `reference_date = date.today()` pasa
a `reference_date = lima_today()`.

### `src/value_analysis.py`

`_check_staleness` (línea ~341): `reference_date or date.today()` pasa a
`reference_date or lima_today()`.

## Tests

- **`tests/test_timezone_utils.py`** (nuevo): `to_lima` con datetime naive
  → `ValueError`; con datetime aware UTC → offset resultante `-05:00`;
  Lima no observa horario de verano, así que el offset es constante en
  cualquier época del año (verificar con al menos dos fechas, una en cada
  mitad del año, para dejarlo explícito en el test).
- **`tests/test_daily_scanner.py`**: nuevo caso de regresión —
  `commence_time="2026-07-27T02:00:00Z"` debe dar
  `match_date == date(2026, 7, 26)` (cruce de día UTC→Lima), no
  `date(2026, 7, 27)`.
- **`tests/test_staleness.py`**: `test_defaults_reference_date_to_today`
  usa hoy `date.today()` (hora del sistema) para construir el DataFrame de
  prueba — con el default cambiado a `lima_today()`, ese test pasa a
  depender de la fecha de Lima en vez de la del sistema, así que se
  actualiza para construir el DataFrame con `lima_today()` en lugar de
  `date.today()`, evitando que sea flaky en una máquina en otra zona
  horaria.

Verificación final: correr toda la suite de tests (`pytest`) para
confirmar que no se rompe nada fuera de los archivos tocados.

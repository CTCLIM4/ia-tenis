# Runbook: Refreshing tennis data, retraining, and handling suspicious edges

Context: on 2026-07-08 the WTA dataset was frozen at 2023 (tennis-data.co.uk
was briefly unreachable, then recovered). This produced two artificially large
(>10%) value-bet edges on real Wimbledon QF matchups that vanished once the
data was refreshed and the model retrained. This runbook is the procedure used
to diagnose and fix that, so it doesn't have to be reconstructed from scratch.

## 1. How staleness shows up

Every `load_model(tour)` call (in `src/value_analysis.py`) prints a warning
automatically when the newest match in the dataset is stale, using
context-aware thresholds (`src/data/staleness.py`, `_check_staleness`)
instead of one flat number — the old flat 30-day rule missed a dataset that
was functionally stale mid-Wimbledon at only 16 days old (see
`docs/superpowers/specs/2026-07-13-staleness-context-aware-design.md`):

- **Off-season (Dec 1 - Jan 15), last match from the season's tail end
  (Oct-Dec):** lenient — OK up to 45 days, CRITICAL beyond that.
- **Live-tournament mode** (`live_tournament_mode=True`, not currently wired
  to a CLI flag — pass it directly if calling `_check_staleness`/
  `evaluate_staleness` programmatically): strict — WARNING >1 day, CRITICAL
  >2 days.
- **Otherwise (regular tournament weeks):** WARNING >3 days, CRITICAL >7 days
  — tighter than the old flat 30-day threshold by design.

```
*** ADVERTENCIA: dataset ATP desactualizado (regla: regular_week) ***
*** Ultimo partido en los datos: 2026-07-14 (6 dias atras).
*** Las predicciones no incorporan resultados posteriores a esa fecha.
```

This fires whether the model came from cache or was freshly rebuilt — it's
about the age of the *match data*, not the age of the *cache file* (a separate,
already-existing 7-day cache-expiry concept, `CACHE_MAX_AGE_DAYS`). Because
the regular-week window is now much tighter (3d/7d vs. the old 30d), expect
to see this warning routinely between tournaments — it's advisory only
(never blocks the CLI), so don't treat every appearance as an emergency,
just as a prompt to check section 2-3 before trusting a pick.

If you see this warning before a prediction you're about to trust, refresh
that tour's data first (sections 2-3), then retrain (section 4), then
optionally validate (section 5) before proceeding.

## 2. Refreshing ATP

```bash
cd D:\ia-tenis
./tenis-env/Scripts/python.exe scripts/download_data.py
```

As of 2026-07-08 this downloads from the `stats.tennismylife.org` API (the
previous source, `git clone` against `Tennismylife/TML-Database`, froze on
2026-01-27 and was migrated away from — see
`docs/superpowers/specs/2026-07-08-tml-api-migration-design.md`). Each file's
local size is compared against the API manifest's reported size; **a file
being skipped ("N already current or unchanged") is the expected, normal
case for historical years, not a failure** — most years' data never changes
between runs. The current year's file and `ongoing_tourneys.csv` (live,
in-progress tournaments) do re-download on essentially every run, since their
sizes change as new matches are played; the script automatically merges
`ongoing_tourneys.csv` into the current year's `{year}.csv` afterward, so
`load_atp_matches()` sees it with no extra steps.

If the manifest fetch itself fails (`ERROR: could not fetch ATP manifest`),
that's a real problem worth investigating — check connectivity to
`https://stats.tennismylife.org/api/data-files` directly:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" https://stats.tennismylife.org/api/data-files
```

A non-200 here means the API itself is down; wait and retry. This is
different from the old failure mode ("upstream repo hasn't published new
matches") — the API is a live service, not a snapshot repo, so a persistent
failure here is either an outage or a genuine bug, not just "no new data yet."

**Windows TLS note:** the ATP download code uses `requests` with `certifi`'s
bundled CA file explicitly, not `urllib`'s OS-trust-store default. This is
deliberate: prior diagnostics on this machine found that tools routing
through Windows' schannel (native `curl.exe`, PowerShell's
`Invoke-WebRequest`, or Python's `ssl` module falling back to the OS cert
store) can intermittently fail revocation checks against otherwise-healthy
HTTPS endpoints. `requests` + `certifi` sidesteps the OS store entirely. If
you ever see `curl.exe` or PowerShell fail TLS against this endpoint while
Python succeeds, that's this known Windows quirk — not a reason to add
`verify=False` anywhere in this codebase.

## 3. Refreshing WTA

Same script, different source (tennis-data.co.uk, per-year `.xlsx`/`.xls`
files). If specific recent years fail to download ("no pattern worked"),
check whether the site itself is reachable before assuming the URL patterns
in `scripts/download_data.py` are wrong:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 -A "Mozilla/5.0" http://www.tennis-data.co.uk/
```

A `503` or an SSL handshake failure means **the site is temporarily down**,
not a bug in this repo — the download patterns (`_download_wta_year` in
`scripts/download_data.py`) already try `{year}w/{year}.xlsx`,
`{year}w/{year}w.xlsx`, and several other historical variants. Wait a few
minutes and retry:

```bash
./tenis-env/Scripts/python.exe scripts/download_data.py
```

Once the site responds (root URL returns `200`), re-run the same command —
already-downloaded years are skipped automatically, only missing years are
fetched.

## 4. Retraining

Refreshing the raw files under `data/raw/` does **not** automatically update
the cached model in `data/model_cache/{tour}.pkl` (that cache has its own
7-day expiry, `CACHE_MAX_AGE_DAYS`, unrelated to how current the underlying
match data is). Force a rebuild directly:

```bash
cd D:\ia-tenis
./tenis-env/Scripts/python.exe -c "
from src.value_analysis import load_model
load_model('wta', retrain=True)   # or 'atp'
"
```

This also regenerates the `last_match_date` stored in the cache, so the
staleness warning (section 1) reflects the refreshed data going forward.

## 5. Validating before/after

Before touching `data/processed/{tour}_features.csv`, capture a baseline from
the *current* (pre-refresh) file:

```bash
./tenis-env/Scripts/python.exe -c "
import numpy as np, pandas as pd
from src.backtest.walkforward import walk_forward_backtest

df = pd.read_csv('data/processed/wta_features.csv', parse_dates=['match_date'])
mirror = df.copy()
for col in ('elo_diff', 'rank_diff', 'form_diff', 'surface_form_diff', 'rest_diff'):
    mirror[col] = -mirror[col]
mirror['elo_prob'] = 1.0 - mirror['elo_prob']
mirror['h2h_rate'] = 1.0 - mirror['h2h_rate']
mirror['outcome'] = 0
mirror['is_mirror'] = True
full = pd.concat([df, mirror], ignore_index=True)

results = walk_forward_backtest(full, warmup_years=10)
accs = [m['accuracy'] for m in results.values()]
lls = [m['log_loss'] for m in results.values()]
briers = [m['brier_score'] for m in results.values()]
print(f'Accuracy: {np.mean(accs):.4f}  LogLoss: {np.mean(lls):.4f}  Brier: {np.mean(briers):.4f}')
print(f'Years tested: {len(results)}')
"
```

Then regenerate the features CSV with the refreshed raw data and re-run the
same snippet against the new file:

```bash
./tenis-env/Scripts/python.exe -m src.pipeline wta 2007 2026
```

Compare the two runs' Accuracy/LogLoss/Brier. Expect no material degradation
— if metrics get meaningfully worse after a refresh, investigate the loader
before trusting the new data (e.g. verify a specific player's known result
history against the raw file directly, the way Jasmine Paolini's 2024
Wimbledon run was cross-checked against `data/raw/tennis_wta_tduk/2024w.xlsx`
on 2026-07-08).

## 6. Suspicious edges

`src/value_analysis.py` defines `SUSPICIOUS_EDGE_THRESHOLD = 0.10`. Run with
`--halt-on-suspicious` when you want the CLI to refuse to log any match whose
best-side edge exceeds 10% — useful for routine/unattended sessions where you
won't be scrutinizing every prediction by hand:

```bash
./tenis-env/Scripts/python.exe -m src.value_analysis --halt-on-suspicious
./tenis-env/Scripts/python.exe -m src.value_analysis --wta --halt-on-suspicious
```

When it blocks a match, the log-save prompt is skipped entirely for that
match in that session — there is no way to override it without re-running
without the flag. Omit the flag when you've already manually verified a large
edge is real (e.g. after confirming the data isn't stale) and want to log it
anyway.

A large edge is not proof of a bug or of real market value by itself — it's a
prompt to check: is the data for both players current (section 1)? Does the
player-name match the right person (see the WTA disambiguation logic in
`_resolve_player_name`)? Only log it once you can answer both confidently.

## 7. Dataset snapshots (reproducibility)

Full design: `docs/superpowers/specs/2026-07-24-snapshot-persistence-design.md`.

`tennis-data.co.uk`/TML can silently revise historical data (see
[[project-tennis-data-couk-silent-historical-updates]] in project memory) —
a backtest or a live prediction run today against `data/raw/`/`data/processed/`
is not guaranteed to be reproducible next month. A snapshot freezes a copy.

**Create a snapshot** (requires `data/processed/{tour}_features.csv` to
already exist for each tour — run `python -m src.pipeline {tour}` first if not):

```bash
./tenis-env/Scripts/python.exe scripts/create_snapshot.py                  # id = today
./tenis-env/Scripts/python.exe scripts/create_snapshot.py --id 2026-07-24  # explicit id
./tenis-env/Scripts/python.exe scripts/create_snapshot.py --tours atp      # one tour only
```

This copies both `data/raw/{tour dir}/` and `data/processed/{tour}_features.csv`
into `data/snapshots/{id}/`, and writes `data/snapshots/{id}/metadata.json`
(per-tour `last_match_date`, row count, and a sha256+byte-count per file).

**Run a backtest against a pinned snapshot** instead of live data:

```bash
./tenis-env/Scripts/python.exe -m src.pipeline atp --snapshot 2026-07-24
```

This skips loading/rebuilding features entirely — it reads the snapshot's
`{tour}_features.csv` directly and does **not** touch the live
`data/processed/{tour}_features.csv`.

**Load a value-bet model pinned to a snapshot**:

```bash
./tenis-env/Scripts/python.exe -m src.value_analysis --snapshot 2026-07-24
./tenis-env/Scripts/python.exe -m src.value_analysis --wta --snapshot 2026-07-24
```

A snapshot-pinned model uses its own cache file
(`data/model_cache/{tour}__snapshot-{id}.pkl`) that never expires — it
can't collide with or be evicted by a live `--retrain`. The usual staleness
warning (section 1) is skipped under `--snapshot`: a deliberately old,
pinned dataset isn't "stale" in the sense that warning exists to catch.

**What's tracked in git:** only `data/snapshots/{id}/metadata.json` — the
copied CSV/xlsx files themselves are gitignored, same as `data/raw/` and
`data/processed/` already are (see `.gitignore`). This keeps a permanent,
lightweight, diffable record of every snapshot ever taken without
versioning tens of MB of data per snapshot.

**Verify a snapshot hasn't been locally corrupted or tampered with:**

```bash
./tenis-env/Scripts/python.exe -c "
from src.data.snapshots import verify_snapshot_integrity
print(verify_snapshot_integrity('2026-07-24'))
"
```

## 8. Calibration audit log

Full design: `docs/superpowers/specs/2026-07-25-calibration-persistence-design.md`.

`value_bets_log.csv` only ever contained rows a human chose to save — a
biased sample that can't support a real reliability/calibration analysis
(you need the full population of predictions the model made, not just the
ones that looked interesting enough to log). `data/prediction_audit_log.csv`
fixes this: **every** prediction the CLI evaluates (once odds are entered)
is appended here automatically, no prompt, regardless of whether it was
also saved to `value_bets_log.csv`.

Each row's `decision` column says what happened to that prediction:

- `logged` — also saved to `value_bets_log.csv` (you said yes).
- `passed_low_edge` — neither side had positive edge; nothing to log.
- `passed_user_declined` — at least one side had positive edge, but you
  said no.
- `blocked_suspicious` — `--halt-on-suspicious` blocked it (edge >10%).
  These never appear in `value_bets_log.csv` at all; this is the only
  record they leave anywhere.
- `invalid_missing_elo` — one or both players had no trained Elo rating.

Each row also carries `shrink_hi`/`shrink_lo`/`shrink_rate` (the exact
calibration constants active when `p_a_cal` was computed — see
`apply_shrinkage()` in `src/value_analysis.py`) and `model_commit` (the git
commit of the code that produced the prediction) plus `model_snapshot_id`
(set when the session was pinned via `--snapshot {id}`, empty for a live
model) — so a future recalibration study can tell exactly which model
version and which calibration constants produced any given historical row,
even after those constants change.

This file is gitignored, same as `value_bets_log.csv` — it's local
prediction history, not something to commit. Writing to it is best-effort:
a failure (disk full, permissions) prints a warning and the interactive
session continues rather than crashing — it never blocks or interferes
with the `value_bets_log.csv` save that happens moments earlier in the
same prompt.

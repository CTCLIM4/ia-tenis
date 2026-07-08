# TML ATP Data Source Migration — Design Spec

**Date:** 2026-07-08
**Status:** Approved for planning

## Goal

The ATP loader has sourced data from `git clone https://github.com/Tennismylife/TML-Database.git`
(via `scripts/download_data.py`). That repo froze on 2026-01-27 (last commit: "Delete
index.html") — it is no longer being updated. The same organization runs an active API,
`https://stats.tennismylife.org/api/data-files`, serving the identical 50-column schema
(verified byte-identical column names against the local `2025.csv`) with per-year CSVs
current through today plus a live `ongoing_tourneys.csv` for in-progress tournaments. This
spec migrates `scripts/download_data.py` from the frozen git clone to this active HTTP
source, with zero changes to `src/data/loader.py` or anything downstream of it.

## Non-goals

- No changes to `src/data/loader.py`, `src/models/`, `src/features/`, `src/backtest/`. The
  loader reads `data/raw/tennis_atp_tml/{year}.csv` by explicit path per year; as long as
  that contract holds, nothing downstream needs to know the source changed.
- No download of `{year}_challenger.csv`, `atp_quali/*`, `challenger_ongoing_tourneys.csv`,
  or the aggregate `ATP_Database.csv` the API also serves. The loader never reads challenger,
  qualifying, or aggregate files — adding them is out of scope.
- No `verify=False` or other TLS-verification bypass, under any circumstance. Pre-flight
  (below) confirmed standard verification works.
- No changes to the WTA download path (`_download_wta_year` and friends) — untouched.

## Pre-flight verification (completed before this spec was written)

- `requests` (2.34.2) + `certifi` against `GET https://stats.tennismylife.org/api/data-files`
  with default verification (certifi installed → `requests` resolves its CA bundle to
  `certifi.where()` automatically; no explicit `verify=` override needed) returned `200`,
  `application/json`, no warnings.
- Native Windows `curl.exe`, Git Bash `curl`, and PowerShell `Invoke-WebRequest` all also
  returned `200` today. The previously-diagnosed schannel/revocation-check failure on Windows
  did not reproduce against this endpoint today — noted as a known-flaky-not-blocking caveat
  in the runbook (see below), not treated as a live blocker.
- Manifest shape: `{"count": 131, "files": [{"name", "url", "size", "mtime"}, ...]}`.
- Schema check: `2025.csv` (API) columns == `2025.csv` (local, from the frozen clone) columns,
  exactly, in order. 50 columns, not 49 (the pre-diagnosis count was off by one; column *names*
  match regardless).
- `ongoing_tourneys.csv` uses the identical 50-column schema. Today it contains only
  `tourney_id = 2026-540` (Wimbledon), `tourney_date` 20260629–20260707, 122 rows.
- `(tourney_id, match_num)` is **not** a reliable unique key within a single year file:
  `2026.csv` has `tourney_id = 2026-416` reused across two unrelated tournaments (Munich and
  Rome Masters), both with `match_num = 1`. Any dedup/merge logic must use a wider key.

## Architecture

All changes are contained in `scripts/download_data.py`. The `WTA_*` / `_download_wta_year`
section is untouched. The `ATP_REPO` git-clone block is replaced by the following.

### Manifest fetch

`_fetch_atp_manifest() -> list[dict]` — `GET https://stats.tennismylife.org/api/data-files`
(default TLS verification, `requests`), returns the `files` list filtered to entries matching
`^\d{4}\.csv$` (main-tour year files) or exactly `ongoing_tourneys.csv`. Everything else in
the manifest (challenger, quali, `ATP_Database.csv`) is dropped at this filter step.

### Skip-download decision

`_should_download(local_path: Path, remote_size: int) -> bool` — pure function: `True` if
`local_path` doesn't exist, or its byte size differs from `remote_size`; `False` otherwise.
Applied per-file against the manifest's `size` field. This means historical years (whose
remote size essentially never changes) are skipped on repeat runs, while `ongoing_tourneys.csv`
and the current year's file (whose sizes change as matches are added) get re-fetched almost
every run — without hardcoding which years count as "historical."

### Atomic download

`_download_file(url: str, dest: Path) -> None` — downloads to `dest.with_suffix(dest.suffix +
".tmp")`, validates the response parses as CSV and its header row exactly matches a hardcoded
`ATP_SCHEMA_COLUMNS` constant (the 50 column names captured in the pre-flight check above, e.g.
`["tourney_id", "tourney_name", ..., "l_bpFaced"]`) — a cheap defensive check that catches
truncated downloads or an HTML error page served with a 200, independent of whether a local
file already exists to compare against. On validation failure, the `.tmp` file is discarded and
the existing `dest` (if any) is left untouched; the download is logged as failed and the script
continues to the next file rather than aborting the whole run. Only after validation passes does
`os.replace()` move the temp file onto `dest`. A failed or partial download never overwrites a
good existing file, and never leaves a corrupt file at the final path.

### Merging `ongoing_tourneys.csv` into the current year

"Current year" is `datetime.date.today().year` (same pattern already used for `WTA_START_YEAR`
in this file). After downloading (or confirming up-to-date) both the current year's
`{year}.csv` and `ongoing_tourneys.csv`, `_merge_ongoing_into_year(year_df, ongoing_df) ->
pd.DataFrame`:

1. Concatenates both frames.
2. Deduplicates on the composite key `(tourney_id, tourney_name, round, match_num)` — wider
   than `(tourney_id, match_num)` specifically because that pair collided across unrelated
   tournaments (Munich/Rome Masters, see pre-flight above).
3. On a duplicate key, the row from `year_df` (the archived file) wins over the row from
   `ongoing_df` (the live feed) — the archived file is treated as the settled record.

The merged, deduplicated result is written — atomically, same as any other download — back to
`data/raw/tennis_atp_tml/{current_year}.csv`. `ongoing_tourneys.csv` is also kept on disk as
its own file (`data/raw/tennis_atp_tml/ongoing_tourneys.csv`), unmerged, purely as an audit
trail; `loader.py` never reads it since it addresses `{year}.csv` by explicit path, not glob.

### One-time directory cleanup

On first run of the new script against the existing `data/raw/tennis_atp_tml/` directory
(currently a git-clone working copy), before downloading: delete `.git/`, `.github/`,
`README.md`, `logo.jpg`, `ATP_Database.csv`, and the existing `ongoing_tourneys.csv` (a stale
pre-freeze copy, 79 matches, never read by `loader.py`). Existing `{year}.csv` files are left
in place — the size-check decides whether each gets re-downloaded in the same run. This is
safe: `data/raw/` is `.gitignore`d (confirmed), so none of this is tracked project history —
it's reproducible cache.

## Compatibility with the loader

`load_atp_matches(start_year, end_year)` in `src/data/loader.py` iterates
`range(start_year, end_year + 1)`, reading `data/raw/tennis_atp_tml/{year}.csv` per year if it
exists. It does not enumerate the directory and has no knowledge of `.git`, `ongoing_tourneys.csv`,
or any other file. Because the current year's merged file is written to the exact same
`{year}.csv` path the loader already expects, **zero changes to `loader.py` are required**.
This was confirmed by reading the current implementation, not assumed.

## Staleness check compatibility

`_check_staleness` in `src/value_analysis.py` derives `last_match_date` from
`df_raw["match_date"].max().date()` on whatever the loader returns — it has no coupling to the
data source, URL, or config. No changes needed there; a fresh `ongoing_tourneys.csv` merge
naturally produces a more current `last_match_date`, which is the entire point of this
migration.

## Rollback strategy

- **Code:** `scripts/download_data.py` is the only file this spec touches and is git-tracked.
  `git revert` on the migration commit(s) restores the git-clone-based flow.
- **Data:** `data/raw/` is `.gitignore`d, so there is no git-level "data rollback" — it's cache,
  not project history. If the new HTTP flow ever produces bad data:
  - Re-running the script is the first recovery step — atomic writes plus size-based skip mean
    already-good historical years are untouched unless their remote size actually changed.
  - As a last resort, the frozen GitHub repo still exists (frozen, not deleted):
    `git clone --depth=1 https://github.com/Tennismylife/TML-Database.git` to a scratch
    directory remains available to recover or diff against a pre-migration snapshot.
- No window exists where a partially-written file can overwrite a good one — atomicity is the
  core safety property, not a manual backup step.

## Test plan

New `tests/test_download_data.py`, structured like `tests/test_odds_api.py` (no real network
calls; `requests.get` mocked/monkeypatched):

- `_should_download`: missing local file → `True`; local size ≠ remote size → `True`; sizes
  equal → `False`.
- `_merge_ongoing_into_year`: dedup collapses a true duplicate match across the two frames;
  archived (`year_df`) row wins on conflict; a synthetic case reproducing the
  Munich/Rome-Masters `tourney_id` collision confirms the composite key does **not**
  incorrectly collapse two distinct matches that merely share `tourney_id` + `match_num`.
- `_download_file`: a simulated failure mid-download (mocked response raises partway) leaves
  any pre-existing file at `dest` untouched, and leaves no `.tmp` artifact behind at the final
  path.
- Manifest filtering: a sample JSON payload containing year files, `ongoing_tourneys.csv`,
  challenger files, quali files, and `ATP_Database.csv` — only the first two categories survive
  the filter.

Existing suite: `tests/test_loader.py` and the rest of the test suite must pass unmodified —
that's the proof the loader contract didn't change. No existing tests should need edits.

## Documentation updates

- `docs/runbooks/data-refresh-and-staleness.md`, section 2 ("Refreshing ATP"): replace the
  `git pull` / "check upstream repo's last commit" instructions with the new HTTP flow
  (re-run `scripts/download_data.py`; explain the size-based skip so "nothing downloaded"
  reads correctly as "already current," not as a failure).
- Add a short Windows TLS caveat note (in the runbook or README, wherever the existing WTA
  curl-based troubleshooting note lives): Python `requests` + `certifi` is the supported check
  for this endpoint; if `curl.exe`/schannel ever fails a revocation check against it while
  Python succeeds, that's a known Windows-networking quirk, not a reason to add
  `verify=False` anywhere.

## Spec coverage check

| Requirement | Covered by |
|---|---|
| TLS pre-flight with requests+certifi, no `verify=False` | Pre-flight section |
| Historical data in `{year}.csv`, active tourney in `ongoing_tourneys.csv` | Architecture — manifest fetch, merge |
| Dedup/precedence policy for ongoing vs. archived year data | Merging section (composite key, archived-file wins) |
| Zero changes to `loader.py`/`models`/`features`/`backtest` | Compatibility with the loader section |
| Rollback strategy without losing already-downloaded historical data | Rollback strategy section |
| Test plan: new + existing tests | Test plan section |
| Staleness check keeps working, updated for new source if needed | Staleness check compatibility (no change needed, confirmed) |

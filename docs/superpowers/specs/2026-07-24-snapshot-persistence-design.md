# Dataset Snapshots & Reproducible Persistence — Design Spec

**Date:** 2026-07-24
**Status:** Confirmed — all open design questions (Q1-Q5) resolved by the user 2026-07-24, all following the recommended option. Ready for an implementation plan. This was the requested persistence design session ([[project-pending-persistence-design]], deferred since 2026-07-09), scoped narrowly to what the user asked for today: dataset snapshots for reproducible backtests and model loading. It does **not** cover the other half of that memory (persisting calibration/PASS observations from value-bet analysis) — that stays open, tracked separately.

## Goal

Let a specific dataset state be frozen and later re-loaded on demand, so:
- A backtest run today against `data/processed/atp_features.csv` can be re-run next month against *exactly the same bytes*, even though `tennis-data.co.uk`/TML silently revise historical files over time ([[project-tennis-data-couk-silent-historical-updates]] — confirmed real, not hypothetical: re-running the downloader once already re-fetched three "closed" WTA seasons).
- A live prediction model can be pinned to a known-good dataset state instead of always reflecting whatever's currently in `data/raw/` — useful after finding a data-corruption bug like the 2029-date typo ([[project-wta-future-date-guard]]) to prove a fix actually changed the outcome, or to reproduce a specific historical value-bet pick's exact model state for audit.

## Current data flow (as of this session, verified against the code)

```
data/raw/tennis_atp_tml/{year}.csv          (60 files, ~37 MB)
data/raw/tennis_wta_tduk/{year}w.xlsx       (20 files, ~12 MB)
        │
        │  src/data/loader.py: load_atp_matches() / load_wta_matches()
        ▼
   raw match DataFrame
        │
        ├─────────────────────────────────────┐
        │                                      │
        ▼ src/pipeline.py: run_pipeline()      ▼ src/value_analysis.py: _build_elo_fb()
   build_match_features()                 build_match_features()
        │                                      │
        ▼                                      ▼
data/processed/{tour}_features.csv        live EloSystem + FeatureBuilder +
   (~22 MB ATP / ~8 MB WTA)                EloHistoryTracker + rank_lookup +
        │                                  age_lookup (in-memory, not written
        │ src/value_analysis.py:           to data/processed/ at all)
        │ _train_lr() reads this               │
        ▼                                      ▼
   LogisticRegression                 data/model_cache/{tour}.pkl
   (trained on this CSV)              (pickles clf + all of the above,
                                        7-day cache, keyed by tour only)
```

**Important finding, not obvious from the user's request as stated:** there are **two independent consumers of raw data**, not one. `run_pipeline()` (backtest) and `_build_elo_fb()` (live model) both call `build_match_features()` on raw data *separately* — `_build_elo_fb` does **not** read `data/processed/{tour}_features.csv` at all, it replays the raw match history itself to build the live `EloSystem`/`FeatureBuilder` state that `predict_match()` needs. The processed CSV is only ever used to train the LR's weights (`_train_lr`).

**Consequence for snapshot scope:** a snapshot containing only `data/processed/{tour}_features.csv` is fully sufficient to pin `walk_forward_backtest()` (the "correr el backtest" half of the request) — but is **not** sufficient to pin `load_model()`'s live Elo/FeatureBuilder state for prediction (the "cargar modelos" half), because `_build_elo_fb` needs the raw per-match data (winner/loser/date/surface/rank/age), not the aggregated feature-diff columns the CSV stores. The features CSV literally can't be reversed back into "what was player X's Elo on date Y" — that information only exists during the raw-data replay.

This is the central open question below (Q1) — not a detail to quietly resolve either way.

## Proposed structure

```
data/snapshots/
  2026-07-24/
    metadata.json
    processed/
      atp_features.csv
      wta_features.csv
    raw/                        ← only if Q1 resolves to "yes, include raw"
      tennis_atp_tml/{year}.csv
      tennis_wta_tduk/{year}w.xlsx
```

`metadata.json` (per the user's spec — `last_match_date`, row counts, file hashes — plus a few fields I'd add and flag for confirmation):

```json
{
  "snapshot_id": "2026-07-24",
  "created_at": "2026-07-24T19:05:00-05:00",
  "git_commit": "f8b4c0c",
  "tours": {
    "atp": {
      "last_match_date": "2026-07-21",
      "n_rows": 116913,
      "files": {
        "processed/atp_features.csv": {
          "sha256": "…",
          "bytes": 22236213
        }
      }
    },
    "wta": {
      "last_match_date": "2026-07-20",
      "n_rows": 47019,
      "files": {
        "processed/wta_features.csv": {
          "sha256": "…",
          "bytes": 8435749
        }
      }
    }
  }
}
```

Additions beyond the user's literal spec, called out for approval/rejection individually:
- `git_commit` — pins the *code* version alongside the *data* version. Without it, "reproduce this backtest" only pins half the equation (feature-engineering logic in `src/backtest/walkforward.py` could have changed between snapshot creation and later reuse). Cheap (`git rev-parse --short HEAD`), high value. Recommend: **include**.
- `created_at` timestamp — informational, not load-bearing. Recommend: **include**, trivial cost.
- Per-file `bytes` alongside `sha256` — cheap sanity check before even hashing (catches truncated copies fast). Recommend: **include**.

## Design decisions (all confirmed 2026-07-24 — recommended option chosen in every case)

### Q1 — Does a snapshot include raw data, or processed-features-only?

**CONFIRMED: (b) raw + processed (~79 MB/snapshot).**

- **(a) Processed-only** (~30 MB/snapshot: 22 MB ATP + 8 MB WTA). Fully covers backtest reproducibility. Does **not** let `load_model()` be pinned for live prediction — `--snapshot` on `value_analysis.py` would only be able to pin *which LR weights* get used, not the Elo ratings/rankings/ages that generate the feature vector for a new prediction. I think this is a materially weaker version of "cargar modelos" than what was asked for.
- **(b) Raw + processed** (~79 MB/snapshot: adds 37 MB ATP + 12 MB WTA raw). Fully covers both use cases — `load_model(tour, snapshot=...)` would run `_build_elo_fb` against the snapshot's raw files instead of live `data/raw/`, producing a genuinely pinned Elo/FeatureBuilder/rank_lookup/age_lookup state. Cost: ~80 MB per snapshot on disk, and `_build_elo_fb`'s replay cost (~30-60s per tour) would still be paid every time a snapshot-pinned model is built (no shortcut — unless a snapshot also caches its own model pickle, see Q4).
- **My recommendation: (b).** The stated goal ("cargar modelos... indicando un snapshot específico") implies pinning what a prediction actually depends on, and (a) would silently only pin the training weights while ratings keep drifting — a confusing half-measure that undersells what "snapshot" implies. The storage cost (79 MB) is small relative to what's already regenerated locally and gitignored.

### Q2 — Committed to git, or gitignored like `data/raw/` and `data/processed/` already are?

**CONFIRMED: metadata.json committed, data files gitignored.**

- The repo's existing convention (README, `.gitignore`) is that all of `data/raw/`, `data/processed/`, `data/model_cache/` are **not** committed — "se genera localmente... nunca se commitea." A snapshot's data files (CSV/xlsx copies) are the same category of artifact.
- But `metadata.json` alone is tiny (a few KB) and is exactly the kind of "git-auditable trace" [[feedback-persist-ephemeral-output]] asks for — it documents *that* a snapshot existed, its hash, its row counts, without bloating the repo.
- **My recommendation:** gitignore `data/snapshots/*/processed/` and `data/snapshots/*/raw/` (same as the live equivalents), but **commit** `data/snapshots/*/metadata.json` specifically (a `.gitignore` pattern like `data/snapshots/*/` + `!data/snapshots/*/metadata.json` handles this). Gives a permanent, lightweight, diffable record of every snapshot ever taken — including ones whose actual data files were later deleted to reclaim disk space — without versioning tens of MB of CSVs per snapshot in git history.

### Q3 — How does `--snapshot` interact with the existing 7-day model cache?

**CONFIRMED: separate cache file per snapshot, no expiry.**

- Today's cache key is `data/model_cache/{tour}.pkl`, one file per tour, no snapshot dimension.
- If `--snapshot 2026-07-24` reuses that same cache file, a snapshot-pinned load could clobber (or be clobbered by) a live/current load — two different intents sharing one cache slot.
- **My recommendation:** snapshot-pinned caches get their own file, `data/model_cache/{tour}__snapshot-{id}.pkl`, and bypass the 7-day expiry entirely (a snapshot is immutable by definition — "expiring" a pinned historical state doesn't make sense the way it does for live data). First load per snapshot pays the ~30-60s rebuild cost once; subsequent loads of the same snapshot hit its dedicated cache file indefinitely.

### Q4 — Staleness check behavior under `--snapshot`

**CONFIRMED (assistant recommendation adopted, not separately asked — lower-stakes UX detail): (i) skip the staleness check under `--snapshot`.**

- `_check_staleness` (`src/value_analysis.py`) compares `last_match_date` against **today**. Loading a deliberately old, pinned snapshot would routinely trigger CRITICAL/WARNING under the context-aware rules — technically correct (the data *is* old relative to today) but noisy/misleading for a use case where "old on purpose" is the entire point.
- When `--snapshot` is active, skip the staleness check entirely and print a one-line "using pinned snapshot {id}, data as of {last_match_date}" notice instead — staleness-vs-today is a meaningless question once you've explicitly opted into a fixed historical dataset.

### Q5 — Snapshot creation trigger: new script, or a flag on existing ones?

**CONFIRMED: (a) new `scripts/create_snapshot.py`.**

- **(a)** New `scripts/create_snapshot.py`, mirroring `scripts/download_data.py`'s role — orchestrates: ensure `data/processed/{tour}_features.csv` is current (calls `run_pipeline`'s load+build steps, or requires the caller to have already run `python -m src.pipeline` — needs deciding), copy raw+processed files into `data/snapshots/{date}/`, hash, write `metadata.json`.
- **(b)** A `snapshot` subcommand bolted onto `src/pipeline.py`'s existing `argparse`.
- **My recommendation: (a)**, new dedicated script — snapshot creation is a distinct operational action (like `download_data.py`), not a pipeline-run variant, and keeping it separate avoids overloading `pipeline.py`'s argparse with an unrelated mode.

## Proposed module: `src/data/snapshots.py`

Mirrors the existing `src/data/staleness.py` pattern (small, pure-function-heavy, its own test file):

- `create_snapshot(snapshot_id: Optional[str] = None, tours=("atp", "wta")) -> Path` — defaults `snapshot_id` to today's ISO date; raises if a snapshot with that id already exists (snapshots are immutable — no silent overwrite; a caller wanting to redo one deletes the directory explicitly first).
- `list_snapshots() -> list[str]` — sorted snapshot ids found under `data/snapshots/`.
- `load_snapshot_metadata(snapshot_id: str) -> dict`.
- `resolve_snapshot_path(snapshot_id: str, tour: str, kind: Literal["processed", "raw"]) -> Path`.
- `verify_snapshot_integrity(snapshot_id: str) -> bool` — re-hashes files on disk against `metadata.json`, catches local corruption/manual tampering. (This is what makes the hashes in `metadata.json` actually useful, not just decorative.)

## Proposed CLI surface

```bash
# Create a snapshot from current data/raw + data/processed state
./tenis-env/Scripts/python.exe scripts/create_snapshot.py
./tenis-env/Scripts/python.exe scripts/create_snapshot.py --id 2026-07-24  # explicit id, else today

# Backtest against a pinned snapshot instead of live data/processed/
./tenis-env/Scripts/python.exe -m src.pipeline atp --snapshot 2026-07-24

# Load a value-bet model pinned to a snapshot instead of live data
./tenis-env/Scripts/python.exe -m src.value_analysis --snapshot 2026-07-24
./tenis-env/Scripts/python.exe -m src.value_analysis --wta --snapshot 2026-07-24
```

`src/pipeline.py` changes: when `--snapshot` is passed, skip the load-raw + build-features steps entirely and read `resolve_snapshot_path(id, tour, "processed")` directly into `walk_forward_backtest()` — faster than a live run too, since feature-building is already done.

`src/value_analysis.py` changes: `load_model(tour, retrain=False, snapshot: Optional[str] = None)` — when `snapshot` is set, `_build_elo_fb` reads from `resolve_snapshot_path(snapshot, tour, "raw")` instead of `data/raw/`, `_train_lr` reads from the snapshot's `processed/` path, and caching/staleness follow Q3/Q4 above.

## Non-goals (this pass)

- Not building a snapshot *retention/pruning* policy (auto-delete old snapshots) — disk space isn't a known problem yet; revisit if `data/snapshots/` actually grows large in practice.
- Not persisting calibration/PASS value-bet observations — that's the other half of [[project-pending-persistence-design]], a separate design conversation.
- Not adding snapshot support to `scripts/download_data.py` itself (e.g. auto-snapshotting before every download) — [[project-pending-deferred-items]] already tracks "auto-snapshot model_cache before retrain" as its own separate, still-open item; worth revisiting together once this lands, not bundled in.
- Not changing `data/raw/` or `data/processed/` themselves — snapshots are copies, the live directories keep working exactly as they do today.

## Test plan (for the eventual implementation plan, not written yet)

- `create_snapshot()`: creates expected directory structure, `metadata.json` has correct hashes/row counts/last_match_date verified against known fixture data, raises on duplicate id, handles missing source files gracefully (e.g. `data/processed/wta_features.csv` doesn't exist yet — clear error, not a silent partial snapshot).
- `verify_snapshot_integrity()`: detects a tampered/truncated file, passes on an untouched snapshot.
- `resolve_snapshot_path()`: correct paths for both `tours` × both `kind`s, raises on unknown snapshot id.
- `src/pipeline.py --snapshot`: backtest results against a snapshot are byte-identical across two separate invocations (the actual reproducibility guarantee this whole feature exists to provide) — the load-bearing regression test for this entire feature.
- `src/value_analysis.py --snapshot`: `load_model` with a snapshot id produces the same `elo`/`clf` state on repeated calls; snapshot cache file is separate from the live one (loading a snapshot doesn't clobber or get clobbered by a live `--retrain`).
- No test should require network access or mutate `data/raw/`/`data/processed/` — snapshot tests build their own tiny fixture directories, consistent with existing test style.

## Rollback strategy

- Fully additive: new module, new script, new `--snapshot` flags with `None` defaults preserving today's exact behavior when omitted. No changes to existing data formats, cache payload shape, or CSV schemas.
- `data/snapshots/` is a new gitignored-mostly directory (metadata committed) — reversible with `git rm` on the committed metadata files and local directory deletion, no migration needed either direction.

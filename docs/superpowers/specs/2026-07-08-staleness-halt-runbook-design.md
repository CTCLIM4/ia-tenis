# Staleness Check + --halt-on-suspicious + Runbook — Design Spec

**Date:** 2026-07-08
**Status:** Approved for planning

## Goal

Codify two safety checks and a runbook that emerged from manually debugging a real incident today: the WTA dataset was frozen at 2023 (tennis-data.co.uk was briefly unreachable, then recovered), which produced two artificially large ">10% edge" value-bet signals on Kostyuk/Paolini and Noskova/Mertens that vanished after refreshing and retraining. Both the staleness of the underlying data and the suspiciously large edge were caught by hand, ad hoc, in this session. This spec turns that manual process into: (1) an always-on warning when a tour's dataset is stale, (2) an opt-in CLI flag that hard-blocks logging a suspiciously large edge instead of relying on the operator to notice and pause, and (3) a runbook documenting the refresh/retrain/validate procedure so it doesn't have to be reconstructed from scratch next time.

## Non-goals

- No automatic data refresh or retraining triggered by the staleness check — it only warns. Actually refreshing (running `download_data.py`, retraining) stays a manual, deliberate action per the runbook.
- No configurable staleness or edge thresholds via environment variables. Both are fixed constants in `src/value_analysis.py`, matching the existing style of `KELLY_CAP`/`CACHE_MAX_AGE_DAYS` rather than the env-var-configurable style used for `ODDS_API_*` (a deliberate choice — these are safety constants, not per-user tuning knobs).
- No changes to `src/odds_api.py` or `The Odds API` integration — unrelated to this feature.
- No retroactive staleness check on `data/value_bets_log.csv` rows already logged — this only affects future `load_model()` calls and future log-saves.

## Architecture

### Staleness check (`src/value_analysis.py`)

- New constant: `STALENESS_WARNING_DAYS = 30`.
- The model cache payload (already `{"elo", "fb", "clf", "rank_lookup", "timestamp", "tour"}` in `_save_cache`) gains one new key: `"last_match_date"`, a `date` — the max `match_date` seen in `df_raw` at build time.
- `_build_elo_fb(tour)` computes and returns this alongside the existing `(elo, fb, rank_lookup)` tuple — signature becomes `(elo, fb, rank_lookup, last_match_date)`.
- New function `_check_staleness(tour: str, last_match_date: date) -> None`: computes `days_stale = (date.today() - last_match_date).days`; if `days_stale > STALENESS_WARNING_DAYS`, prints a warning block (same visual style as the existing missing-Elo warning in `_print_prediction`):
  ```
  *** ADVERTENCIA: dataset {TOUR} desactualizado ***
  *** Ultimo partido en los datos: {last_match_date} ({days_stale} dias atras).
  *** Las predicciones no incorporan resultados posteriores a esa fecha.
  ```
- `load_model()` calls `_check_staleness(tour, last_match_date)` unconditionally on every call — whether `last_match_date` came from a freshly-built model or from the cached payload — so the warning reflects true data age relative to *today*, not cache age. (Cache age, governed by `CACHE_MAX_AGE_DAYS`, is a separate, already-existing concept — how old the *pickle* is, not how old the *match data* is.)
- Cache payloads written before this change won't have `"last_match_date"`. `_load_cache` treats a missing key as itself stale (forces a rebuild) rather than crashing — the same defensive posture the existing `_load_cache` already takes for a corrupt/unreadable cache.

### `--halt-on-suspicious` (`src/value_analysis.py`)

- New constant: `SUSPICIOUS_EDGE_THRESHOLD = 0.10`, next to `KELLY_CAP`.
- `main()`'s argparse gains `--halt-on-suspicious` (`action="store_true"`, default `False`), passed through to `interactive_cli(tour, retrain, halt_on_suspicious)`.
- `interactive_cli` gains the parameter `halt_on_suspicious: bool = False`.
- After `val_a`/`val_b` are computed (existing code) and the prediction is printed (existing `_print_prediction` call), insert a check before the existing `elo_ok` / "Guardar en log?" block:
  ```python
  best_edge = max(val_a["edge"], val_b["edge"])
  if halt_on_suspicious and best_edge > SUSPICIOUS_EDGE_THRESHOLD:
      print(f"\n  *** BLOQUEADO: edge sospechoso ({best_edge*100:.1f}% > "
            f"{SUSPICIOUS_EDGE_THRESHOLD*100:.0f}%) ***")
      print("  *** Posible dato stale o error de matching. Revisa manualmente.")
      print("  *** No se guarda en esta sesion. Corre sin --halt-on-suspicious para loguear igual.")
  elif not elo_ok:
      ... # existing branch, unchanged
  else:
      ... # existing "Guardar en log?" prompt, unchanged
  ```
  This is a hard block: when triggered, the save prompt never appears for that match in that session — consistent with the "bloqueo duro" behavior chosen (not a flippable default).

## Data flow

1. User runs `python -m src.value_analysis --halt-on-suspicious [--wta] [--retrain]`.
2. `load_model()` loads or rebuilds the model; `_check_staleness` unconditionally prints a warning if data is >30 days old, regardless of cache hit/miss.
3. `interactive_cli` proceeds exactly as today through prediction and value calculation.
4. Before the log-save prompt, the new suspicious-edge check runs. If it trips, the match is never offered for logging in this session; otherwise behavior is unchanged from today (including the existing missing-Elo hard-block, which stays checked first... actually checked as an `elif` alongside it — see Edge Cases below).

## Edge cases

- **Both missing-Elo AND suspicious-edge conditions true at once:** the missing-Elo block already prevents logging today (`elo_ok` gate) — order the new check first (as shown above) since a suspicious edge is itself often *caused* by a name-resolution/stale-data problem, and the more specific "edge sospechoso" message is more actionable than "Elo faltante" when both apply. If only Elo is missing (not suspicious edge), the existing missing-Elo message still shows via the `elif`.
- **`--halt-on-suspicious` not passed:** behavior is 100% unchanged from today (opt-in, default off) — this is what happened in the current session's manual runs and what still allows deliberately logging a large edge when the operator has already verified it's real (as we did for the ATP matches).
- **Cache from before this change (no `last_match_date` key):** treated as a cache miss, forcing one rebuild that populates it going forward. No crash, no silent staleness blind spot.

## Runbook

New file: `docs/runbooks/data-refresh-and-staleness.md`. Sections, each with copy-pasteable commands drawn from this session's actual transcript:

1. **How staleness shows up** — what the new warning looks like and what it means.
2. **Refreshing ATP** — `scripts/download_data.py`'s `git pull` step; note that "Already up to date" can legitimately mean the upstream `Tennismylife/TML-Database` repo itself hasn't published new matches (check its own last commit date with `git -C data/raw/tennis_atp_tml log -1` before assuming a bug).
3. **Refreshing WTA** — same script's tennis-data.co.uk download step; troubleshooting a `503`/SSL-handshake failure (seen today) — it means the site itself is temporarily down, not a bug in the download patterns; retry after a few minutes.
4. **Retraining** — `load_model(tour, retrain=True)` one-liner to rebuild the cached Elo/FeatureBuilder/LR model from refreshed raw data.
5. **Validating before/after** — the walk-forward-backtest comparison recipe used today (load `data/processed/{tour}_features.csv` before touching it, mirror it, run `walk_forward_backtest`, compare mean accuracy/log-loss/Brier after regenerating the features CSV via `python -m src.pipeline {tour} ...`).
6. **Suspicious edges** — what `SUSPICIOUS_EDGE_THRESHOLD` and `--halt-on-suspicious` do, and when to use the flag (routine/unattended runs) vs. omit it (a single verified pick you already trust).

## Testing

- `tests/test_value_analysis.py` gains:
  - A test for `_check_staleness` (or the underlying day-math) printing the warning when `last_match_date` is old enough, and not printing when recent — using `capsys` to assert on stdout, consistent with how this codebase already tests print-based CLI helpers... *(there is no existing precedent for asserting on prints in this file; see Self-Review)*.
  - A test that `load_model`'s cache payload round-trips `last_match_date` correctly (write a cache with a known date via `_save_cache`, `_load_cache` it back, assert the date matches) — mirrors the existing cache-loading tests' style once written (there are none today for `_save_cache`/`_load_cache` either — this is new coverage).
  - Tests for the suspicious-edge gate logic extracted as a small pure function (see Self-Review below) rather than asserting on `interactive_cli`'s internals, since `interactive_cli` itself has no test coverage today (it's an `input()`-driven REPL, consistent with the rest of the file).
- No new tests for the runbook (it's documentation).

## Self-Review

### Placeholder scan
Original draft left the suspicious-edge decision embedded directly in `interactive_cli`'s `if/elif/else` chain with no unit-testable seam — fixed by extracting it (see below).

### Ambiguity fix: testable seam for the halt logic
To make the "Testing" section's claim honest (this file has no precedent for testing REPL internals), the halt decision is factored into a small pure function that *is* unit-testable without touching `interactive_cli`:
```python
def _should_halt_on_suspicious_edge(val_a: dict, val_b: dict, halt_on_suspicious: bool) -> bool:
    return halt_on_suspicious and max(val_a["edge"], val_b["edge"]) > SUSPICIOUS_EDGE_THRESHOLD
```
`interactive_cli` calls this function instead of inlining the `max(...) > ...` comparison. This function is directly unit-testable (no `input()`, no I/O) with plain dict fixtures, matching the existing `TestCalculateValue`-style tests already in `tests/test_value_analysis.py`.

### Consistency check
`_check_staleness` printing is analogous in style to existing warnings (missing Elo in `_print_prediction`, cache-expiry messages in `_load_cache`) — no new visual language introduced.

### Scope check
Single cohesive change across one file (`src/value_analysis.py`) plus one new doc — does not need decomposition into multiple specs.

### Spec coverage

| Requirement | Covered by |
|---|---|
| Staleness warning, 30-day threshold, always-on | Architecture → Staleness check |
| Warning reflects true data age even when using a week-old cache | `last_match_date` stored in cache payload, checked on every `load_model` call |
| `--halt-on-suspicious` flag, 10% fixed threshold | Architecture → `--halt-on-suspicious` |
| Hard block (not a flippable default) | Data flow step 4, Edge cases |
| Runbook covering ATP refresh, WTA refresh + 503 troubleshooting, retrain, before/after validation, suspicious-edge flag | Runbook section |
| Testable without new REPL test infrastructure | Self-Review → testable seam |

# Multi-Region / Multi-Bookmaker Odds — Design Spec

**Date:** 2026-08-21
**Status:** Approved for planning

## Goal

Today the odds pipeline is hardcoded to `regions=eu` and a single fixed
bookmaker (`DEFAULT_BOOKMAKER = "pinnacle"`, `src/odds_api.py`). This expands
coverage to multiple regions and selects the best available price per side
across an allow-listed set of bookmakers, instead of one fixed book. This is
the first of three candidate market-expansion directions discussed (more
regions/bookmakers, more bet types, more tours/circuits) — the other two are
out of scope for this spec (see chat context 2026-08-21).

## Non-goals

- No new bet types (totals, set handicaps) — still h2h only.
- No new tours/circuits (still ATP/WTA only).
- No best-of-region-then-book UI; the existing CLI/scanner/report surfaces
  are extended, not redesigned.

## Config (`.env` / `.env.example`)

- `ODDS_API_REGIONS` — comma-separated, default `eu,uk,us`. Passed directly
  as the `regions=` query param on every odds request.
- `ALLOWED_BOOKMAKERS` — comma-separated bookmaker keys (e.g.
  `pinnacle,bet365,williamhill`). Empty or unset means "allow every
  bookmaker the API returns for the configured regions."
- `ODDS_API_BOOKMAKER` (existing) and `--bookmaker` (existing CLI flag on
  `daily_scanner.py`) are kept as backward-compatible overrides: when set,
  they collapse the allow-list to that single bookmaker, reproducing today's
  fixed-book behavior exactly. This is a deliberate escape hatch for anyone
  already relying on the old flag/env var.

## Architecture

### `src/odds_api.py` — shared price-selection helper

```python
def _best_price(
    event: dict, allowed_bookmakers: Optional[set[str]]
) -> tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
```

Iterates `event["bookmakers"]`, skipping any whose `key` is not in
`allowed_bookmakers` (when the set is non-empty/non-None). For each side
(`home`/`away`) independently, tracks the highest price seen and which
bookmaker offered it. Returns `(odds_home, odds_away, bookmaker_home,
bookmaker_away)`, with `None`s if no allowed bookmaker quoted a side at all
(mirrors today's "bookmaker didn't quote it" `None` case).

**Independent-side selection is deliberate**: the best price for player A
and the best price for player B may come from different bookmakers (e.g.
Bet365 has the best price on A, Pinnacle on B). Both are recorded
separately — `bookmaker_a`/`bookmaker_b` are never assumed to be the same
book.

`find_match_odds` (event-list + player-name matching) and
`daily_scanner._extract_h2h_odds` (single already-matched event) both
delegate to `_best_price` instead of each re-implementing the same
bookmaker-loop, removing the current duplication.

`fetch_odds_events_by_key`/`fetch_odds_events` build the `regions=` query
param from `ODDS_API_REGIONS` (env, via `src.config`) instead of the
hardcoded `eu` literal.

### `MatchOdds` dataclass

Adds `bookmaker_a: str` and `bookmaker_b: str` alongside the existing
`odds_a`/`odds_b`/`matched_home`/`matched_away`.

### `daily_scanner.py`

- `DiscoveredMatch` adds `bookmaker_a: str`, `bookmaker_b: str`.
- `discover_matches`/`_extract_h2h_odds` thread the allow-list (built from
  `ALLOWED_BOOKMAKERS`, collapsed to a single-item set if
  `ODDS_API_BOOKMAKER`/`--bookmaker` is set) through to `_best_price` and
  populate the new fields on `DiscoveredMatch`.

### Logging (`value_analysis.py`, `calibration_audit.py`)

- `_LOG_FIELDS` gains `bookmaker_a`, `bookmaker_b` (placed next to
  `odds_a`/`odds_b`).
- `log_query`/`log_prediction_audit` gain `bookmaker_a`/`bookmaker_b`
  parameters, threaded from every call site (`daily_scanner.run_scan`,
  `daily_workflow.run`, `value_analysis.interactive_cli`).
- `_migrate_log_header_if_needed` backfills existing rows with
  `bookmaker_a = bookmaker_b = "pinnacle"` (the fixed book every historical
  row actually used), same pattern already used for
  `odds_a_source`/`odds_b_source` migration.

### Notification / reporting

- `notifier._picks_table_html` adds a "Bookmaker" column (shows
  `bookmaker_a`/`bookmaker_b` for whichever side was picked).
- `daily_workflow.run`'s `selected.append({...})` dict gains the bookmaker
  key for the picked side, consumed by the notifier.
- `settle_workflow._build_report`'s markdown table adds a "Bookmaker"
  column, read from the settled row's `bookmaker_a`/`bookmaker_b` (whichever
  matches the settled side).

### Interactive CLI (`value_analysis.py`)

`try_auto_odds`'s confirmation line (`Cuotas encontradas ({bookmaker_label})`)
switches from printing the fixed `ODDS_API_BOOKMAKER`/`DEFAULT_BOOKMAKER`
env value to printing the actual `bookmaker_a`/`bookmaker_b` returned by
`MatchOdds` for this match — since with independent per-side selection the
book may differ from what the user configured as a preference.

## Error handling

Unchanged failure semantics: no allowed bookmaker quoting a side is treated
identically to today's "bookmaker didn't quote it" — `None` from
`_best_price`, which propagates to "no match" in the scanner (skip, printed
reason) and "no auto-fill" in the interactive CLI (falls back to manual
entry). No new exception types.

## Testing

`tests/test_odds_api.py`:
- `_best_price`: allow-list filtering, independent max-per-side selection
  (different winning bookmaker per side), empty/unset allow-list means "all
  bookmakers", no allowed bookmaker quotes either side → `(None, None,
  None, None)`.
- `find_match_odds`/`_extract_h2h_odds`: updated to assert on the new
  `bookmaker_a`/`bookmaker_b` return values instead of the old
  single-bookmaker path.
- `ODDS_API_BOOKMAKER`/`--bookmaker` override collapses the allow-list to
  one bookmaker (regression test for the backward-compat path).

`tests/test_daily_scanner.py`, `tests/test_value_analysis.py`,
`tests/test_calibration_audit.py`: updated fixtures/assertions for the new
`bookmaker_a`/`bookmaker_b` fields on `DiscoveredMatch` and the CSV log
schema.

`tests/test_notifier.py`, settle-workflow report tests (if present): assert
the new "Bookmaker" column renders.

Full suite (509 existing tests) must stay green; new tests added for the
above, no real network calls (same mocking approach as the existing odds_api
tests).

## Open risk (verified 2026-08-21 against real API, resolved as a known cost)

Confirmed empirically with a real `ODDS_API_KEY` against `tennis_atp_cincinnati_open`:
`regions=eu,uk,us` costs **3 credits per call** (`x-requests-last: 3`), i.e.
exactly 1 credit per region — the `regions × markets` scaling was correct,
not just a theoretical concern. The `/v4/sports` discovery call itself is
free (`x-requests-last: 0`).

Quota state at verification time: 396/500 monthly requests remaining
(104 used before this check). A daily automated scan across N active
tournaments now costs `3N` credits/day instead of the old `N` (single
`eu` region), i.e. roughly a 3x increase in daily quota burn. With a
2-tournament day (a common case — one ATP + one WTA event) that's 6
credits/day (~180/month at that rate), leaving headroom under the 500/month
free tier under normal single-tournament-per-day operation, but multi-
tournament days (e.g. 3+ concurrent ATP/WTA events during overlapping
tour weeks) could approach the ceiling faster than before. Not a blocker —
no code change needed for this spec — but worth monitoring `x-requests-remaining`
in `daily_workflow.py`'s output if scans start being skipped for API-key
reasons, and worth revisiting (narrower default region list, e.g. dropping
to `eu,uk` if `us` rarely surfaces better prices in practice) if quota
pressure becomes real.

## Spec coverage check

| Requirement | Covered by |
|---|---|
| `ODDS_API_REGIONS` config, multi-region requests | Config, Architecture |
| `ALLOWED_BOOKMAKERS` allow-list, empty = all | Config, `_best_price` |
| Best price selected independently per side | `_best_price` design note |
| `bookmaker_a`/`bookmaker_b` on `MatchOdds`/`DiscoveredMatch` | Architecture |
| Backward-compat via `ODDS_API_BOOKMAKER`/`--bookmaker` | Config |
| CSV log + migration of old rows to "pinnacle" | Logging section |
| Email + settlement report show bookmaker | Notification/reporting |
| Interactive CLI shows actual bookmaker used | Interactive CLI section |
| Tests for new logic, full suite green | Testing |
| Quota-cost risk flagged | Open risk |

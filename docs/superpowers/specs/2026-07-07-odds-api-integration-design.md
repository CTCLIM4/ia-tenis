# Odds API Integration — Design Spec

**Date:** 2026-07-07
**Status:** Approved for planning

## Goal

`src/value_analysis.py` currently requires the user to type in decimal odds by hand for every match. This adds automatic odds lookup via [The Odds API](https://the-odds-api.com) (free tier, 500 req/month) as a **default-filling assist** inside the existing interactive CLI — it never replaces or blocks the manual flow, it just pre-fills it when possible.

## Non-goals

- No "scan the board" mode that lists all upcoming matches with value bets automatically. Out of scope for this spec — a possible future iteration, not this one.
- No new fuzzy-matching dependency (e.g. `rapidfuzz`). The Odds API returns full player names for both ATP and WTA, matching what the user already types, so simple normalization (lowercase, strip accents/punctuation, compare surname) is sufficient.
- No multi-bookmaker aggregation (best-of or average). One fixed configured bookmaker only, so the odds shown are actually bettable by the user on that book.

## Architecture

### `src/odds_api.py` (new module)

- `fetch_odds_events(tour: str) -> list[dict]` — calls `GET https://api.the-odds-api.com/v4/sports/tennis_{atp,wta}/odds/` with `apiKey`, `regions=eu`, `markets=h2h`, `oddsFormat=decimal` via `urllib` (same HTTP approach as `scripts/download_data.py` — no new HTTP library dependency). Returns the raw list of event dicts (`home_team`, `away_team`, `commence_time`, `bookmakers`).
- `_normalize_name(name: str) -> str` — lowercase, strip accents and periods, collapse whitespace.
- `find_match_odds(events: list[dict], player_a: str, player_b: str, bookmaker: str = "bet365") -> Optional[MatchOdds]` — normalizes `player_a`/`player_b` and each event's `home_team`/`away_team`, matches by surname in either order. Returns `None` if no event matches. If an event matches but the configured bookmaker did not quote it, also returns `None` (never substitutes a different bookmaker).
- `MatchOdds` (dataclass): `odds_a: float`, `odds_b: float`, `matched_home: str`, `matched_away: str` — the raw names the API used, so the caller can show the user what was matched before they accept it.

### Event cache (`data/odds_cache/{tour}.json`)

One JSON file per tour: `{"fetched_at": ISO timestamp, "events": [...]}`. Refetched when missing or older than `ODDS_API_CACHE_MINUTES` (default 15, env-configurable). A single fetch covers every match looked up during a session (or across sessions within the TTL window), keeping API usage low relative to the free tier's monthly cap. Added to `.gitignore` alongside `data/model_cache/`.

### Config (env vars, no new file/format)

- `ODDS_API_KEY` — required to attempt auto-fetch. If unset, auto-fetch is silently skipped (no key configured is not an error state) and the flow behaves exactly as it does today.
- `ODDS_API_BOOKMAKER` — default `"bet365"`.

## Data flow inside `interactive_cli()`

1. User enters `player_a`, `player_b` (existing WTA disambiguation logic runs first, unchanged).
2. `try_auto_odds(tour, player_a, player_b) -> Optional[MatchOdds]`:
   - Returns `None` immediately if `ODDS_API_KEY` is unset.
   - Loads the cached event list for `tour`, refetching via `fetch_odds_events` if stale/missing.
   - Calls `find_match_odds`.
3. If a `MatchOdds` is found: print a one-line confirmation (e.g. `Cuotas encontradas (bet365): Sinner vs Djokovic`) showing the matched names, then pass `odds_a`/`odds_b` into `_ask_odds` as the pre-filled default — same "Enter = accept, type to override" pattern already used for auto-filled rankings.
4. If not found (no key, no match, bookmaker didn't quote it, or a network/API error occurred): `_ask_odds` behaves exactly as it does today, no default, no extra prompt text.
5. `log_query` gains `odds_a_source` / `odds_b_source` fields (`auto` / `manual`) in `_LOG_FIELDS`, mirroring the existing `rank_a_source`/`rank_b_source` pattern, so `value_bets_log.csv` can be audited for which picks used live odds vs typed ones.

## Error handling

All failure modes inside `try_auto_odds` (timeout, HTTP error, invalid key, rate limit / 429, malformed JSON) are caught in one place and treated identically: fall back to manual entry. A short warning is printed **once per CLI session** (not once per match) to avoid repeated noise when the API is unreachable for an entire session. The manual-entry path must never be blocked or altered by an API failure — auto-fetch is strictly additive.

## Testing

`tests/test_odds_api.py`, no real network calls (events passed as in-memory dicts, `fetch_odds_events` mocked where cache-refresh logic is under test):

- `_normalize_name`: accents, periods, casing, extra whitespace.
- `find_match_odds`: direct order match, reversed home/away order match, no match, event matches but configured bookmaker absent.
- Cache read/write round-trip; TTL expiry forces a refetch (mocking `fetch_odds_events`), fresh cache does not refetch.

## Spec coverage check

| Requirement | Covered by |
|---|---|
| Auto-fetch odds inside existing 1-by-1 CLI flow (not a new scan mode) | Data flow section |
| Fixed bookmaker, not best-of/average | `find_match_odds` bookmaker param |
| Env var for API key, no new config file format | Config section |
| Fallback to manual entry on any failure, never blocks | Error handling section |
| No new fuzzy-matching dependency | Architecture section, non-goals |
| Auditable in the CSV log which odds were auto vs manual | Data flow step 5 |

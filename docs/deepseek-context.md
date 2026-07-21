# ia-tenis — Context Briefing

## What this is
ML tennis match-prediction project at D:\ia-tenis. Elo ratings (general +
surface-specific, blended by match-count weight) + engineered features
(rank_diff, form_diff, surface_form_diff, h2h_rate, rest_diff) feeding a
Logistic Regression, trained on the full mirrored history (each match
duplicated with sides flipped for symmetry). Separate models for ATP and
WTA. Sibling project: ia-futbol (Dixon-Coles + Elo + Poisson for football).

## Data sources
- ATP: stats.tennismylife.org API (migrated from a stale git-clone source
  in July 2026).
- WTA: tennis-data.co.uk, per-year xlsx/xls, 2007-present.
- atptour.com is NOT fetchable (403 anti-bot) — draws/scores come from
  tennisexplorer.com, sofascore, 10sballs, flashscore, Wikipedia instead.

## Model details worth knowing
- Elo yearly decay: 0.75 retention applied when a player's match-year
  jumps — this bit us today (see gotchas below).
- Calibration shrinkage: raw LR probabilities >90% or <10% get compressed
  60% toward the boundary (documented overconfidence in that range).
- Model cache expires after 7 days; a fresh cache does NOT mean fresh
  data — there's a separate "last_match_date" staleness check (warns if
  >30 days old) that's a known blind spot (misses a stale dataset during
  a live tournament, since 30 days can still be "before this week's
  results").

## Value-betting workflow
- edge = model_prob - implied_prob (from decimal odds)
- ev = model_prob*(odds-1) - (1-model_prob)
- kelly = edge/(odds-1), capped at 5%
- Discipline rules in use: edge >=3% & Kelly>0 → log as a tracked bet;
  edge >10% → treat as suspicious, do NOT auto-log, get human
  confirmation (large edges are usually stale data or a name-matching
  bug, not free money); edge <3% → pass, don't log.
- Odds: The Odds API integration exists in code but no API key is
  configured in this environment — odds are entered manually.
- Everything logs to data/value_bets_log.csv. Tracking only, no real
  stakes.

## Hard-won data-quality gotchas
- tennis-data.co.uk can silently revise *historical* years, not just add
  current-season matches — backtests need a frozen snapshot to be
  reproducible.
- Found & fixed today (2026-07-21): one WTA row (Iasi Open final,
  Sherif def. Badosa, retired) had its date typo'd as 2029 instead of
  2026 in the source spreadsheet. This spuriously triggered the Elo
  yearly-decay logic for both players in that match, corrupting their
  ratings until caught by cross-checking last_match_date against known
  results.
- WTA names are abbreviated ("Sabalenka A."); sibling disambiguation
  (e.g. Karolina vs Kristyna Pliskova) resolved by picking the
  "Lastname X." key with the most career matches.
- Rule: never let real match results leak into analysis for a pick
  that's still pending/being built.

## Today's session (2026-07-21)
Refreshed & retrained both models, verified sanity checks against known
results (Rublev/Darderi Båstad final, Tsitsipas/Collignon Gstaad final).
Pulled real R32 pairings for ATP Kitzbühel, ATP Estoril, WTA Hamburg, WTA
Prague. Applied a surface-experience filter (n_clay>=10 both sides for
clay events, n_hard>=10 for Prague which is hard court) to 35 matchups →
19 passed. Got manual odds for 9; ran value analysis:
- Logged (edge 3-10%, Kelly>0): Trungelliti, Halys, Baez S., Collignon,
  Dzumhur.
- Passed (edge <3%): Shevchenko.
- Flagged suspicious (edge >10%, NOT logged, pending review): Molcan
  (+17.2%), Gaubas (+13.6%), Burruchaga (+24.8% — model gives Wawrinka
  only 36% win prob despite his huge career clay Elo; likely not
  capturing his 2026 age-related decline, 7-14 record, hip issues).
- 10 skipped for lack of odds.

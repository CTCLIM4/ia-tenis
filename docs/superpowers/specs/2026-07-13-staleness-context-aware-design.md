# Context-Aware Staleness Threshold — Design Spec

**Date:** 2026-07-13
**Status:** Draft — design choices below need confirmation before an implementation plan is written (this document is the "confirm design" half of the two-step structural-decision process; the "confirm topic" half — should staleness become tournament-aware at all — was already agreed when this spec was commissioned).

## Goal

On 2026-07-13, diagnosing the WTA value-bet pick logged for Coco Gauff vs Karolina
Muchova (Wimbledon SF, 2026-07-09) found that it was generated from a model whose
dataset's newest match was 2026-06-27 — **zero Wimbledon matches were in the
training data**, even though Wimbledon started 2026-06-29 and the pick concerned
two players actively competing in it. The existing staleness check
(`_check_staleness` in `src/value_analysis.py`, `STALENESS_WARNING_DAYS = 30`,
see `docs/runbooks/data-refresh-and-staleness.md` §1) did not fire, because 16
days old is genuinely not stale under a threshold calibrated for the normal
between-tournament cadence of the tour.

This spec designs a **context-aware staleness threshold**: a tighter effective
window when a tournament relevant to the pick is actively underway, the current
30-day window otherwise. It does not implement anything — the output is a design
for the confirm-design step, per standing process (structural decisions get a
topic confirmation and a separate design confirmation, not one bundled ask).

## Non-goals

- No implementation in this pass. Zero code changes.
- No auto-refresh or auto-retrain triggered by a tightened threshold — mirrors
  the existing non-goal in `docs/superpowers/specs/2026-07-08-staleness-halt-runbook-design.md`:
  staleness checks only warn (or, opt-in, hard-block); refreshing the data stays
  a manual, deliberate action.
- No change to `CACHE_MAX_AGE_DAYS` (pickle cache expiry) — a separate, already-existing
  concept (age of the cache file, not age of the match data); unaffected by this work.
- No implementation of the WTA download size-comparison fix
  (`project-wta-download-size-bug` in project memory) — tracked separately, but
  see "Interaction with the WTA download bug" below; likely a sequencing dependency.
- Not designing a general-purpose tournament-calendar service. Scoped narrowly to
  "is there a tournament active right now that's relevant to today's staleness
  check for this tour" — not a full ATP/WTA schedule feature for other uses.

## Definition of "active tournament"

Three candidate sources, evaluated for how they'd detect "a tournament is in
progress or starts within N days":

**Option A — TML `ongoing_tourneys.csv` (ATP only).** Already fetched on every
`scripts/download_data.py` run and merged into the current year's ATP file (see
`docs/superpowers/specs/2026-07-08-tml-api-migration-design.md`, pre-flight
notes: confirmed live-updated, e.g. contained `tourney_id = 2026-540`
(Wimbledon), `tourney_date` 20260629–20260707, 122 rows, on 2026-07-08).
*Pros:* zero new dependency, data already downloaded every refresh, genuinely
live. *Cons:* ATP-only — `tennis-data.co.uk` (WTA's source) ships static
per-year files with no equivalent live in-progress feed. A Slam like Wimbledon
is shared by both tours, so an ATP-sourced signal *can* answer "is a Slam
active" for WTA purposes too, but only for tour-agnostic events (Slams), not
WTA-only events (e.g. an active WTA 500 wouldn't show up here at all).

**Option B — hardcoded ATP/WTA calendar** (start/end dates per tournament,
maintained by hand or scraped once per season). *Pros:* works uniformly for
both tours, doesn't depend on either source's own freshness. *Cons:* ongoing
maintenance burden (roughly 60+ events/year across both tours), doesn't
self-correct for schedule changes (rain delays, format changes, tournament
relocations), another artifact that can silently drift out of date — the same
category of problem this spec exists to prevent, just moved one level up.

**Option C — derive "active" from the dataset's own recent match density**
(e.g., "are there a cluster of matches in the last few days involving this
tour's ranked players"). *Pros:* no external dependency at all. *Cons:*
**disqualifying circularity** — the dataset being stale is exactly the failure
mode this spec exists to catch; a signal computed from the stale dataset itself
cannot detect that the dataset is stale. Ruled out.

**Recommendation to confirm:** Option A for ATP (already available, zero
marginal cost) plus a minimal hardcoded Slam calendar (4 fixed date ranges per
year: Australian Open, Roland-Garros, Wimbledon, US Open — extremely low
maintenance, changes essentially never) to cover cross-tour relevance for WTA.
Explicitly **not** recommending Option B's full 250-through-1000-level
hardcoded calendar — the maintenance cost is disproportionate to the risk this
spec is mitigating, which is specifically the "no matches from the Slam my pick
is about" blind spot, not general WTA-500 freshness.

## Contextual staleness rules

- **No relevant active tournament:** unchanged — `STALENESS_WARNING_DAYS = 30`.
- **Relevant active tournament, this tour:** tightened threshold, tentatively
  **3 days**.
- **"Relevant" — open question, three candidate definitions:**
  - **(a) narrow — Slams/Masters/WTA-1000-or-above only.** The highest-stakes,
    most-liquid events; smaller 250-level events rarely justify a same-day
    freshness bar. Simplest to implement given Option A's Slam-calendar overlap.
  - **(b) broad — any active tournament on the same tour**, regardless of level.
    More conservative, but likely over-triggers during e.g. the current ATP
    clay-250 swing (Båstad/Umag/Gstaad, all live this week) for picks that don't
    even involve those events' players.
  - **(c) player-scoped — only when the specific matchup being priced includes a
    player who's actually in that active tournament's field.** Most precise (it
    would have caught the actual Gauff/Muchova gap and *not* fired for
    unrelated picks), but requires cross-referencing the pick's two player names
    against the active tournament's entry list at prediction time — meaningfully
    more implementation surface than a pure tour/date check.
  - **Recommendation:** (a) for v1. It's the cheapest to build correctly given
    the Option A data already available, and it directly covers the scenario
    that motivated this spec (a Slam). Revisit (c) only if (a) demonstrably
    leaves a real gap in practice — don't build the more precise, more complex
    version speculatively.

## Guardrail behavior at threshold crossing

Reuse the existing two-tier pattern from `--halt-on-suspicious`
(`SUSPICIOUS_EDGE_THRESHOLD`, `docs/superpowers/specs/2026-07-08-staleness-halt-runbook-design.md`)
rather than inventing a third posture:

- **Always-on:** print an explicit warning — `_check_staleness`'s existing
  message, extended to say *which* rule fired (e.g. "torneo activo detectado
  (Wimbledon) — umbral estricto de 3 dias en vez de 30" vs. today's generic
  message), so the operator can tell at a glance whether they're looking at the
  default or the tightened rule.
- **Opt-in hard block:** new `--strict-staleness` flag (naming mirrors
  `--halt-on-suspicious`), same "opt-in, not a flippable default, rerun without
  the flag to override" posture as the suspicious-edge gate — refuses the
  log-save prompt when the *contextual* threshold (not the base 30-day one) is
  exceeded.
- **No fail-hard-by-default.** Matches the project's existing philosophy
  established in the halt-runbook spec: warn by default, block only when the
  operator explicitly opts in.

## Interaction with the WTA download size-comparison bug

Likely a **sequencing dependency, not just a related item.** The whole point of
tightening the threshold is that a stale dataset gets caught sooner — but
today, re-running `scripts/download_data.py` for WTA silently no-ops on the
current year's file regardless of how stale it actually is (`project-wta-download-size-bug`
in project memory: `_download_wta_year` skips on local-file-existence, never
compares against the remote). Shipping a 3-day contextual threshold without
that fix would just make the warning fire more often while leaving the
operator with the same manual workaround used in the 2026-07-13 session
(rename/delete the local file by hand before re-running the script) — the
guardrail would be more sensitive without the refresh path actually being more
capable. Recommend the WTA download fix lands first or in the same
implementation pass, and that the eventual implementation plan for this spec
names it as a blocking dependency rather than a footnote. The WTA download fix
itself stays blocked on network access per its own memory entry (TDD needs a
live remote to test against) — that blocker transitively blocks this feature's
full rollout for WTA specifically, though the ATP half (Option A) has no such
dependency and could ship independently if the two are ever decoupled.

## Test plan

- Parametrized tests over `(has_active_tournament, days_stale)` combinations,
  mocking `date.today()` and `last_match_date`:
  - no active tournament, 35 days stale → warn (today's existing behavior,
    regression-checked)
  - no active tournament, 10 days stale → no warn (existing behavior)
  - active Slam, 5 days stale → warn under the new contextual rule, though it
    would **not** have warned under the old 30-day-only rule (this is the
    specific case this spec exists to fix — a dedicated test asserting the
    *difference* in behavior, not just the new behavior in isolation)
  - active Slam, 2 days stale → no warn (inside the tightened 3-day window)
  - active non-Slam ATP 250, 5 days stale → behavior depends on which of
    (a)/(b)/(c) is chosen for "relevant"; needs one explicit test per the
    chosen definition once confirmed
- Mock the active-tournament source (presence/absence of a relevant row in
  `ongoing_tourneys.csv`, or the hardcoded Slam-calendar lookup) — no test
  should hit the network or read real cache/data files, consistent with the
  existing style in `tests/test_value_analysis.py`.
- `--strict-staleness` hard-block: extract a small pure function (mirroring
  `_should_halt_on_suspicious_edge` from the 2026-07-08 spec's Self-Review) so
  the block decision is unit-testable without touching the REPL-driven
  `interactive_cli`.
- Regression: all existing `_check_staleness`-related tests must keep passing
  unmodified for the no-active-tournament path — this feature is additive, not
  a replacement of the base rule.

## Rollback strategy

- Single cohesive change surface: `src/value_analysis.py` plus one small new
  lookup module or constant for the Slam calendar — no data-schema change to
  `data/value_bets_log.csv`, no change to the shape of the existing model-cache
  pickle payload. Revertible with a single `git revert`, no data migration.
- `--strict-staleness` is additive and opt-in — even a buggy "active
  tournament" detection has a blast radius limited to an incorrect *warning
  message*; nothing about what gets logged changes unless the operator
  explicitly passes the flag, same safety posture as `--halt-on-suspicious`.
- If Option A's `ongoing_tourneys.csv`-based detection ever breaks (TML API
  stops shipping it, or a future change to the merge logic in
  `scripts/download_data.py` drops it before `load_model` can read it), the
  hardcoded Slam-calendar fallback keeps the feature working in degraded form
  (Slam coverage only, no live non-Slam ATP detection) rather than silently
  reverting to "always 30 days" with no indication anything changed — the
  degraded path should print its own distinct message so the gap is visible,
  not silent. Exact wording/trigger for that fallback message is left for the
  implementation plan, not decided here.

## Open questions to resolve before an implementation plan is written

1. Confirm Option A + Slam-calendar as the "active tournament" source (vs. B or
   a partial combination not explored above).
2. Confirm "relevant" = (a) Slams/Masters/WTA-1000+ only, vs. (b) any active
   tournament on the tour, vs. (c) player-scoped.
3. Confirm the 3-day tightened threshold specifically (vs. some other number —
   3 was chosen loosely as "roughly one round's worth of matches," not derived
   from data).
4. Confirm sequencing: does this feature wait for the WTA download bug fix, or
   ship ATP-only first with WTA following once network access allows the
   download fix to be built and tested?

# Calibration / PASS Engine Persistence & Audit — Design Spec

**Date:** 2026-07-25
**Status:** Confirmed — all open design questions resolved 2026-07-25 (all the recommended option). Ready for an implementation plan. This is item 1 of the two-part persistence TODO from [[project-pending-persistence-design]] (item 2, dataset snapshots, shipped 2026-07-24/25).

## 1. How probabilities are calibrated today

`predict_match()` (`src/value_analysis.py:654-656`):

```python
X       = np.array([[feats[c] for c in FEATURE_COLS]])
p_a_raw = float(clf.predict_proba(X)[0, 1])
p_a_cal = apply_shrinkage(p_a_raw)
```

`p_a_raw` is the sigmoid output of the trained `Pipeline(StandardScaler, LogisticRegression)` — already the product of a scaled LR (not "raw" in the sense of unscaled features, "raw" only relative to the next step).

`apply_shrinkage()` (`:74-84`) is the entire calibration mechanism:

```python
_SHRINK_HI   = 0.90
_SHRINK_LO   = 0.10
_SHRINK_RATE = 0.60

def apply_shrinkage(p: float) -> float:
    if p > _SHRINK_HI:
        return _SHRINK_HI + (p - _SHRINK_HI) * (1 - _SHRINK_RATE)
    if p < _SHRINK_LO:
        return _SHRINK_LO - (_SHRINK_LO - p) * (1 - _SHRINK_RATE)
    return p
```

A fixed, piecewise-linear post-hoc correction: predictions outside `[0.10, 0.90]` get 60% of their excess-beyond-the-boundary compressed back toward the boundary; the `[0.10, 0.90]` range passes through unchanged. The comment above the constants attributes them to "Section 13: reliability analysis, ATP+WTA 2016-23" — **no script, notebook, or document in this repo reproduces that study**. The three numbers are effectively a black box: correct as far as anyone can currently verify, but not re-derivable or auditable from the codebase itself, and unchanged since introduced even though the model has been retrained/extended repeatedly since (Elo decay features, StandardScaler pipeline, snapshot pinning — none of which triggered a recalibration check).

**This is itself the first finding relevant to "what needs auditing": the calibrator is not versioned, not tied to the model's own version, and not reproducible.**

## 2. What needs to be persisted/audited, and why

### 2.1 Calibrator parameters at logging time

`log_query()` (`:721-779`) writes a single boolean, `shrinkage_applied` (`:734`, `:767`) — whether shrinkage fired, not *what it did*. If `_SHRINK_HI`/`_SHRINK_LO`/`_SHRINK_RATE` are ever retuned (the whole point of eventually auditing calibration quality), every historical log row becomes ambiguous: you can't tell which constants actually produced that row's `p_a_cal` from `p_a_raw` without also knowing the exact code version active at logging time.

### 2.2 Model identity/version

`_LOG_FIELDS` (`:681-697`) has no field tying a row to *which trained model* produced it. The model retrains on cache expiry (≤7 days), on `--retrain`, or can be pinned via `--snapshot {id}` (shipped 2026-07-24) — the same matchup logged on two different days could reflect materially different models with zero traceability today. `src/data/snapshots.py` already has `_current_git_commit()` for exactly this kind of stamping; nothing analogous exists for the value-bet log.

### 2.3 PASS / declined observations — the core gap named in [[project-pending-persistence-design]]

Traced the full CLI flow (`interactive_cli`, `:1080-1122`): there is **no automatic PASS category in code**. `calculate_value()` sets `has_value = edge > 0` and `_print_prediction()` prints "Sin value en ninguno de los lados. Pasar este partido." when neither side clears zero edge — but this is display-only. The save decision is a single uniform prompt (`_should_log_prediction`, `:128-140`) regardless of edge size; a below-threshold match gets the *exact same* "Guardar en log? (s/n)" prompt as a real value bet, and if the user says "n" (or the session it was mentioned in was just chat narration, as in the 2026-07-21 DeepSeek session's "Passed (edge <3%): Shevchenko" — never written to any file), **nothing is persisted.** `value_bets_log.csv` today is a biased sample (only rows a human chose to log) — you cannot build a reliability diagram or track Brier score over time from it, because you don't have the denominator (every prediction the model actually made).

### 2.4 Outcome ground-truth (adjacent finding, not core to calibration persistence)

`log_query` always writes `result="pending"`, `profit=""` (`:769-770`) — no code in this repo ever writes anything else. Yet the real `data/value_bets_log.csv` has rows with `result` values `A_win`, `B_win`, `excluded`, and `status` value `test_never_played` — none of which any function produces. **This confirms outcome-filling is an entirely manual, out-of-band process today** (presumably direct CSV edits). Not this spec's problem to solve, but worth naming: any calibration audit built on top of the log inherits this manual, unaudited step as its ground-truth source.

## 3. Technical proposal — two options, need your decision

Both options add the same new data (calibrator params, model identity) — they differ only in **where PASS/declined observations go**.

### Option A — Extend `value_bets_log.csv` itself

Log *every* prediction evaluated in a session (not just user-confirmed ones), with a new `decision` field: `logged` / `passed_low_edge` / `passed_user_declined` / `blocked_suspicious` / `invalid_missing_elo`. Add `model_commit`, `model_snapshot_id`, `shrink_hi`, `shrink_lo`, `shrink_rate` columns to `_LOG_FIELDS`.

- **Pros:** single source of truth, no new file to reconcile.
- **Cons:** changes the *meaning* of an existing, already-in-production file (17 real rows, with manual `result`/`profit` edits already layered on top — migration/compatibility risk via `_migrate_log_header_if_needed`); changes default CLI behavior — every declined/low-edge match now leaves a permanent on-disk trace, which is a verbosity/privacy shift the user should decide on explicitly, not something to default into.

### Option B — New parallel append-only audit log (recommended)

New file (e.g. `data/prediction_audit_log.csv`), logging *every* prediction the model computes — logged, passed, blocked, invalid — with the same new fields (calibrator params, model identity) plus a `decision` column. `value_bets_log.csv` stays **exactly as it is today**: schema unchanged, semantics unchanged ("bets I decided to track"), zero migration risk.

- **Pros:** additive only, matches the "don't touch what's already working" pattern the snapshot feature just followed; cleanly separates "audit trail of what the model said" from "record of bets I'm tracking."
- **Cons:** two files if you ever want a unified view (a logged pick appears in both); more surface than Option A.

**Recommendation: Option B**, for the same reason snapshots landed as a pure addition rather than a retrofit — lower risk, and the two concepts (bet-tracking vs. calibration audit) are genuinely different questions with different audiences (you, vs. a future you doing a reliability study).

## 4. Design decisions (all confirmed 2026-07-25 — recommended option chosen in every case)

1. **Option A vs. B.** **CONFIRMED: Option B** — new parallel `data/prediction_audit_log.csv`, `value_bets_log.csv` untouched.
2. **Auto-log every prediction, or keep a confirm step?** **CONFIRMED: silent auto-log**, no prompt — every prediction the model evaluates is written to the audit log regardless of the separate `value_bets_log.csv` save decision. This is a real default-behavior change (every evaluated match now leaves a trace on disk) — noted, confirmed, intentional.
3. **Model identity.** **CONFIRMED: git commit hash**, reusing `_current_git_commit()`'s pattern (move it to a shared location both `src/data/snapshots.py` and the audit-log code can import, rather than duplicating it — implementation-plan detail). Known accepted gap: a live (non-pinned) model's *data* version isn't captured even for the live path (only `model_snapshot_id` covers data version, and only when `--snapshot` is used) — out of scope for this pass, flagged as a possible future follow-up (stamp `last_match_date` too), not blocking.
4. **Calibrator params.** Raw literal values per row (`shrink_hi`, `shrink_lo`, `shrink_rate`) — no versioned registry. (Not separately asked — adopting the stated recommendation, consistent with the pattern of accepting recommended defaults throughout this design pass; YAGNI until a second calibration study actually happens.)
5. **Audit-log granularity.** Log only once odds are entered (complete rows; matches `log_query`'s existing all-fields-populated convention). Abandoned/Ctrl+C'd sessions before odds entry are not captured. (Adopting stated recommendation — simpler, no partial-row schema needed for v1.)
6. **Suspicious-edge-blocked matches.** **CONFIRMED: yes**, `decision="blocked_suspicious"` rows go into the audit log even though they're never eligible for `value_bets_log.csv`.
7. **File format.** CSV, matching `value_bets_log.csv`'s existing format/tooling. (Adopting stated recommendation — no reason to diverge, all fields are flat.)

## 5. TDD test plan sketch (once the above is confirmed)

- `_should_log_prediction`-style pure function for the new `decision` classification (logged / passed_low_edge / passed_user_declined / blocked_suspicious / invalid_missing_elo) — testable without touching the CLI's `input()` loop, same pattern already used for the existing save-gate logic.
- A `log_prediction_audit()` (or similar) function, tested the same way `log_query()` already is (`tests/test_value_analysis.py::TestLogQueryStatus` is the existing precedent): write a row, read it back, assert every new field (`model_commit`, `model_snapshot_id`, `shrink_hi/lo/rate`, `decision`) is correct — including the `model_snapshot_id=None` case for a live (non-pinned) model.
- Regression test: `value_bets_log.csv`'s schema/behavior is provably unchanged (Option B) — reuse/extend the existing `TestLogQueryStatus`/`TestLogQueryOddsSource` classes unmodified as the regression guard.
- A test proving the audit log captures a `blocked_suspicious` row that `value_bets_log.csv` never gets (the specific gap named in open question 6).
- A test for `model_commit` capture, mocking `_current_git_commit()`-equivalent the same way `tests/test_snapshots.py` already does.

## Non-goals (this pass)

- Not solving the outcome/`result`/`profit` manual-fill problem (§2.4) — real, but a separate piece of work.
- Not building a reliability-diagram/Brier-over-time *analysis* tool — this spec is about capturing the data such an analysis would need, not building the analysis itself.
- Not retroactively backfilling audit data for predictions made before this ships — starts capturing from whenever it lands, no historical reconstruction attempted.

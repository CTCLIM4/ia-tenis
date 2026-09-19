# ia-tenis Project Context

## Overview
Tennis match prediction model (ATP/WTA/Davis Cup) with Elo + features + logistic calibration, an interactive value-bet analyzer, an automated daily scanner, and a visualization dashboard.

## Stack
- Python 3.14.5, pandas 3.0.3, numpy 2.5.0, sklearn 1.9.0, scipy 1.18.0, matplotlib 3.10
- pytest 9.1.1, 694 tests all passing
- Key modules: `src/models/elo.py`, `src/features/engineering.py`, `src/features/decay.py`, `src/backtest/walkforward.py`, `src/data/loader.py`, `src/value_analysis.py`, `src/calibration_audit.py`, `src/calibration_metrics.py`, `src/dashboard.py`, `src/daily_scanner.py`, `src/surface_resolver.py`, `src/player_matcher.py`, `scripts/backup_logs.py`

## Architecture
- **EloSystem** (`src/models/elo.py`): General + surface Elo ratings with yearly decay, K-factor based on match count
- **FeatureBuilder** (`src/features/engineering.py`): Recent win rates, momentum (3/5-match), H2H rate + trend, surface win-rate trend, rest days
- **EloHistoryTracker + decay features** (`src/features/decay.py`): Rolling Elo, age/rust/fatigue/surface-transition multipliers
- **Pipeline** (`src/pipeline.py`): Load → build features → walk-forward backtest, tour-agnostic (atp/wta/davis)
- **Value Analysis** (`src/value_analysis.py`): LR model + shrinkage calibration + odds API + interactive CLI (`--wta`/`--davis`)
- **Backtest** (`src/backtest/walkforward.py`): Walk-forward LR evaluation with mirror rows; `return_predictions=True` exposes raw (prob, outcome) pairs (including a mirror-balanced `probs_calibration`/`y_true_calibration` pair) for calibration analysis
- **Calibration metrics** (`src/calibration_metrics.py`): Reliability/calibration-curve analysis against the large backtest sample (never the small real-bet log — see `docs/metrics/2026-09-18-calibration-investigation.md`)
- **Dashboard** (`src/dashboard.py`): Backtest accuracy by year, calibration curves, LR feature importance, value-bet history — PNGs to `data/dashboard/` (gitignored, regenerate with `python -m src.dashboard`)
- **Daily scanner** (`src/daily_scanner.py`): Discovers live matches via The Odds API, no staleness gate (prints warnings only), `--auto-save`, `--in-play` (opt-in, loud warning — model has no live-match-state signal), rate-limit retry with backoff. Before logging a value bet, calls `check_existing_log_entry()` (`src/value_analysis.py`): a genuinely new match logs normally, odds within 2% of an already-logged pending match is skipped as a duplicate, odds moved >2% updates that row in place (`update_log_entry()`) instead of appending a second one, and a match whose row already has a real result is never touched. Added 2026-09-19 after a re-scan of the same still-pending match logged it twice, which would have double-counted the position once it settled. That fix only guards future scans — it doesn't retroactively clean history, and indeed the two pre-existing Stearns/Jovic rows it was built to explain (odds 3.80 and 3.90) were still both present in the real `value_bets_log.csv` until the 2026-09-19 production verification found and removed the stale one (kept the fresher row); `tests/test_value_bets_log_integrity.py` now asserts the real log has no duplicate-identity pending rows so this can't silently reappear. Separately noted but not fixed (cosmetic, doesn't affect logging correctness): `classify_audit_decision()`'s `passed_user_declined` label is a catch-all for "had value but wasn't logged" and also fires when the scanner auto-skips a duplicate — the audit log doesn't currently distinguish "user declined" from "auto-skipped duplicate."
- **Daily workflow** (`scripts/daily_workflow.py`): Stricter automation wrapper — DOES hard-exclude a tour on any staleness (unless `--live-tournament` opts out of that), auto-logs + emails + git-pushes
- **Runner** (`scripts/run_prediction.py`): download → pipeline → retrain (non-interactive `load_model(tour, retrain=True)`, never the interactive CLI) → scanner --auto-save, `--log` writes `data/logs/daily_{date}.log`
- **Scheduler** (`scripts/schedule_daily.py`): Builds/removes a Windows Scheduled Task for `run_prediction.py`. Registered on the production machine as `ia-tenis-daily-prediction`, daily 08:00 AM. **2026-09-19 production incident**: the very first scheduled run failed immediately (exit code 2) because `create_task()` baked a `--auto-save` flag into the `/tr` command, but `run_prediction.py`'s argparse has no such flag (auto-save is hardcoded internally in step 4/4) — so schtasks ran a command argparse rejected before any pipeline step executed, silently, since a scheduled task's stderr goes nowhere. Fixed by dropping `--auto-save` from the built command; regression test `test_generated_command_is_accepted_by_run_prediction_argparse` in `tests/test_schedule_daily.py` parses the real generated command against `run_prediction.build_parser()` so this class of drift can't recur unnoticed. Task was re-registered with the corrected command. Lesson: `schedule_daily`'s tests all mocked `subprocess.run`, so nothing ever executed the generated command against the real target script — a scheduler test suite needs at least one test like this that exercises the actual argument contract, not just the schtasks call shape.
- **Log backups** (`scripts/backup_logs.py`): timestamped `shutil.copy2` backups of `data/prediction_audit_log.csv`/`data/value_bets_log.csv` to `data/logs/backup_{timestamp}_{filename}.csv` (gitignored, rotated to the last 7 per file), called by `daily_scanner.py` right before it writes any value bets. Second line of defense alongside both log files being git-tracked (`!`-negated in `.gitignore`) — added after a 2026-09-19 incident where an ad-hoc dedup script deleted 26 rows from the then-untracked audit log (no unique data was actually lost — see `docs/metrics/2026-09-19-audit-log-protection.md` — but the file had no recovery path at the time)
- **Data Loading** (`src/data/loader.py`): ATP (Tennismylife API) and WTA (tennis-data.co.uk, scraped live for current URLs) loaders; `load_davis_cup_matches` filters the ATP feed by `tourney_level == "D"` — tennis-data.co.uk has no separate Davis Cup source
- **Config** (`src/config.py`): Environment variables, MIN_MATCHES_THRESHOLD=25, MAX_SUSPICIOUS_EDGE=0.10

## Key Design Patterns
- Mirror rows: every match has a mirrored counterpart (loser's perspective, outcome=0)
- No-lookahead: all features computed sequentially before match update
- Shrinkage calibration: predictions outside [0.10, 0.90] compressed 60% toward boundary (verified near-optimal empirically — see calibration investigation doc; only affects the tails, never the [0.20, 0.80] core range)
- Cache: model persisted in `data/model_cache/{tour}.pkl` with 7-day expiry
- Davis Cup shares ATP's raw directory (`_RAW_TOUR_DIRS`) and staleness mechanism — no separate data source exists

## FEATURE_COLS (15, `src/features/__init__.py`)
`elo_diff, elo_prob, rank_diff, form_diff, surface_form_diff, h2h_rate, rest_diff, rolling_elo_diff, age_multiplier_diff, rust_factor_diff, fatigue_multiplier_diff, surface_transition_multiplier_diff, adjusted_elo_diff, h2h_trend, surface_win_rate_trend_diff`

`momentum_3_diff`/`momentum_5_diff` are computed by `FeatureBuilder` but deliberately **not** in `FEATURE_COLS` — A/B backtest showed no clear benefit and a small net negative when bundled with the other two additions.

## Test Coverage
- 694 tests, all passing
- Key test areas: backtest (incl. calibration-prediction instrumentation), bankroll simulation, calibration audit, calibration metrics, dashboard, features, Elo, pipeline, value analysis (incl. log dedup: `check_existing_log_entry`/`update_log_entry`), daily scanner, daily workflow, surface resolver, player matcher, schedule_daily (incl. a regression test that the generated schtasks command is accepted by `run_prediction.py`'s real argparse), run_prediction, backup_logs, gitignore protection (real `git ls-files` checks), value_bets_log duplicate-identity invariant against the real production CSV

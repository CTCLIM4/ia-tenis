# ia-tenis Project Context

## Overview
Tennis match prediction model (ATP/WTA/Davis Cup) with Elo + features + logistic calibration, an interactive value-bet analyzer, an automated daily scanner, and a visualization dashboard.

## Stack
- Python 3.14.5, pandas 3.0.3, numpy 2.5.0, sklearn 1.9.0, scipy 1.18.0, matplotlib 3.10
- pytest 9.1.1, 663 tests all passing
- Key modules: `src/models/elo.py`, `src/features/engineering.py`, `src/features/decay.py`, `src/backtest/walkforward.py`, `src/data/loader.py`, `src/value_analysis.py`, `src/calibration_audit.py`, `src/calibration_metrics.py`, `src/dashboard.py`, `src/daily_scanner.py`, `src/surface_resolver.py`, `src/player_matcher.py`

## Architecture
- **EloSystem** (`src/models/elo.py`): General + surface Elo ratings with yearly decay, K-factor based on match count
- **FeatureBuilder** (`src/features/engineering.py`): Recent win rates, momentum (3/5-match), H2H rate + trend, surface win-rate trend, rest days
- **EloHistoryTracker + decay features** (`src/features/decay.py`): Rolling Elo, age/rust/fatigue/surface-transition multipliers
- **Pipeline** (`src/pipeline.py`): Load → build features → walk-forward backtest, tour-agnostic (atp/wta/davis)
- **Value Analysis** (`src/value_analysis.py`): LR model + shrinkage calibration + odds API + interactive CLI (`--wta`/`--davis`)
- **Backtest** (`src/backtest/walkforward.py`): Walk-forward LR evaluation with mirror rows; `return_predictions=True` exposes raw (prob, outcome) pairs (including a mirror-balanced `probs_calibration`/`y_true_calibration` pair) for calibration analysis
- **Calibration metrics** (`src/calibration_metrics.py`): Reliability/calibration-curve analysis against the large backtest sample (never the small real-bet log — see `docs/metrics/2026-09-18-calibration-investigation.md`)
- **Dashboard** (`src/dashboard.py`): Backtest accuracy by year, calibration curves, LR feature importance, value-bet history — PNGs to `data/dashboard/` (gitignored, regenerate with `python -m src.dashboard`)
- **Daily scanner** (`src/daily_scanner.py`): Discovers live matches via The Odds API, no staleness gate (prints warnings only), `--auto-save`, `--in-play` (opt-in, loud warning — model has no live-match-state signal), rate-limit retry with backoff
- **Daily workflow** (`scripts/daily_workflow.py`): Stricter automation wrapper — DOES hard-exclude a tour on any staleness (unless `--live-tournament` opts out of that), auto-logs + emails + git-pushes
- **Runner** (`scripts/run_prediction.py`): download → pipeline → retrain (non-interactive `load_model(tour, retrain=True)`, never the interactive CLI) → scanner --auto-save, `--log` writes `data/logs/daily_{date}.log`
- **Scheduler** (`scripts/schedule_daily.py`): Builds/removes a Windows Scheduled Task for `run_prediction.py` — not registered by default, run manually to activate
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
- 663 tests, all passing
- Key test areas: backtest (incl. calibration-prediction instrumentation), bankroll simulation, calibration audit, calibration metrics, dashboard, features, Elo, pipeline, value analysis, daily scanner, daily workflow, surface resolver, player matcher, schedule_daily, run_prediction

"""Shared feature-column schema — single source of truth for the exact list
and order of columns fed into the LogisticRegression, used identically at
training time (src/backtest/walkforward.py) and prediction time
(src/value_analysis.py). Order matters: it's positional in the model's
input vector (np.array([[feats[c] for c in FEATURE_COLS]])).
"""
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff", "surface_transition_multiplier_diff",
    "adjusted_elo_diff",
    # Fase 5 (2026-09-18): h2h_trend and surface_win_rate_trend_diff each
    # showed a small walk-forward accuracy improvement (ATP + WTA,
    # 2020-2026) with no calibration/log-loss cost, kept here.
    # momentum_3_diff/momentum_5_diff (also computed by FeatureBuilder, see
    # src/features/engineering.py) showed no clear benefit and made things
    # slightly worse bundled with the other two -- deliberately NOT added.
    "h2h_trend", "surface_win_rate_trend_diff",
]

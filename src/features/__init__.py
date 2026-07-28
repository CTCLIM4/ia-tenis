"""Shared feature-column schema — single source of truth for the exact list
and order of columns fed into the LogisticRegression, used identically at
training time (src/backtest/walkforward.py) and prediction time
(src/value_analysis.py). Order matters: it's positional in the model's
input vector (np.array([[feats[c] for c in FEATURE_COLS]])).
"""
FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff",
]

"""Tests for src/features/__init__.py's FEATURE_COLS -- the single source of
truth for which columns actually reach the LR model.

Fase 5 (2026-09-18) added 4 candidate features (momentum_3_diff,
momentum_5_diff, h2h_trend, surface_win_rate_trend_diff) to the feature
pipeline, but only added h2h_trend and surface_win_rate_trend_diff to
FEATURE_COLS -- an A/B walk-forward backtest (both ATP and WTA, 2020-2026)
showed momentum_3_diff/momentum_5_diff added no clear benefit and made
things slightly worse when bundled with the other two, while h2h_trend and
surface_win_rate_trend_diff each showed a small accuracy improvement with
no measurable calibration/log-loss cost. See
docs/metrics/2026-09-18-feature-engineering-fase5.md for the numbers.
"""
from __future__ import annotations

from src.features import FEATURE_COLS


def test_includes_the_two_features_that_earned_their_place():
    assert "h2h_trend" in FEATURE_COLS
    assert "surface_win_rate_trend_diff" in FEATURE_COLS


def test_excludes_momentum_features_that_did_not_help():
    assert "momentum_3_diff" not in FEATURE_COLS
    assert "momentum_5_diff" not in FEATURE_COLS


def test_no_duplicate_columns():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

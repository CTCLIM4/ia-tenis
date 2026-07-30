"""Tests for src/bankroll_simulation.py — Módulo 4 (bankroll Monte Carlo simulation)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.bankroll_simulation import estimate_bet_profile, sample_bet


def _resolved_df_for_profile():
    """3 resolved bets shaped like backtest_analytics.load_resolved_bets output
    (has bet_side/odds_taken derived, plus the raw edge_a/edge_b columns)."""
    return pd.DataFrame({
        "bet_side":  ["a", "b", "a"],
        "edge_a":    [0.05, -0.05, 0.03],
        "edge_b":    [-0.1, 0.08, -0.06],
        "odds_taken": [2.0, 1.8, 2.5],
    })


class TestEstimateBetProfile:
    def test_computes_mean_and_std_from_bet_side_edge(self):
        profile = estimate_bet_profile(_resolved_df_for_profile())
        # edge_taken = [0.05, 0.08, 0.03] (side actually bet on each row)
        assert profile["mean_edge"] == pytest.approx(0.0533333, rel=1e-4)
        assert profile["std_edge"] == pytest.approx(0.0251661, rel=1e-4)
        assert profile["mean_odds"] == pytest.approx(2.1)
        assert profile["std_odds"] == pytest.approx(0.3605551, rel=1e-4)
        assert profile["n"] == 3

    def test_single_row_has_zero_std(self):
        df = pd.DataFrame({
            "bet_side": ["a"], "edge_a": [0.05], "edge_b": [-0.1],
            "odds_taken": [2.0],
        })
        profile = estimate_bet_profile(df)
        assert profile["mean_edge"] == pytest.approx(0.05)
        assert profile["std_edge"] == 0.0
        assert profile["mean_odds"] == pytest.approx(2.0)
        assert profile["std_odds"] == 0.0
        assert profile["n"] == 1

    def test_empty_df_raises(self):
        df = pd.DataFrame({"bet_side": [], "edge_a": [], "edge_b": [], "odds_taken": []})
        with pytest.raises(ValueError):
            estimate_bet_profile(df)


class TestSampleBet:
    def test_deterministic_when_std_is_zero(self):
        profile = {"mean_edge": 0.05, "std_edge": 0.0, "mean_odds": 2.0, "std_odds": 0.0, "n": 5}
        rng = np.random.default_rng(0)
        p_win, odds = sample_bet(rng, profile)
        # odds collapses to mean_odds exactly (sigma=0 lognormal)
        assert odds == pytest.approx(2.0)
        # edge collapses to mean_edge exactly (sigma=0 normal); implied=1/2.0=0.5
        assert p_win == pytest.approx(0.55)

    def test_edge_floor_prevents_nonpositive_edge(self):
        profile = {"mean_edge": -0.02, "std_edge": 0.0, "mean_odds": 2.0, "std_odds": 0.0, "n": 5}
        rng = np.random.default_rng(0)
        p_win, odds = sample_bet(rng, profile)
        # edge floored to 0.001; implied=0.5 -> p_win=0.501
        assert p_win == pytest.approx(0.501)

    def test_seed_reproducibility_with_dispersion(self):
        profile = {"mean_edge": 0.05, "std_edge": 0.02, "mean_odds": 2.0, "std_odds": 0.3, "n": 10}
        r1 = sample_bet(np.random.default_rng(42), profile)
        r2 = sample_bet(np.random.default_rng(42), profile)
        assert r1 == r2

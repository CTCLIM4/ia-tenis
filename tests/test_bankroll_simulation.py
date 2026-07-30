"""Tests for src/bankroll_simulation.py — Módulo 4 (bankroll Monte Carlo simulation)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.bankroll_simulation import estimate_bet_profile, kelly_stake, sample_bet
from src.bankroll_simulation import simulate_path  # add to existing import block
from src.bankroll_simulation import run_monte_carlo  # add to existing import block
from src.bankroll_simulation import print_report  # add to existing import block


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


class TestKellyStake:
    def test_positive_edge_scales_with_multiplier(self):
        # p_win=0.55, odds=2.0 -> implied=0.5, edge=0.05, raw=0.05/(1.0)=0.05
        assert kelly_stake(0.55, 2.0, kelly_multiplier=1.0) == pytest.approx(0.05)
        assert kelly_stake(0.55, 2.0, kelly_multiplier=0.5) == pytest.approx(0.025)

    def test_negative_edge_returns_zero(self):
        # p_win=0.3, odds=2.0 -> implied=0.5, edge=-0.2 -> no bet
        assert kelly_stake(0.3, 2.0, kelly_multiplier=1.0) == 0.0

    def test_cap_dominates_at_high_edge(self):
        # p_win=0.9, odds=1.5 -> implied=0.6667, edge=0.2333, raw=0.4667
        # full Kelly and 1/4 Kelly both still hit the 5% cap
        assert kelly_stake(0.9, 1.5, kelly_multiplier=1.0) == pytest.approx(0.05)
        assert kelly_stake(0.9, 1.5, kelly_multiplier=0.25) == pytest.approx(0.05)
        # 1/10 Kelly finally drops below the cap: 0.1 * 0.4667 = 0.04667
        assert kelly_stake(0.9, 1.5, kelly_multiplier=0.1) == pytest.approx(0.046667, rel=1e-3)


class _FixedRNG:
    """Deterministic stand-in for np.random.Generator. normal()/lognormal()
    collapse to their location parameter (only used with std=0 profiles in
    these tests), and random() replays a fixed win/loss sequence — lets us
    hand-verify the bankroll arithmetic exactly instead of trusting opaque
    Generator internals."""

    def __init__(self, wins):
        self._wins = list(wins)
        self._i = 0

    def normal(self, loc, scale):
        return loc

    def lognormal(self, mean, sigma):
        return math.exp(mean)

    def random(self):
        win = self._wins[self._i]
        self._i += 1
        return 0.0 if win else 0.999


class TestSimulatePath:
    def test_hand_verified_bankroll_sequence(self):
        # profile is deterministic (std=0 both): every sampled bet is
        # p_win=0.55, odds=2.0 -> kelly_stake(1.0) = 0.05 (at the cap)
        profile = {"mean_edge": 0.05, "std_edge": 0.0, "mean_odds": 2.0, "std_odds": 0.0, "n": 5}
        rng = _FixedRNG(wins=[True, False, True])
        path = simulate_path(rng, profile, kelly_multiplier=1.0, n_bets=3, initial_bankroll=1000.0)

        # bet1: stake=50,  win  -> 1000 + 50*(2.0-1)   = 1050.0
        # bet2: stake=52.5,lose -> 1050 - 52.5         = 997.5
        # bet3: stake=49.875,win-> 997.5 + 49.875*1.0  = 1047.375
        expected = np.array([1000.0, 1050.0, 997.5, 1047.375])
        np.testing.assert_allclose(path, expected)

    def test_bankroll_never_negative(self):
        # extreme profile: huge edge relative to odds still caps at 5%/bet,
        # so bankroll shrinks but must never cross zero even on an
        # all-losses run.
        profile = {"mean_edge": 0.05, "std_edge": 0.0, "mean_odds": 2.0, "std_odds": 0.0, "n": 5}
        rng = _FixedRNG(wins=[False] * 50)
        path = simulate_path(rng, profile, kelly_multiplier=1.0, n_bets=50, initial_bankroll=1000.0)
        assert (path >= 0.0).all()

    def test_same_seed_reproducible(self):
        profile = {"mean_edge": 0.05, "std_edge": 0.02, "mean_odds": 2.0, "std_odds": 0.3, "n": 10}
        p1 = simulate_path(np.random.default_rng(7), profile, 1.0, n_bets=20, initial_bankroll=1000.0)
        p2 = simulate_path(np.random.default_rng(7), profile, 1.0, n_bets=20, initial_bankroll=1000.0)
        np.testing.assert_array_equal(p1, p2)


_MC_PROFILE = {"mean_edge": 0.05, "std_edge": 0.02, "mean_odds": 2.0, "std_odds": 0.3, "n": 10}


class TestRunMonteCarlo:
    def test_output_structure_and_sane_ranges(self):
        result = run_monte_carlo(
            _MC_PROFILE, kelly_multipliers=[1.0, 0.5, 0.25],
            n_simulations=200, n_bets=15, initial_bankroll=1000.0,
            ruin_threshold=0.5, seed=42,
        )
        assert set(result["results"].keys()) == {1.0, 0.5, 0.25}
        assert result["n_simulations"] == 200
        assert result["n_bets"] == 15
        assert result["initial_bankroll"] == 1000.0
        assert result["ruin_threshold"] == 0.5
        assert result["profile"] == _MC_PROFILE

        for stats in result["results"].values():
            assert stats["p10"] <= stats["p50"] <= stats["p90"]
            assert stats["p10"] >= 0.0
            assert 0.0 <= stats["ruin_probability"] <= 1.0
            assert stats["median_max_drawdown_pct"] <= 0.0

    def test_same_seed_is_reproducible(self):
        kwargs = dict(
            kelly_multipliers=[1.0], n_simulations=50, n_bets=10,
            initial_bankroll=1000.0, ruin_threshold=0.5, seed=7,
        )
        r1 = run_monte_carlo(_MC_PROFILE, **kwargs)
        r2 = run_monte_carlo(_MC_PROFILE, **kwargs)
        assert r1 == r2


def _mc_result(n=7):
    return {
        "profile": {"mean_edge": 0.058, "std_edge": 0.041, "mean_odds": 2.03, "std_odds": 0.60, "n": n},
        "results": {
            1.0:  {"p10": 612.34, "p50": 1340.55, "p90": 2850.90, "ruin_probability": 0.184, "median_max_drawdown_pct": -34.2},
            0.5:  {"p10": 780.11, "p50": 1190.20, "p90": 1950.44, "ruin_probability": 0.062, "median_max_drawdown_pct": -19.8},
            0.25: {"p10": 890.02, "p50": 1080.15, "p90": 1420.33, "ruin_probability": 0.011, "median_max_drawdown_pct": -10.5},
        },
        "n_simulations": 10000, "n_bets": 50,
        "initial_bankroll": 1000.0, "ruin_threshold": 0.5,
    }


class TestPrintReport:
    def test_report_contains_expected_sections(self, capsys):
        print_report(_mc_result())
        out = capsys.readouterr().out
        assert "SIMULACION MONTE CARLO DE BANCA" in out
        assert "$1,000.00" in out
        assert "10000 simulaciones x 50 apuestas" in out
        assert "N=7" in out
        assert "Perfil estimado de muestra chica" in out
        assert "Kelly completo" in out
        assert "1/2 Kelly" in out
        assert "1/4 Kelly" in out
        assert "Prob. de ruina" in out

    def test_manual_profile_omits_n_and_warning(self, capsys):
        mc_result = _mc_result(n="manual")
        print_report(mc_result)
        out = capsys.readouterr().out
        assert "N=" not in out
        assert "muestra chica" not in out

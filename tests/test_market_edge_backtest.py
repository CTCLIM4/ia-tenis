"""Synthetic tests for scripts/market_edge_backtest.py's selection logic --
it must replay the production rule exactly, or its ROI says nothing about
production."""
import pandas as pd
import pytest

from scripts import market_edge_backtest as meb


def _sides(**cols):
    base = {"year": 2024, "y": 1, "q": 0.5, "q_ps": 0.5, "b365": 2.0, "max": 2.0}
    n = len(cols["p"])
    return pd.DataFrame({**{k: [v] * n for k, v in base.items()}, **cols})


def test_select_bets_applies_min_and_suspicious_edge_bounds():
    # odds 2.0 -> implied 0.5; edges 0.02 (below), 0.03, 0.10, 0.11 (suspicious)
    sides = _sides(p=[0.52, 0.53, 0.60, 0.61], avg=[2.0] * 4)
    bets = meb.select_bets(sides, "avg")
    assert bets["p"].tolist() == [0.53, 0.60]


def test_select_bets_stake_is_capped_quarter_kelly_and_return_is_per_unit():
    sides = _sides(p=[0.60], avg=[2.0], y=[0])
    bet = meb.select_bets(sides, "avg").iloc[0]
    assert bet["stake"] == pytest.approx(min(0.10 / 1.0, meb.KELLY_CAP) / meb.KELLY_DIVISOR)
    assert bet["ev"] == pytest.approx(0.60 * 1.0 - 0.40)
    assert bet["ret"] == -1.0


def test_bet_sides_mirror_probabilities_and_devig():
    matches = pd.DataFrame({"year": [2024], "p": [0.7], "AvgW": [1.5], "AvgL": [3.0],
                            "PSW": [1.5], "PSL": [3.0], "B365W": [1.5], "B365L": [3.0],
                            "MaxW": [1.6], "MaxL": [3.2]})
    sides = meb.bet_sides(matches)
    assert sides["p"].tolist() == pytest.approx([0.7, 0.3])
    assert sides["y"].tolist() == [1, 0]
    assert sides["q"].sum() == pytest.approx(1.0)
    assert sides["q"].iloc[0] == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 3.0))

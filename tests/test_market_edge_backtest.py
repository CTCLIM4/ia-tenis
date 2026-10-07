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


@pytest.mark.parametrize("tennis_data, tennismylife", [
    ("Alcaraz C.", "Carlos Alcaraz"),
    ("Bautista Agut R.", "Roberto Bautista Agut"),
    ("Del Potro J.M.", "Juan Martin del Potro"),
    ("de Minaur A.", "Alex de Minaur"),
    ("Auger-Aliassime F.", "Felix Auger Aliassime"),
    ("Van De Zandschulp B.", "Botic van de Zandschulp"),
])
def test_player_key_matches_both_name_formats(tennis_data, tennismylife):
    assert meb.player_key(tennis_data) == meb.player_key(tennismylife)


def test_player_key_separates_different_initials():
    assert meb.player_key("Zverev A.") != meb.player_key("Mischa Zverev")


def _odds_row(date, winner, loser, avg_w=1.5):
    row = {c: float("nan") for c in meb.ODDS_COLS}
    row.update(match_date=pd.Timestamp(date), winner=winner, loser=loser, AvgW=avg_w, AvgL=2.5)
    return row


def test_atp_join_uses_tournament_window_and_drops_ambiguous_pairs():
    # Tennismylife dates every match with the tournament start date.
    pred = pd.DataFrame({
        "match_date": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-03-04"]),
        "year": 2024, "p": 0.6,
        "winner": ["Carlos Alcaraz", "Alexander Zverev", "Carlos Alcaraz"],
        "loser": ["Richard Gasquet", "Carlos Alcaraz", "Richard Gasquet"],
    })
    odds = pd.DataFrame([
        _odds_row("2024-01-16", "Alcaraz C.", "Gasquet R.", avg_w=1.02),   # 1 day later: joins
        _odds_row("2024-02-20", "Zverev A.", "Alcaraz C."),                # 36 days later: outside window
        _odds_row("2024-03-06", "Alcaraz C.", "Gasquet R.", avg_w=1.10),   # second Alcaraz-Gasquet: joins row 3 only
    ])
    joined = meb.join_odds(pred, odds, "atp")
    assert joined["AvgW"].tolist()[0] == 1.02
    assert pd.isna(joined["AvgW"].tolist()[1])
    assert joined["AvgW"].tolist()[2] == 1.10


def test_anchored_predictions_are_symmetric_and_walk_forward():
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2010, 2015):
        for _ in range(60):
            w, l = rng.uniform(1.2, 3.5, 2)
            row = {c: rng.normal() for c in meb._FEATURE_COLS}
            row.update(elo_prob=rng.uniform(0.2, 0.8), h2h_rate=rng.uniform(0, 1))
            row.update(year=year, p=rng.uniform(0.2, 0.8), AvgW=w, AvgL=l, PSW=w, PSL=l,
                       B365W=w, B365L=l, MaxW=w, MaxL=l)
            rows.append(row)
    out = meb.anchored_predictions(pd.DataFrame(rows))
    assert sorted(out["year"].unique()) == [2013, 2014]  # first 3 odds years only train
    n = len(out) // 2
    for col in ("M1", "M2", "M3"):
        winner_side = out[out["y"] == 1][col].to_numpy()
        loser_side = out[out["y"] == 0][col].to_numpy()
        assert winner_side + loser_side == pytest.approx(np.ones(n))

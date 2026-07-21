from datetime import date, timedelta

import pandas as pd

from src.backtest.walkforward import build_match_features
from src.features.decay import EloHistoryTracker
from src.features.engineering import FeatureBuilder
from src.models.elo import EloSystem


def _row(winner, loser, surface, d, winner_age=None, loser_age=None):
    return {
        "winner_name": winner,
        "loser_name": loser,
        "surface": surface,
        "match_date": pd.Timestamp(d),
        "winner_rank": 10.0,
        "loser_rank": 20.0,
        "winner_age": winner_age,
        "loser_age": loser_age,
    }


def test_output_includes_decay_columns_neutral_for_brand_new_players():
    df = pd.DataFrame([_row("A", "B", "clay", date(2023, 1, 1))])
    match_df = build_match_features(df, EloSystem(), FeatureBuilder())

    for col in ("rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff"):
        assert col in match_df.columns

    original = match_df[~match_df["is_mirror"]].iloc[0]
    assert original["rolling_elo_diff"] == 0.0
    assert original["age_multiplier_diff"] == 0.0
    assert original["rust_factor_diff"] == 0.0
    # both brand-new -> both adjusted_elo_surface == 1500 -> diff 0
    assert original["adjusted_elo_diff"] == 0.0


def test_mirror_rows_negate_decay_columns():
    df = pd.DataFrame(
        [
            _row("A", "B", "clay", date(2023, 1, 1), winner_age=35, loser_age=22),
            _row("A", "B", "clay", date(2023, 3, 1), winner_age=35, loser_age=22),
        ]
    )
    match_df = build_match_features(df, EloSystem(), FeatureBuilder())
    original = match_df[~match_df["is_mirror"]]
    mirror = match_df[match_df["is_mirror"]]

    for col in ("rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff"):
        assert list(mirror[col]) == [-v for v in original[col]]


def test_adjusted_elo_diff_penalizes_inactive_aging_veteran():
    """A veteran with a strong clay Elo but no recent matches and an
    advanced age should show a much smaller (or negative) adjusted_elo_diff
    than the raw elo_diff would suggest, against an active young opponent."""
    rows = []
    d = date(2020, 1, 1)
    # Veteran builds a strong clay record over several years (all wins vs filler opponents)
    for i in range(20):
        rows.append(_row("Veteran", f"Filler{i}", "clay", d, winner_age=35, loser_age=25))
        d += timedelta(days=60)

    # Long injury/inactivity gap, then faces a young active opponent
    final_date = d + timedelta(days=400)
    rows.append(_row("Veteran", "YoungGun", "clay", final_date, winner_age=41, loser_age=24))

    df = pd.DataFrame(rows)
    match_df = build_match_features(df, EloSystem(), FeatureBuilder())
    original = match_df[~match_df["is_mirror"]]
    final_row = original.iloc[-1]

    assert final_row["elo_diff"] > 100  # Veteran's raw Elo is far ahead
    assert final_row["adjusted_elo_diff"] < final_row["elo_diff"] - 100


def test_elo_tracker_argument_gets_populated():
    tracker = EloHistoryTracker()
    d = date(2023, 1, 1)
    rows = [_row("A", "B", "clay", d + timedelta(days=10 * i)) for i in range(6)]
    df = pd.DataFrame(rows)
    build_match_features(df, EloSystem(), FeatureBuilder(), elo_tracker=tracker)

    assert tracker.rolling_elo("A", "clay") is not None

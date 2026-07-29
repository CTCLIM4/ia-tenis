from datetime import date, timedelta

import pandas as pd
import pytest

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


def test_mirror_row_negates_h2h_rate_as_one_minus_rate():
    """h2h_rate isn't in _MIRROR_FLIP_COLS (it's a bounded [0,1] rate, not a
    signed diff) — the mirror row must instead show 1 - original, and this
    must still hold exactly under the new weighted+shrunk formula."""
    df = pd.DataFrame(
        [
            _row("A", "B", "clay", date(2023, 1, 1)),
            _row("B", "A", "clay", date(2023, 2, 1)),
            _row("A", "B", "clay", date(2023, 3, 1)),
        ]
    )
    match_df = build_match_features(df, EloSystem(), FeatureBuilder())
    original = match_df[~match_df["is_mirror"]]
    mirror = match_df[match_df["is_mirror"]]

    for orig_rate, mirror_rate in zip(original["h2h_rate"], mirror["h2h_rate"]):
        assert mirror_rate == pytest.approx(1 - orig_rate)


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


class TestDefensiveSort:
    """build_match_features processes rows sequentially assuming chronological
    order (no-lookahead requirement) — it must not trust the caller to have
    sorted the input, or an out-of-order df silently corrupts every Elo/form
    feature computed from it."""

    @staticmethod
    def _chronological_rows():
        d0 = date(2023, 1, 1)
        return [
            _row("A", "B", "clay", d0),
            _row("A", "C", "clay", d0 + timedelta(days=30)),
            _row("A", "D", "clay", d0 + timedelta(days=60)),
        ]

    def test_output_is_ordered_by_match_date_even_if_input_is_not(self):
        rows = self._chronological_rows()
        shuffled = pd.DataFrame([rows[2], rows[0], rows[1]]).reset_index(drop=True)

        result = build_match_features(shuffled, EloSystem(), FeatureBuilder())
        original = result[~result["is_mirror"]]

        assert list(original["match_date"]) == sorted(original["match_date"])

    def test_shuffled_input_produces_same_result_as_presorted_input(self):
        rows = self._chronological_rows()
        df_sorted = pd.DataFrame(rows)
        df_shuffled = pd.DataFrame([rows[2], rows[0], rows[1]]).reset_index(drop=True)

        result_sorted = build_match_features(df_sorted, EloSystem(), FeatureBuilder())
        result_shuffled = build_match_features(df_shuffled, EloSystem(), FeatureBuilder())

        cols = ["winner", "loser", "elo_diff", "elo_prob"]
        original_sorted = result_sorted[~result_sorted["is_mirror"]][cols].reset_index(drop=True)
        original_shuffled = result_shuffled[~result_shuffled["is_mirror"]][cols].reset_index(drop=True)

        pd.testing.assert_frame_equal(original_sorted, original_shuffled)

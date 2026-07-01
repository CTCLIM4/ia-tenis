import pytest
from datetime import date
from src.features.engineering import FeatureBuilder


def test_new_player_returns_defaults():
    fb = FeatureBuilder()
    f = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    assert f["recent_win_rate"] == 0.5
    assert f["h2h_win_rate"] == 0.5
    assert f["h2h_matches"] == 0
    assert f["rest_days"] == 14  # default when no history


def test_recent_win_rate_after_all_wins():
    fb = FeatureBuilder()
    for _ in range(10):
        fb.update("A", "B", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 1.0


def test_recent_win_rate_after_all_losses():
    fb = FeatureBuilder()
    for _ in range(10):
        fb.update("B", "A", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 0.0


def test_recent_win_rate_uses_only_last_n():
    fb = FeatureBuilder(recent_n=5)
    # 10 losses then 5 wins — should see only the 5 wins
    for _ in range(10):
        fb.update("B", "A", "hard", date(2023, 1, 1))
    for _ in range(5):
        fb.update("A", "B", "hard", date(2023, 2, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 1.0


def test_h2h_tracks_wins_per_direction():
    fb = FeatureBuilder()
    fb.update("A", "B", "hard", date(2023, 1, 1))
    fb.update("A", "B", "hard", date(2023, 2, 1))
    fb.update("B", "A", "hard", date(2023, 3, 1))

    fa = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    fb_ = fb.get_features("B", "A", "hard", date(2023, 6, 1))

    assert fa["h2h_matches"] == 3
    assert abs(fa["h2h_win_rate"] - 2 / 3) < 1e-10
    assert abs(fb_["h2h_win_rate"] - 1 / 3) < 1e-10


def test_rest_days_calculated_correctly():
    fb = FeatureBuilder()
    fb.update("A", "B", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 1, 8))
    assert f["rest_days"] == 7


def test_surface_form_is_surface_specific():
    fb = FeatureBuilder(surface_n=5)
    fb.update("A", "B", "clay", date(2023, 1, 1))
    fb.update("A", "B", "clay", date(2023, 1, 2))
    fb.update("B", "A", "hard", date(2023, 1, 3))
    fb.update("B", "A", "hard", date(2023, 1, 4))

    f_clay = fb.get_features("A", "B", "clay", date(2023, 6, 1))
    f_hard = fb.get_features("A", "B", "hard", date(2023, 6, 1))

    assert f_clay["recent_win_rate_surface"] == 1.0
    assert f_hard["recent_win_rate_surface"] == 0.0


def test_update_does_not_affect_features_for_current_match():
    """Features must use only pre-match information."""
    fb = FeatureBuilder()
    f_before = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    fb.update("A", "B", "hard", date(2023, 1, 1))
    f_after = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    # h2h before update should be 0, after should be 1
    assert f_before["h2h_matches"] == 0
    assert f_after["h2h_matches"] == 1

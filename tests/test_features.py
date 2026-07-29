import pytest
from datetime import date
from src.features.engineering import FeatureBuilder, _weighted_h2h_rate


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


def test_h2h_matches_counts_both_directions():
    """h2h_matches (total count) is unaffected by the weighted/shrunk rate
    formula change — still a plain count of meetings either direction."""
    fb = FeatureBuilder()
    fb.update("A", "B", "hard", date(2023, 1, 1))
    fb.update("A", "B", "hard", date(2023, 2, 1))
    fb.update("B", "A", "hard", date(2023, 3, 1))

    fa = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    fb_ = fb.get_features("B", "A", "hard", date(2023, 6, 1))

    assert fa["h2h_matches"] == 3
    assert fb_["h2h_matches"] == 3


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


# ── _weighted_h2h_rate ────────────────────────────────────────────────────────
# Hand-computed expected values — see docs/superpowers/specs/2026-07-28-h2h-smoothing-design.md
# for the formula: shrink = n/(n+4), rate = shrink*weighted_rate + (1-shrink)*0.5,
# weighted_rate = weighted_average(wins, weights=linspace(0.5, 1.0, n)).

class TestWeightedH2hRate:
    def test_no_matches_returns_neutral(self):
        assert _weighted_h2h_rate([]) == 0.5

    def test_single_win_shrinks_toward_neutral(self):
        # n=1: weighted_rate=1.0 (single point, weight cancels), shrink=1/5=0.2
        # rate = 0.2*1.0 + 0.8*0.5 = 0.6 (not 1.0 — one win isn't strong evidence)
        matches = [(date(2023, 1, 1), True)]
        assert _weighted_h2h_rate(matches) == pytest.approx(0.6)

    def test_single_loss_shrinks_toward_neutral(self):
        matches = [(date(2023, 1, 1), False)]
        assert _weighted_h2h_rate(matches) == pytest.approx(0.4)

    def test_recency_weighting_favors_more_recent_result(self):
        # Same 1-1 record, opposite chronological order -> different rate.
        # n=2, weights=[0.5,1.0]: recent win -> weighted_rate=1.0/1.5=2/3,
        # shrink=2/6=1/3, rate=1/3*2/3 + 2/3*0.5 = 2/9+1/3 = 5/9.
        lost_old_won_recent = [(date(2023, 1, 1), False), (date(2023, 2, 1), True)]
        won_old_lost_recent = [(date(2023, 1, 1), True), (date(2023, 2, 1), False)]

        recent_win_rate = _weighted_h2h_rate(lost_old_won_recent)
        recent_loss_rate = _weighted_h2h_rate(won_old_lost_recent)

        assert recent_win_rate == pytest.approx(5 / 9)
        assert recent_loss_rate == pytest.approx(4 / 9)
        assert recent_win_rate > recent_loss_rate

    def test_opponent_perspective_is_exact_complement(self):
        # h2h_rate(B,A) must equal 1 - h2h_rate(A,B) for any history — the
        # mirror-row rule (h2h_rate -> 1-h2h_rate) depends on this holding
        # exactly, not approximately.
        player_view = [(date(2023, 1, 1), True), (date(2023, 2, 1), False), (date(2023, 3, 1), True)]
        opponent_view = [(date(2023, 1, 1), False), (date(2023, 2, 1), True), (date(2023, 3, 1), False)]

        assert _weighted_h2h_rate(opponent_view) == pytest.approx(
            1 - _weighted_h2h_rate(player_view)
        )

    def test_larger_sample_still_shrinks_but_less(self):
        # n=10, all wins: weighted_rate=1.0 (uniform result regardless of
        # weights), shrink=10/14=5/7, rate = 5/7*1.0 + 2/7*0.5 = 6/7.
        # Still not 1.0 even with 10 straight wins, but much closer than n=1's 0.6.
        matches = [(date(2023, 1, i + 1), True) for i in range(10)]
        assert _weighted_h2h_rate(matches) == pytest.approx(6 / 7)


class TestWorkloadHistory:
    def test_empty_for_new_player(self):
        fb = FeatureBuilder()
        assert fb.workload_history("A") == []

    def test_records_sets_played_on_update(self):
        fb = FeatureBuilder()
        fb.update("A", "B", "hard", date(2023, 1, 1), sets_played=3)
        assert fb.workload_history("A") == [(date(2023, 1, 1), 3)]
        assert fb.workload_history("B") == [(date(2023, 1, 1), 3)]

    def test_sets_played_defaults_to_zero(self):
        """Existing update() call sites (H2H/form/rest tests) don't pass
        sets_played -- must not break them."""
        fb = FeatureBuilder()
        fb.update("A", "B", "hard", date(2023, 1, 1))
        assert fb.workload_history("A") == [(date(2023, 1, 1), 0)]

    def test_accumulates_across_matches_in_order(self):
        fb = FeatureBuilder()
        fb.update("A", "B", "hard", date(2023, 1, 1), sets_played=2)
        fb.update("A", "C", "hard", date(2023, 1, 10), sets_played=3)
        assert fb.workload_history("A") == [
            (date(2023, 1, 1), 2), (date(2023, 1, 10), 3),
        ]

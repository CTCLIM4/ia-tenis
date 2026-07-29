from datetime import date

import pytest

from src.features.decay import (
    EloHistoryTracker,
    age_multiplier,
    calculate_decay_features,
    fatigue_multiplier,
    rust_factor,
)


# ── age_multiplier ────────────────────────────────────────────────────────────

def test_age_multiplier_is_neutral_at_or_below_30():
    assert age_multiplier(30) == 1.0
    assert age_multiplier(22) == 1.0


def test_age_multiplier_penalizes_veterans():
    # 41yo: 1.0 - (41-30)*0.025 = 0.725
    assert abs(age_multiplier(41) - 0.725) < 1e-9


def test_age_multiplier_floors_at_point_six():
    # 70yo would compute to 1.0 - 40*0.025 = 0.0, must floor at 0.60
    assert age_multiplier(70) == 0.60


def test_age_multiplier_neutral_when_age_unknown():
    assert age_multiplier(None) == 1.0


# ── rust_factor ───────────────────────────────────────────────────────────────

def test_rust_factor_full_when_four_or_more_recent_matches():
    today = date(2026, 7, 21)
    recent = [date(2026, 7, 1), date(2026, 6, 20), date(2026, 6, 5), date(2026, 5, 25)]
    assert rust_factor(recent, today) == 1.0


def test_rust_factor_scales_down_with_fewer_matches():
    today = date(2026, 7, 21)
    recent = [date(2026, 7, 1)]  # 1 match in last 60 days
    assert rust_factor(recent, today) == 0.25


def test_rust_factor_zero_when_veteran_has_history_but_no_recent_matches():
    today = date(2026, 7, 21)
    old_only = [date(2020, 1, 1), date(2019, 1, 1)]  # long career, nothing recent
    assert rust_factor(old_only, today) == 0.0


def test_rust_factor_neutral_for_brand_new_player_with_no_history():
    today = date(2026, 7, 21)
    assert rust_factor([], today) == 1.0


def test_rust_factor_ignores_matches_outside_60_day_window():
    today = date(2026, 7, 21)
    just_outside = [date(2026, 5, 1)]  # >60 days before 2026-07-21
    assert rust_factor(just_outside, today) == 0.0


# ── EloHistoryTracker ─────────────────────────────────────────────────────────

def test_tracker_returns_none_with_fewer_than_five_snapshots():
    tracker = EloHistoryTracker()
    for v in [1500, 1510, 1520]:
        tracker.record("A", "clay", v)
    assert tracker.rolling_elo("A", "clay") is None


def test_tracker_returns_weighted_average_once_five_snapshots_recorded():
    tracker = EloHistoryTracker()
    for v in [1500, 1500, 1500, 1500, 1600]:
        tracker.record("A", "clay", v)
    rolling = tracker.rolling_elo("A", "clay")
    # Weighted toward the most recent (1600), so above the flat 1520 average
    assert rolling is not None
    assert rolling > 1520


def test_tracker_keeps_surfaces_independent():
    tracker = EloHistoryTracker()
    for v in [1500, 1500, 1500, 1500, 1500]:
        tracker.record("A", "clay", v)
    assert tracker.rolling_elo("A", "hard") is None


def test_tracker_caps_window_at_fifteen_snapshots():
    tracker = EloHistoryTracker()
    for v in [1000] * 15:
        tracker.record("A", "clay", v)
    for v in [2000] * 5:  # pushes out the oldest 1000s
        tracker.record("A", "clay", v)
    # window is now 10x1000 + 5x2000 capped at 15 -> last 15 = 10x1000, 5x2000
    rolling = tracker.rolling_elo("A", "clay")
    assert rolling > 1000  # confirms old values were evicted, pulling avg up
    assert rolling < 2000


# ── calculate_decay_features ─────────────────────────────────────────────────

def test_new_player_gets_neutral_decay_features():
    tracker = EloHistoryTracker()
    feats = calculate_decay_features(
        historical_elo_surface=1500.0,
        tracker=tracker,
        player="Newcomer",
        surface="clay",
        match_dates=[],
        current_date=date(2026, 7, 21),
        player_age=None,
    )
    assert feats["rolling_elo_diff"] == 0.0
    assert feats["age_multiplier"] == 1.0
    assert feats["rust_factor"] == 1.0
    assert feats["adjusted_elo_surface"] == 1500.0


def test_aging_inactive_veteran_gets_penalized_adjusted_elo():
    tracker = EloHistoryTracker()
    # Long, strong clay history but nothing recent
    old_dates = [date(2015, 1, 1), date(2016, 1, 1), date(2018, 1, 1)]
    feats = calculate_decay_features(
        historical_elo_surface=1700.0,
        tracker=tracker,
        player="Veteran",
        surface="clay",
        match_dates=old_dates,
        current_date=date(2026, 7, 21),
        player_age=41,
    )
    # age_multiplier=0.725, rust_factor=0.0 (no matches in last 60 days)
    assert feats["age_multiplier"] == 0.725
    assert feats["rust_factor"] == 0.0
    assert feats["adjusted_elo_surface"] == 0.0  # 1700 * 0.725 * 0.0


# ── fatigue_multiplier ────────────────────────────────────────────────────────
# Hand-computed values — see docs/superpowers/specs/2026-07-28-fatigue-workload-design.md

class TestFatigueMultiplier:
    def test_no_history_returns_neutral(self):
        assert fatigue_multiplier([], date(2023, 6, 1)) == 1.0

    def test_single_light_match_barely_reduces_multiplier(self):
        # 1 match/2 sets, 3 days ago (inside both windows).
        # load_7d  = 0.5*min(1,1/3) + 0.5*min(1,2/6)  = 1/3
        # load_14d = 0.5*min(1,1/5) + 0.5*min(1,2/10) = 0.2
        # fatigue_load = 0.6*(1/3) + 0.4*0.2 = 0.28
        # multiplier = 1 - 0.15*0.28 = 0.958
        history = [(date(2023, 5, 29), 2)]
        assert fatigue_multiplier(history, date(2023, 6, 1)) == pytest.approx(0.958)

    def test_max_load_in_both_windows_hits_penalty_cap(self):
        # 3 matches/6 sets inside the last 7 days (saturates the 7d window),
        # plus 2 more matches/4 sets between day 8-14 (saturates the 14d
        # window too: 5 matches/10 sets total). Both windows load=1.0.
        # fatigue_load = 0.6*1.0 + 0.4*1.0 = 1.0
        # multiplier = 1 - 0.15*1.0 = 0.85 (the maximum possible penalty)
        history = [
            (date(2023, 5, 20), 2), (date(2023, 5, 22), 2),  # 8-10 days ago
            (date(2023, 5, 27), 1), (date(2023, 5, 28), 2), (date(2023, 5, 30), 3),  # <=7 days ago
        ]
        assert fatigue_multiplier(history, date(2023, 6, 1)) == pytest.approx(0.85)

    def test_match_outside_7d_window_still_counts_in_14d(self):
        # 1 match/2 sets, 10 days ago: excluded from the 7d window (>7),
        # included in the 14d window (<=14).
        # load_7d = 0; load_14d = 0.5*min(1,1/5) + 0.5*min(1,2/10) = 0.2
        # fatigue_load = 0.6*0 + 0.4*0.2 = 0.08
        # multiplier = 1 - 0.15*0.08 = 0.988
        history = [(date(2023, 5, 22), 2)]
        assert fatigue_multiplier(history, date(2023, 6, 1)) == pytest.approx(0.988)

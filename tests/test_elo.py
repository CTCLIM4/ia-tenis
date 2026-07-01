import pytest
from datetime import date
from src.models.elo import EloSystem


def test_new_player_starts_at_initial_rating():
    elo = EloSystem()
    assert elo.get_general_rating("Unknown Player") == 1500.0


def test_winner_gains_rating_loser_loses():
    elo = EloSystem()
    elo.update("A", "B", "hard", date(2023, 1, 1))
    assert elo.get_general_rating("A") > 1500.0
    assert elo.get_general_rating("B") < 1500.0


def test_ratings_are_conserved_equal_players():
    elo = EloSystem()
    elo.update("A", "B", "hard", date(2023, 1, 1))
    total = elo.get_general_rating("A") + elo.get_general_rating("B")
    assert abs(total - 3000.0) < 1e-9


def test_upset_win_gives_larger_elo_gain():
    elo = EloSystem()
    d = date(2022, 6, 1)
    for _ in range(8):
        elo.update("Strong", "Weak", "hard", d)

    weak_before = elo.get_general_rating("Weak")
    # Weak player (underdog) wins — should gain > half of K
    prob_weak_wins = elo.update("Weak", "Strong", "hard", date(2022, 7, 1))
    gain = elo.get_general_rating("Weak") - weak_before

    assert prob_weak_wins < 0.5          # Elo says strong player was favourite
    assert gain > elo.k * 0.5           # upset → bigger gain than 50-50


def test_surface_ratings_independent():
    elo = EloSystem()
    elo.update("A", "B", "clay", date(2023, 1, 1))
    assert elo.get_surface_rating("A", "hard") == 1500.0  # hard untouched
    assert elo.get_surface_rating("A", "clay") != 1500.0  # clay changed


def test_blend_weight_increases_with_surface_experience():
    elo = EloSystem()
    w0 = elo.get_blend_weight("A", "clay")
    elo.update("A", "B", "clay", date(2023, 1, 1))
    w1 = elo.get_blend_weight("A", "clay")
    assert w1 > w0
    assert w0 == 0.0   # zero experience at start


def test_yearly_decay_regresses_towards_1500():
    elo = EloSystem()
    for i in range(20):
        elo.update("A", "B", "hard", date(2022, (i % 11) + 1, 1))
    rating_end_2022 = elo.get_general_rating("A")
    assert rating_end_2022 > 1500.0

    # First match in 2023 triggers decay
    elo.update("A", "B", "hard", date(2023, 1, 15))
    rating_after_decay = elo.get_general_rating("A")

    # Decayed but still above 1500 (decay only regresses partially)
    assert 1500.0 < rating_after_decay < rating_end_2022


def test_k_factor_decreases_with_experience():
    elo = EloSystem()
    k_new = elo.get_k("new_player")   # 0 matches
    elo.match_counts["veteran"] = 150
    k_veteran = elo.get_k("veteran")
    assert k_veteran < k_new


def test_expected_score_sums_to_one():
    elo = EloSystem()
    p_ab = elo.expected_score(1600.0, 1400.0)
    p_ba = elo.expected_score(1400.0, 1600.0)
    assert abs(p_ab + p_ba - 1.0) < 1e-10
    assert p_ab > 0.5


def test_update_returns_win_probability_for_winner():
    elo = EloSystem()
    prob = elo.update("A", "B", "hard", date(2023, 1, 1))
    assert 0.0 < prob < 1.0

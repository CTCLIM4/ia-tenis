"""Tests for src/player_matcher.py — matching Odds API player names to the
canonical names known to the Elo/features dataset."""
from __future__ import annotations

from src.player_matcher import match_player_name


class TestExactMatch:
    def test_exact_name_returns_itself(self):
        canonical = ["Alexander Zverev", "Novak Djokovic"]
        assert match_player_name("Novak Djokovic", canonical) == "Novak Djokovic"


class TestAbbreviatedFormats:
    def test_first_initial_first_format(self):
        canonical = ["Alexander Zverev", "Novak Djokovic"]
        assert match_player_name("A. Zverev", canonical) == "Alexander Zverev"

    def test_surname_first_format(self):
        canonical = ["Aryna Sabalenka", "Iga Swiatek"]
        assert match_player_name("Sabalenka A.", canonical) == "Aryna Sabalenka"

    def test_accented_name_variant(self):
        canonical = ["Carlos Alcaraz Garfia"]
        assert match_player_name("Carlos Alcaraz Garfía", canonical) == "Carlos Alcaraz Garfia"


class TestDisambiguation:
    def test_same_surname_different_initial_picks_correct_sibling(self):
        # Zverev brothers: surname alone must not be enough.
        canonical = ["Alexander Zverev", "Mischa Zverev"]
        assert match_player_name("A. Zverev", canonical) == "Alexander Zverev"
        assert match_player_name("M. Zverev", canonical) == "Mischa Zverev"

    def test_same_surname_same_initial_is_ambiguous_returns_none(self):
        # Pliskova sisters: both first names start with "K" — a single
        # initial can't disambiguate, and guessing wrong is worse than
        # returning no match.
        canonical = ["Karolina Pliskova", "Kristyna Pliskova"]
        assert match_player_name("K. Pliskova", canonical) is None


class TestNoMatch:
    def test_unknown_player_returns_none(self):
        canonical = ["Novak Djokovic", "Jannik Sinner"]
        assert match_player_name("Carlos Alcaraz", canonical) is None

    def test_empty_canonical_list_returns_none(self):
        assert match_player_name("Novak Djokovic", []) is None

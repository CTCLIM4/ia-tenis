"""Tests for src/odds_api.py — name matching, no real HTTP calls."""
from __future__ import annotations

from src.odds_api import MatchOdds, _normalize_name, find_match_odds


# ── _normalize_name ──────────────────────────────────────────────────────────

class TestNormalizeName:
    def test_lowercases(self):
        assert _normalize_name("Djokovic") == "djokovic"

    def test_strips_accents(self):
        assert _normalize_name("Alcaraz Garfia") == _normalize_name("Alcaraz Garfía")

    def test_strips_periods(self):
        assert _normalize_name("Sabalenka A.") == "sabalenka a"

    def test_collapses_whitespace(self):
        assert _normalize_name("Novak   Djokovic") == "novak djokovic"


# ── find_match_odds ───────────────────────────────────────────────────────────

def _event(home, away, bookmaker_key="bet365", prices=None):
    prices = prices or {home: 1.50, away: 2.60}
    return {
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": bookmaker_key,
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": prices[home]},
                            {"name": away, "price": prices[away]},
                        ],
                    }
                ],
            }
        ],
    }


class TestFindMatchOdds:
    def test_direct_order_match(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Novak Djokovic", "Jannik Sinner")
        assert result == MatchOdds(
            odds_a=1.50, odds_b=2.60,
            matched_home="Novak Djokovic", matched_away="Jannik Sinner",
        )

    def test_reversed_order_match(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Jannik Sinner", "Novak Djokovic")
        assert result == MatchOdds(
            odds_a=2.60, odds_b=1.50,
            matched_home="Novak Djokovic", matched_away="Jannik Sinner",
        )

    def test_no_match_returns_none(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Carlos Alcaraz", "Daniil Medvedev")
        assert result is None

    def test_bookmaker_absent_returns_none(self):
        events = [_event("Novak Djokovic", "Jannik Sinner", bookmaker_key="pinnacle")]
        result = find_match_odds(events, "Novak Djokovic", "Jannik Sinner", bookmaker="bet365")
        assert result is None

    def test_matches_regardless_of_periods_and_accents(self):
        events = [_event("Sabalenka A.", "Swiatek I.")]
        result = find_match_odds(events, "Aryna Sabalenka", "Iga Swiatek")
        assert result is not None
        assert result.matched_home == "Sabalenka A."

"""Tests for src/odds_api.py — name matching, no real HTTP calls."""
from __future__ import annotations

import json

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


from datetime import datetime, timedelta, timezone


class TestGetEvents:
    def test_fresh_cache_is_used_without_refetch(self, tmp_path, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.setattr(odds_api, "CACHE_DIR", tmp_path)

        cache_file = tmp_path / "atp.json"
        cache_file.write_text(
            json.dumps({
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "events": [{"home_team": "A", "away_team": "B", "bookmakers": []}],
            }),
            encoding="utf-8",
        )

        def _boom(tour, api_key):
            raise AssertionError("should not refetch when cache is fresh")

        monkeypatch.setattr(odds_api, "fetch_odds_events", _boom)
        events = odds_api.get_events("atp", api_key="fake", cache_minutes=15)
        assert events == [{"home_team": "A", "away_team": "B", "bookmakers": []}]

    def test_stale_cache_triggers_refetch(self, tmp_path, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.setattr(odds_api, "CACHE_DIR", tmp_path)

        cache_file = tmp_path / "atp.json"
        old_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        cache_file.write_text(
            json.dumps({"fetched_at": old_time.isoformat(), "events": []}),
            encoding="utf-8",
        )

        fresh_events = [{"home_team": "X", "away_team": "Y", "bookmakers": []}]
        monkeypatch.setattr(odds_api, "fetch_odds_events", lambda tour, api_key: fresh_events)

        events = odds_api.get_events("atp", api_key="fake", cache_minutes=15)
        assert events == fresh_events

    def test_missing_cache_triggers_fetch_and_saves(self, tmp_path, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.setattr(odds_api, "CACHE_DIR", tmp_path)

        fresh_events = [{"home_team": "X", "away_team": "Y", "bookmakers": []}]
        monkeypatch.setattr(odds_api, "fetch_odds_events", lambda tour, api_key: fresh_events)

        events = odds_api.get_events("atp", api_key="fake", cache_minutes=15)
        assert events == fresh_events
        assert (tmp_path / "atp.json").exists()

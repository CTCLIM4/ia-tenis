"""Tests for src/odds_api.py — name matching, no real HTTP calls."""
from __future__ import annotations

import json

from src.odds_api import (
    DEFAULT_BOOKMAKER,
    MatchOdds,
    _first_initial,
    _normalize_name,
    find_match_odds,
    list_tennis_sport_keys,
)


class TestDefaultBookmaker:
    def test_is_pinnacle_not_bet365(self):
        # bet365 does not appear in The Odds API's feed for any region
        # (verified against a real API key on 2026-07-28, tennis_atp_washington_open,
        # regions=eu,uk,us,us2,au combined) — a bookmaker=bet365 default silently
        # finds zero events forever. pinnacle is used as the reference "sharp"
        # book instead (lowest vig, closest proxy to true probability).
        assert DEFAULT_BOOKMAKER == "pinnacle"


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


# ── _first_initial ───────────────────────────────────────────────────────────

class TestFirstInitial:
    def test_extracts_from_full_name_first_last(self):
        assert _first_initial("Aryna Sabalenka") == "a"

    def test_extracts_from_abbreviated_last_first(self):
        assert _first_initial("Sabalenka A.") == "a"

    def test_empty_for_single_token_name(self):
        assert _first_initial("Djokovic") == ""

    def test_strips_accents(self):
        assert _first_initial("Alcaraz Garfía") == "a"

    def test_different_first_names_give_different_initials(self):
        # Zverev brothers: same surname, must not resolve to the same initial.
        assert _first_initial("Alexander Zverev") != _first_initial("Mischa Zverev")


# ── find_match_odds ───────────────────────────────────────────────────────────

def _event(home, away, bookmaker_key=DEFAULT_BOOKMAKER, prices=None):
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


def _multi_bk_event(home, away, prices_by_bookmaker):
    """prices_by_bookmaker: {bookmaker_key: {home: price, away: price}}"""
    return {
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": bk,
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": home, "price": prices[home]},
                        {"name": away, "price": prices[away]},
                    ],
                }],
            }
            for bk, prices in prices_by_bookmaker.items()
        ],
    }


class TestBestPrice:
    def test_picks_max_price_per_side_independently(self):
        from src.odds_api import _best_price
        event = _multi_bk_event("Djokovic", "Sinner", {
            "pinnacle": {"Djokovic": 1.50, "Sinner": 2.60},
            "bet365":   {"Djokovic": 1.60, "Sinner": 2.50},
        })
        odds_home, odds_away, bk_home, bk_away = _best_price(event, allowed_bookmakers=None)
        assert odds_home == 1.60 and bk_home == "bet365"
        assert odds_away == 2.60 and bk_away == "pinnacle"

    def test_allowed_bookmakers_filters_out_others(self):
        from src.odds_api import _best_price
        event = _multi_bk_event("Djokovic", "Sinner", {
            "pinnacle": {"Djokovic": 1.50, "Sinner": 2.60},
            "bet365":   {"Djokovic": 1.60, "Sinner": 2.50},
        })
        odds_home, odds_away, bk_home, bk_away = _best_price(event, allowed_bookmakers={"pinnacle"})
        assert odds_home == 1.50 and bk_home == "pinnacle"
        assert odds_away == 2.60 and bk_away == "pinnacle"

    def test_empty_allowed_bookmakers_set_means_none_allowed(self):
        from src.odds_api import _best_price
        event = _multi_bk_event("Djokovic", "Sinner", {
            "pinnacle": {"Djokovic": 1.50, "Sinner": 2.60},
        })
        result = _best_price(event, allowed_bookmakers=set())
        assert result == (None, None, None, None)

    def test_no_bookmakers_quote_event_returns_all_none(self):
        from src.odds_api import _best_price
        event = _multi_bk_event("Djokovic", "Sinner", {
            "bet365": {"Djokovic": 1.60, "Sinner": 2.50},
        })
        result = _best_price(event, allowed_bookmakers={"pinnacle"})
        assert result == (None, None, None, None)

    def test_ignores_non_h2h_markets(self):
        from src.odds_api import _best_price
        event = {
            "home_team": "Djokovic", "away_team": "Sinner",
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{"key": "totals", "outcomes": []}],
            }],
        }
        result = _best_price(event, allowed_bookmakers=None)
        assert result == (None, None, None, None)


class TestFindMatchOdds:
    def test_direct_order_match(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Novak Djokovic", "Jannik Sinner")
        assert result == MatchOdds(
            odds_a=1.50, odds_b=2.60,
            matched_home="Novak Djokovic", matched_away="Jannik Sinner",
            bookmaker_a="pinnacle", bookmaker_b="pinnacle",
        )

    def test_reversed_order_match(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Jannik Sinner", "Novak Djokovic")
        assert result == MatchOdds(
            odds_a=2.60, odds_b=1.50,
            matched_home="Novak Djokovic", matched_away="Jannik Sinner",
            bookmaker_a="pinnacle", bookmaker_b="pinnacle",
        )

    def test_no_match_returns_none(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Carlos Alcaraz", "Daniil Medvedev")
        assert result is None

    def test_bookmaker_absent_returns_none(self):
        events = [_event("Novak Djokovic", "Jannik Sinner", bookmaker_key="pinnacle")]
        result = find_match_odds(
            events, "Novak Djokovic", "Jannik Sinner", allowed_bookmakers={"bet365"},
        )
        assert result is None

    def test_returns_none_when_only_one_side_has_an_allowed_quote(self):
        # pinnacle only quotes the home side (e.g. away outcome missing from
        # its response) -- odds_home resolves but odds_away stays None, so
        # the `or` in `if odds_home is None or odds_away is None` must fire.
        event = {
            "home_team": "Novak Djokovic", "away_team": "Jannik Sinner",
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [{"name": "Novak Djokovic", "price": 1.50}],
                }],
            }],
        }
        result = find_match_odds([event], "Novak Djokovic", "Jannik Sinner")
        assert result is None

    def test_independent_bookmaker_per_side(self):
        events = [_multi_bk_event("Novak Djokovic", "Jannik Sinner", {
            "pinnacle": {"Novak Djokovic": 1.50, "Jannik Sinner": 2.60},
            "bet365":   {"Novak Djokovic": 1.55, "Jannik Sinner": 2.55},
        })]
        result = find_match_odds(events, "Novak Djokovic", "Jannik Sinner")
        assert result.odds_a == 1.55 and result.bookmaker_a == "bet365"
        assert result.odds_b == 2.60 and result.bookmaker_b == "pinnacle"

    def test_independent_bookmaker_per_side_with_reversed_order(self):
        events = [_multi_bk_event("Novak Djokovic", "Jannik Sinner", {
            "pinnacle": {"Novak Djokovic": 1.50, "Jannik Sinner": 2.60},
            "bet365":   {"Novak Djokovic": 1.55, "Jannik Sinner": 2.55},
        })]
        # player_a=Sinner (away team), player_b=Djokovic (home team) -> reversed order branch
        result = find_match_odds(events, "Jannik Sinner", "Novak Djokovic")
        assert result.odds_a == 2.60 and result.bookmaker_a == "pinnacle"
        assert result.odds_b == 1.55 and result.bookmaker_b == "bet365"

    def test_matches_regardless_of_periods_and_accents(self):
        events = [_event("Sabalenka A.", "Swiatek I.")]
        result = find_match_odds(events, "Aryna Sabalenka", "Iga Swiatek")
        assert result is not None
        assert result.matched_home == "Sabalenka A."

    def test_rejects_surname_match_with_different_first_initial(self):
        # Zverev brothers: same surname, different first name. Querying for
        # Alexander must not silently return odds for a Mischa Zverev event —
        # surname alone was enough to false-positive-match before this check.
        events = [_event("Mischa Zverev", "Novak Djokovic")]
        result = find_match_odds(events, "Alexander Zverev", "Novak Djokovic")
        assert result is None

    def test_accepts_surname_match_when_first_initial_also_matches(self):
        events = [_event("Alexander Zverev", "Novak Djokovic")]
        result = find_match_odds(events, "Alexander Zverev", "Novak Djokovic")
        assert result is not None

    def test_first_initial_check_skipped_when_name_has_no_first_name_info(self):
        # A bare surname-only name carries no initial to compare — must fall
        # back to surname-only matching rather than always rejecting.
        events = [_event("Zverev", "Novak Djokovic")]
        result = find_match_odds(events, "Zverev", "Novak Djokovic")
        assert result is not None


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


# ── list_tennis_sport_keys ───────────────────────────────────────────────────

class TestListTennisSportKeys:
    def test_filters_to_tennis_only(self):
        sports_index = [
            {"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon"},
            {"key": "basketball_nba", "title": "NBA"},
        ]
        result = list_tennis_sport_keys(sports_index)
        assert result == [{"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon", "tour": "atp"}]

    def test_infers_atp_and_wta_tour(self):
        sports_index = [
            {"key": "tennis_atp_us_open", "title": "ATP US Open"},
            {"key": "tennis_wta_us_open", "title": "WTA US Open"},
        ]
        result = list_tennis_sport_keys(sports_index)
        tours = {s["key"]: s["tour"] for s in result}
        assert tours == {"tennis_atp_us_open": "atp", "tennis_wta_us_open": "wta"}

    def test_excludes_non_atp_wta_tennis_keys(self):
        # e.g. an ITF key, if The Odds API ever lists one — no model for it.
        sports_index = [{"key": "tennis_itf_men", "title": "ITF Men"}]
        assert list_tennis_sport_keys(sports_index) == []

    def test_empty_index_returns_empty_list(self):
        assert list_tennis_sport_keys([]) == []


# ── resolve_allowed_bookmakers ───────────────────────────────────────────────

class TestResolveAllowedBookmakers:
    def test_explicit_bookmaker_collapses_to_single_item_set(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.delenv("ALLOWED_BOOKMAKERS", raising=False)
        assert resolve_allowed_bookmakers("pinnacle") == {"pinnacle"}

    def test_explicit_bookmaker_overrides_env_var(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.setenv("ALLOWED_BOOKMAKERS", "bet365,williamhill")
        assert resolve_allowed_bookmakers("pinnacle") == {"pinnacle"}

    def test_no_explicit_bookmaker_parses_env_var(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.setenv("ALLOWED_BOOKMAKERS", "bet365, williamhill ,pinnacle")
        assert resolve_allowed_bookmakers(None) == {"bet365", "williamhill", "pinnacle"}

    def test_unset_env_var_and_no_explicit_means_allow_all(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.delenv("ALLOWED_BOOKMAKERS", raising=False)
        assert resolve_allowed_bookmakers(None) is None

    def test_empty_env_var_means_allow_all(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.setenv("ALLOWED_BOOKMAKERS", "")
        assert resolve_allowed_bookmakers(None) is None

    def test_empty_explicit_string_falls_back_to_env(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.setenv("ALLOWED_BOOKMAKERS", "bet365")
        assert resolve_allowed_bookmakers("") == {"bet365"}


# ── regions configuration ────────────────────────────────────────────────────

class TestRegionsParam:
    def test_default_regions_when_env_unset(self, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.delenv("ODDS_API_REGIONS", raising=False)
        captured = {}

        class _FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"[]"

        def _fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResp()

        monkeypatch.setattr(odds_api.urllib.request, "urlopen", _fake_urlopen)
        odds_api.fetch_odds_events_by_key("tennis_atp_wimbledon", "fake-key")
        assert "regions=eu,uk,us" in captured["url"]

    def test_custom_regions_from_env(self, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.setenv("ODDS_API_REGIONS", "au,us2")
        captured = {}

        class _FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"[]"

        def _fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResp()

        monkeypatch.setattr(odds_api.urllib.request, "urlopen", _fake_urlopen)
        odds_api.fetch_odds_events_by_key("tennis_atp_wimbledon", "fake-key")
        assert "regions=au,us2" in captured["url"]

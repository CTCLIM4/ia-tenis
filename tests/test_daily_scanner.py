"""Tests for src/daily_scanner.py — automated match discovery + batch value-bet scan."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.daily_scanner import (
    EvaluatedMatch,
    _extract_h2h_odds,
    _is_within_window,
    discover_matches,
    evaluate_matches,
    print_value_bets_table,
)
from src.value_analysis import SUSPICIOUS_EDGE_THRESHOLD


# ── _is_within_window ─────────────────────────────────────────────────────────

class TestIsWithinWindow:
    def test_now_is_within_window(self):
        now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        assert _is_within_window("2026-07-27T12:00:00Z", days_ahead=1, now=now)

    def test_within_days_ahead_is_included(self):
        now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        assert _is_within_window("2026-07-28T10:00:00Z", days_ahead=1, now=now)

    def test_beyond_days_ahead_is_excluded(self):
        now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        assert not _is_within_window("2026-07-30T10:00:00Z", days_ahead=1, now=now)

    def test_past_commence_time_is_excluded(self):
        now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        assert not _is_within_window("2026-07-26T10:00:00Z", days_ahead=1, now=now)


# ── _extract_h2h_odds ─────────────────────────────────────────────────────────

def _event(home="Novak Djokovic", away="Jannik Sinner", bookmaker_key="bet365",
           prices=None, commence_time="2026-07-27T12:00:00Z"):
    prices = prices or {home: 1.50, away: 2.60}
    return {
        "home_team": home,
        "away_team": away,
        "commence_time": commence_time,
        "bookmakers": [{
            "key": bookmaker_key,
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": home, "price": prices[home]},
                    {"name": away, "price": prices[away]},
                ],
            }],
        }],
    }


class TestExtractH2hOdds:
    def test_extracts_prices_for_configured_bookmaker(self):
        event = _event(prices={"Novak Djokovic": 1.5, "Jannik Sinner": 2.6})
        assert _extract_h2h_odds(event, "bet365") == (1.5, 2.6)

    def test_returns_none_when_bookmaker_absent(self):
        event = _event(bookmaker_key="pinnacle")
        assert _extract_h2h_odds(event, "bet365") is None


# ── discover_matches ──────────────────────────────────────────────────────────

class TestDiscoverMatches:
    def test_discovers_and_matches_full_pipeline(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "fetch_sports_index", lambda api_key: [
            {"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon"},
        ])
        monkeypatch.setattr(
            scanner, "fetch_odds_events_by_key",
            lambda sport_key, api_key: [_event("Novak Djokovic", "Jannik Sinner")],
        )
        now = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(scanner, "_now", lambda: now)

        matches = discover_matches(
            api_key="fake",
            canonical_names_by_tour={"atp": ["Novak Djokovic", "Jannik Sinner"], "wta": []},
            days_ahead=1,
        )
        assert len(matches) == 1
        m = matches[0]
        assert m.tour == "atp"
        assert m.surface == "grass"
        assert m.player_a == "Novak Djokovic"
        assert m.player_b == "Jannik Sinner"
        assert m.odds_a == 1.5 and m.odds_b == 2.6
        assert m.match_date == date(2026, 7, 27)

    def test_skips_event_outside_time_window(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "fetch_sports_index", lambda api_key: [
            {"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon"},
        ])
        monkeypatch.setattr(
            scanner, "fetch_odds_events_by_key",
            lambda sport_key, api_key: [_event(commence_time="2026-08-15T12:00:00Z")],
        )
        now = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(scanner, "_now", lambda: now)

        matches = discover_matches(
            api_key="fake",
            canonical_names_by_tour={"atp": ["Novak Djokovic", "Jannik Sinner"], "wta": []},
            days_ahead=1,
        )
        assert matches == []

    def test_skips_unresolvable_surface(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "fetch_sports_index", lambda api_key: [
            {"key": "tennis_atp_fictional", "title": "ATP Fictional Open 3000"},
        ])
        monkeypatch.setattr(
            scanner, "fetch_odds_events_by_key",
            lambda sport_key, api_key: [_event()],
        )
        now = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(scanner, "_now", lambda: now)

        matches = discover_matches(
            api_key="fake",
            canonical_names_by_tour={"atp": ["Novak Djokovic", "Jannik Sinner"], "wta": []},
            days_ahead=1,
        )
        assert matches == []

    def test_skips_unmatched_player(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "fetch_sports_index", lambda api_key: [
            {"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon"},
        ])
        monkeypatch.setattr(
            scanner, "fetch_odds_events_by_key",
            lambda sport_key, api_key: [_event("Novak Djokovic", "Unknown Player")],
        )
        now = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(scanner, "_now", lambda: now)

        matches = discover_matches(
            api_key="fake",
            canonical_names_by_tour={"atp": ["Novak Djokovic", "Jannik Sinner"], "wta": []},
            days_ahead=1,
        )
        assert matches == []


# ── evaluate_matches ──────────────────────────────────────────────────────────

def _val(edge=0.1, ev=0.15, kelly=0.05, has_value=True):
    return {"implied_prob": 0.5, "edge": edge, "ev": ev, "kelly_fraction": kelly, "has_value": has_value}


def _pred(elo_found_a=True, elo_found_b=True):
    return {
        "player_a": "Novak Djokovic", "player_b": "Jannik Sinner",
        "p_a_raw": 0.6, "p_a_cal": 0.6, "p_b_raw": 0.4, "p_b_cal": 0.4,
        "rank_a": 1, "rank_b": 2, "rank_a_source": "auto", "rank_b_source": "auto",
        "features": {}, "elo_a": 2000, "elo_b": 1900,
        "elo_found_a": elo_found_a, "elo_found_b": elo_found_b,
    }


class TestEvaluateMatches:
    def test_marks_elo_ok_and_has_value(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "predict_match", lambda *a, **k: _pred())
        monkeypatch.setattr(scanner, "calculate_value", lambda p, o: _val(edge=0.1, has_value=True))

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Wimbledon", surface="grass",
            match_date=date(2026, 7, 27), player_a="Novak Djokovic", player_b="Jannik Sinner",
            odds_a=1.5, odds_b=2.6, raw_home="Novak Djokovic", raw_away="Jannik Sinner",
        )
        results = evaluate_matches([m], models={"atp": (None, None, None, None, None, None)})
        assert len(results) == 1
        r = results[0]
        assert r.elo_ok is True
        assert r.suspicious is False
        assert r.has_value is True

    def test_flags_missing_elo(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "predict_match", lambda *a, **k: _pred(elo_found_a=False))
        monkeypatch.setattr(scanner, "calculate_value", lambda p, o: _val())

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Wimbledon", surface="grass",
            match_date=date(2026, 7, 27), player_a="Novak Djokovic", player_b="Jannik Sinner",
            odds_a=1.5, odds_b=2.6, raw_home="Novak Djokovic", raw_away="Jannik Sinner",
        )
        results = evaluate_matches([m], models={"atp": (None, None, None, None, None, None)})
        assert results[0].elo_ok is False

    def test_flags_suspicious_edge(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "predict_match", lambda *a, **k: _pred())
        monkeypatch.setattr(
            scanner, "calculate_value",
            lambda p, o: _val(edge=SUSPICIOUS_EDGE_THRESHOLD + 0.05),
        )

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Wimbledon", surface="grass",
            match_date=date(2026, 7, 27), player_a="Novak Djokovic", player_b="Jannik Sinner",
            odds_a=1.5, odds_b=2.6, raw_home="Novak Djokovic", raw_away="Jannik Sinner",
        )
        results = evaluate_matches(
            [m], models={"atp": (None, None, None, None, None, None)}, halt_on_suspicious=True,
        )
        assert results[0].suspicious is True

    def test_no_value_when_edge_not_positive(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "predict_match", lambda *a, **k: _pred())
        monkeypatch.setattr(
            scanner, "calculate_value",
            lambda p, o: _val(edge=-0.05, has_value=False),
        )

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Wimbledon", surface="grass",
            match_date=date(2026, 7, 27), player_a="Novak Djokovic", player_b="Jannik Sinner",
            odds_a=1.5, odds_b=2.6, raw_home="Novak Djokovic", raw_away="Jannik Sinner",
        )
        results = evaluate_matches([m], models={"atp": (None, None, None, None, None, None)})
        assert results[0].has_value is False


# ── print_value_bets_table ────────────────────────────────────────────────────

class TestPrintValueBetsTable:
    def test_empty_list_prints_no_value_bets_message(self, capsys):
        print_value_bets_table([])
        assert "no se encontraron" in capsys.readouterr().out.lower()

    def test_prints_row_for_valued_side(self, capsys):
        import src.daily_scanner as scanner

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Wimbledon", surface="grass",
            match_date=date(2026, 7, 27), player_a="Novak Djokovic", player_b="Jannik Sinner",
            odds_a=1.5, odds_b=2.6, raw_home="Novak Djokovic", raw_away="Jannik Sinner",
        )
        r = EvaluatedMatch(
            match=m, pred=_pred(), val_a=_val(has_value=True), val_b=_val(has_value=False),
            elo_ok=True, suspicious=False,
        )
        print_value_bets_table([r])
        out = capsys.readouterr().out
        assert "Novak Djokovic" in out
        assert "Wimbledon" in out

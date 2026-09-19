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
from src.config import MAX_SUSPICIOUS_EDGE
from src.odds_api import DEFAULT_BOOKMAKER


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
    def test_extracts_prices_for_allowed_bookmaker(self):
        event = _event(prices={"Novak Djokovic": 1.5, "Jannik Sinner": 2.6}, bookmaker_key="bet365")
        assert _extract_h2h_odds(event, {"bet365"}) == (1.5, 2.6, "bet365", "bet365")

    def test_returns_none_when_no_allowed_bookmaker_quotes_it(self):
        event = _event(bookmaker_key="pinnacle")
        assert _extract_h2h_odds(event, {"bet365"}) is None

    def test_none_allowed_bookmakers_means_all_allowed(self):
        event = _event(prices={"Novak Djokovic": 1.5, "Jannik Sinner": 2.6}, bookmaker_key="pinnacle")
        assert _extract_h2h_odds(event, None) == (1.5, 2.6, "pinnacle", "pinnacle")


# ── discover_matches ──────────────────────────────────────────────────────────

class TestDiscoverMatches:
    def test_discovers_and_matches_full_pipeline(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "fetch_sports_index", lambda api_key: [
            {"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon"},
        ])
        monkeypatch.setattr(
            scanner, "fetch_odds_events_by_key",
            lambda sport_key, api_key: [
                _event("Novak Djokovic", "Jannik Sinner", bookmaker_key=DEFAULT_BOOKMAKER)
            ],
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
        assert m.bookmaker_a == DEFAULT_BOOKMAKER and m.bookmaker_b == DEFAULT_BOOKMAKER
        assert m.match_date == date(2026, 7, 27)

    def test_match_date_uses_lima_calendar_day_not_utc(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "fetch_sports_index", lambda api_key: [
            {"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon"},
        ])
        monkeypatch.setattr(
            scanner, "fetch_odds_events_by_key",
            lambda sport_key, api_key: [
                _event("Novak Djokovic", "Jannik Sinner", bookmaker_key=DEFAULT_BOOKMAKER,
                       commence_time="2026-07-27T02:00:00Z")
            ],
        )
        now = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(scanner, "_now", lambda: now)

        matches = discover_matches(
            api_key="fake",
            canonical_names_by_tour={"atp": ["Novak Djokovic", "Jannik Sinner"], "wta": []},
            days_ahead=1,
        )
        assert len(matches) == 1
        # 2026-07-27T02:00Z is 2026-07-26 21:00 in America/Lima (UTC-5) —
        # match_date must reflect the Lima calendar day, not the UTC one.
        assert matches[0].match_date == date(2026, 7, 26)

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


def _pred(elo_found_a=True, elo_found_b=True, matches_a=100, matches_b=100):
    return {
        "player_a": "Novak Djokovic", "player_b": "Jannik Sinner",
        "p_a_raw": 0.6, "p_a_cal": 0.6, "p_b_raw": 0.4, "p_b_cal": 0.4,
        "rank_a": 1, "rank_b": 2, "rank_a_source": "auto", "rank_b_source": "auto",
        "features": {}, "elo_a": 2000, "elo_b": 1900,
        "elo_found_a": elo_found_a, "elo_found_b": elo_found_b,
        "matches_a": matches_a, "matches_b": matches_b,
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
            lambda p, o: _val(edge=MAX_SUSPICIOUS_EDGE + 0.05),
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

    def test_flags_low_sample_when_either_player_below_threshold(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "predict_match", lambda *a, **k: _pred(matches_a=14, matches_b=302))
        monkeypatch.setattr(scanner, "calculate_value", lambda p, o: _val())

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Cincinnati Open", surface="hard",
            match_date=date(2026, 8, 11), player_a="Henrique Rocha", player_b="Marcos Giron",
            odds_a=2.81, odds_b=1.46, raw_home="Henrique Rocha", raw_away="Marcos Giron",
        )
        results = evaluate_matches([m], models={"atp": (None, None, None, None, None, None)})
        assert results[0].low_sample is True

    def test_not_low_sample_when_both_players_meet_threshold(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "predict_match", lambda *a, **k: _pred(matches_a=30, matches_b=40))
        monkeypatch.setattr(scanner, "calculate_value", lambda p, o: _val())

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Wimbledon", surface="grass",
            match_date=date(2026, 7, 27), player_a="Novak Djokovic", player_b="Jannik Sinner",
            odds_a=1.5, odds_b=2.6, raw_home="Novak Djokovic", raw_away="Jannik Sinner",
        )
        results = evaluate_matches([m], models={"atp": (None, None, None, None, None, None)})
        assert results[0].low_sample is False

    def test_is_value_bet_false_when_low_sample_even_with_value(self, monkeypatch):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "predict_match", lambda *a, **k: _pred(matches_a=14, matches_b=302))
        monkeypatch.setattr(scanner, "calculate_value", lambda p, o: _val(edge=0.05, has_value=True))

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Cincinnati Open", surface="hard",
            match_date=date(2026, 8, 11), player_a="Henrique Rocha", player_b="Marcos Giron",
            odds_a=2.81, odds_b=1.46, raw_home="Henrique Rocha", raw_away="Marcos Giron",
        )
        results = evaluate_matches([m], models={"atp": (None, None, None, None, None, None)})
        assert results[0].is_value_bet is False

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


class TestScannerGracefulFailures:
    """Rule: if The Odds API has no key, or finds nothing, run_scan() must
    fail gracefully with a clear message -- never crash."""

    def test_missing_api_key_fails_gracefully_no_crash(self, monkeypatch, capsys):
        import src.daily_scanner as scanner

        monkeypatch.delenv("ODDS_API_KEY", raising=False)

        scanner.run_scan()  # must not raise

        assert "ODDS_API_KEY" in capsys.readouterr().out

    def test_scanner_handles_no_matches_found(self, monkeypatch, capsys):
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "_load_models", lambda tours, retrain: {})
        monkeypatch.setattr(scanner, "_canonical_names", lambda models: {})
        monkeypatch.setattr(scanner, "discover_matches", lambda *a, **kw: [])
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")

        scanner.run_scan()  # must not raise

        assert "No se encontraron" in capsys.readouterr().out


class TestScannerProducesPicksRegardlessOfDataAge:
    def test_scanner_produces_picks_with_stale_data(self, monkeypatch):
        """daily_scanner.py (unlike scripts/daily_workflow.py) has no
        staleness gate at all -- it logs a value bet using whatever model
        _load_models hands it, regardless of how old that model's
        underlying match data is. Pins this down explicitly since it's easy
        to wrongly assume some staleness check happens somewhere in this
        path (it doesn't -- that's scripts/daily_workflow.py's job)."""
        import src.daily_scanner as scanner

        m = scanner.DiscoveredMatch(
            tour="wta", tournament="WTA Guadalajara Open", surface="hard",
            match_date=date(2026, 9, 18), player_a="Iga Swiatek", player_b="Coco Gauff",
            odds_a=1.8, odds_b=2.0, raw_home="Iga Swiatek", raw_away="Coco Gauff",
        )
        r = EvaluatedMatch(
            match=m, pred=_pred(), val_a=_val(has_value=True), val_b=_val(has_value=False),
            elo_ok=True, suspicious=False,
        )
        monkeypatch.setattr(scanner, "_load_models", lambda tours, retrain: {"wta": (None,) * 6})
        monkeypatch.setattr(scanner, "_canonical_names", lambda models: {"wta": []})
        monkeypatch.setattr(scanner, "discover_matches", lambda *a, **kw: [m])
        monkeypatch.setattr(scanner, "evaluate_matches", lambda matches, models, halt_on_suspicious=True: [r])
        monkeypatch.setattr("builtins.input", lambda *a: "s")
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")

        logged = []
        monkeypatch.setattr(scanner, "log_query", lambda *a, **kw: logged.append(kw))
        monkeypatch.setattr(scanner, "log_prediction_audit", lambda *a, **kw: None)

        scanner.run_scan(tours=("wta",))

        assert len(logged) == 1  # nothing blocked the pick from being logged


class TestScannerPlayerNameResolution:
    def test_resolves_surname_and_initial_variant_to_canonical_name(self, monkeypatch):
        # discover_matches uses src.player_matcher.match_player_name (surname
        # + first-initial heuristic), not exact string equality -- a
        # bookmaker-style "N. Djokovic" must still resolve to the dataset's
        # canonical "Novak Djokovic".
        import src.daily_scanner as scanner

        monkeypatch.setattr(scanner, "fetch_sports_index", lambda api_key: [
            {"key": "tennis_atp_wimbledon", "title": "ATP Wimbledon"},
        ])
        monkeypatch.setattr(
            scanner, "fetch_odds_events_by_key",
            lambda sport_key, api_key: [
                _event("N. Djokovic", "J. Sinner", bookmaker_key=DEFAULT_BOOKMAKER)
            ],
        )
        now = datetime(2026, 7, 27, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(scanner, "_now", lambda: now)

        matches = discover_matches(
            api_key="fake",
            canonical_names_by_tour={"atp": ["Novak Djokovic", "Jannik Sinner"], "wta": []},
            days_ahead=1,
        )

        assert len(matches) == 1
        assert matches[0].player_a == "Novak Djokovic"
        assert matches[0].player_b == "Jannik Sinner"


class TestScannerDavisTourDispatch:
    def test_davis_tour_runs_without_crashing(self, monkeypatch):
        import src.daily_scanner as scanner

        captured = {}

        def _fake_load_models(tours, retrain):
            captured["tours"] = tours
            return {"davis": (None,) * 6}

        monkeypatch.setattr(scanner, "_load_models", _fake_load_models)
        monkeypatch.setattr(scanner, "_canonical_names", lambda models: {"davis": []})
        monkeypatch.setattr(scanner, "discover_matches", lambda *a, **kw: [])
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")

        scanner.run_scan(tours=("davis",))  # must not raise

        assert captured["tours"] == ("davis",)


class TestRunScanDefaultToursIncludesDavis:
    def test_default_tours_includes_davis(self, monkeypatch):
        import src.daily_scanner as scanner

        captured = {}

        def _fake_load_models(tours, retrain):
            captured["tours"] = tours
            return {}

        monkeypatch.setattr(scanner, "_load_models", _fake_load_models)
        monkeypatch.setattr(scanner, "_canonical_names", lambda models: {})
        monkeypatch.setattr(scanner, "discover_matches", lambda *a, **kw: [])
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")

        scanner.run_scan()  # no tours kwarg -- exercises the default

        assert captured["tours"] == ("atp", "wta", "davis")


# ── print_value_bets_table ────────────────────────────────────────────────────

class TestRunScanBookmakerLogging:
    def test_logs_bookmaker_a_and_b_from_match(self, monkeypatch):
        import src.daily_scanner as scanner

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Wimbledon", surface="grass",
            match_date=date(2026, 7, 27), player_a="Novak Djokovic", player_b="Jannik Sinner",
            odds_a=1.5, odds_b=2.6, raw_home="Novak Djokovic", raw_away="Jannik Sinner",
            bookmaker_a="bet365", bookmaker_b="pinnacle",
        )
        r = EvaluatedMatch(
            match=m, pred=_pred(), val_a=_val(has_value=True), val_b=_val(has_value=False),
            elo_ok=True, suspicious=False,
        )
        monkeypatch.setattr(scanner, "_load_models", lambda tours, retrain: {"atp": (None,)*6})
        monkeypatch.setattr(scanner, "_canonical_names", lambda models: {"atp": [], "wta": []})
        monkeypatch.setattr(scanner, "discover_matches", lambda *a, **kw: [m])
        monkeypatch.setattr(scanner, "evaluate_matches", lambda matches, models, halt_on_suspicious=True: [r])
        monkeypatch.setattr("builtins.input", lambda *a: "s")
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")

        logged = []
        monkeypatch.setattr(scanner, "log_query", lambda *a, **kw: logged.append(kw))
        monkeypatch.setattr(scanner, "log_prediction_audit", lambda *a, **kw: None)

        scanner.run_scan(tours=("atp",))

        assert logged[0]["bookmaker_a"] == "bet365"
        assert logged[0]["bookmaker_b"] == "pinnacle"


class TestRunScanDefaultBookmaker:
    def test_default_bookmaker_arg_is_none_not_pinnacle(self):
        import argparse
        import src.daily_scanner as scanner
        parser = argparse.ArgumentParser()
        parser.add_argument("--bookmaker", default=None)
        args = parser.parse_args([])
        assert args.bookmaker is None


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


# ── auto-save ─────────────────────────────────────────────────────────

class TestScannerAutoSave:
    """Cuando auto_save=True, run_scan() guarda value bets sin invocar
    input() — modo no-interactivo para produccion."""

    def test_auto_save_skips_input_prompt(self, monkeypatch):
        import src.daily_scanner as scanner

        m = scanner.DiscoveredMatch(
            tour="wta", tournament="WTA Test", surface="hard",
            match_date=date(2026, 9, 19), player_a="Test Player A",
            player_b="Test Player B", odds_a=1.8, odds_b=2.0,
            raw_home="Test Player A", raw_away="Test Player B",
        )
        r = EvaluatedMatch(
            match=m, pred=_pred(), val_a=_val(has_value=True), val_b=_val(has_value=False),
            elo_ok=True, suspicious=False,
        )
        monkeypatch.setattr(scanner, "_load_models", lambda tours, retrain: {"wta": (None,) * 6})
        monkeypatch.setattr(scanner, "_canonical_names", lambda models: {"wta": []})
        monkeypatch.setattr(scanner, "discover_matches", lambda *a, **kw: [m])
        monkeypatch.setattr(scanner, "evaluate_matches", lambda matches, models, halt_on_suspicious=True: [r])
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")

        logged = []
        monkeypatch.setattr(scanner, "log_query", lambda *a, **kw: logged.append(kw))
        monkeypatch.setattr(scanner, "log_prediction_audit", lambda *a, **kw: None)

        input_called = [False]
        def fake_input(*a):
            input_called[0] = True
            return "s"
        monkeypatch.setattr("builtins.input", fake_input)

        scanner.run_scan(tours=("wta",), auto_save=True)

        assert not input_called[0], "input() should never be called when auto_save=True"
        assert len(logged) == 1, "value bet should be auto-logged"

    def test_auto_save_false_calls_input(self, monkeypatch):
        import src.daily_scanner as scanner

        m = scanner.DiscoveredMatch(
            tour="wta", tournament="WTA Test", surface="hard",
            match_date=date(2026, 9, 19), player_a="Test Player A",
            player_b="Test Player B", odds_a=1.8, odds_b=2.0,
            raw_home="Test Player A", raw_away="Test Player B",
        )
        r = EvaluatedMatch(
            match=m, pred=_pred(), val_a=_val(has_value=True), val_b=_val(has_value=False),
            elo_ok=True, suspicious=False,
        )
        monkeypatch.setattr(scanner, "_load_models", lambda tours, retrain: {"wta": (None,) * 6})
        monkeypatch.setattr(scanner, "_canonical_names", lambda models: {"wta": []})
        monkeypatch.setattr(scanner, "discover_matches", lambda *a, **kw: [m])
        monkeypatch.setattr(scanner, "evaluate_matches", lambda matches, models, halt_on_suspicious=True: [r])
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")

        logged = []
        monkeypatch.setattr(scanner, "log_query", lambda *a, **kw: logged.append(kw))
        monkeypatch.setattr(scanner, "log_prediction_audit", lambda *a, **kw: None)

        input_called = [False]
        def fake_input(*a):
            input_called[0] = True
            return "s"
        monkeypatch.setattr("builtins.input", fake_input)

        scanner.run_scan(tours=("wta",), auto_save=False)

        assert input_called[0], "input() should be called when auto_save=False"
        assert len(logged) == 1

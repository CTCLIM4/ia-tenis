"""Tests for scripts/daily_workflow.py — non-interactive daily scan + filters."""
from __future__ import annotations

from datetime import date

import scripts.daily_workflow as workflow
from src.daily_scanner import DiscoveredMatch, EvaluatedMatch


def _pred():
    return {"p_a_raw": 0.6, "p_a_cal": 0.6, "p_b_raw": 0.4, "p_b_cal": 0.4, "features": {}}


def _val(edge=0.0, kelly=0.0, has_value=False):
    return {"implied_prob": 0.5, "edge": edge, "ev": edge, "kelly_fraction": kelly, "has_value": has_value}


def _match(player_a="A Player", player_b="B Player", odds_a=2.0, odds_b=1.8, tour="atp"):
    return DiscoveredMatch(
        tour=tour, tournament="ATP Test Open", surface="hard",
        match_date=date(2026, 8, 19), player_a=player_a, player_b=player_b,
        odds_a=odds_a, odds_b=odds_b, raw_home=player_a, raw_away=player_b,
    )


def _result(match=None, val_a=None, val_b=None, elo_ok=True, suspicious=False, low_sample=False):
    return EvaluatedMatch(
        match=match or _match(), pred=_pred(),
        val_a=val_a or _val(), val_b=val_b or _val(),
        elo_ok=elo_ok, suspicious=suspicious, low_sample=low_sample,
    )


class TestQualifyingSides:
    def test_edge_above_min_and_has_value_qualifies(self):
        r = _result(val_a=_val(edge=0.05, kelly=0.02, has_value=True))
        assert workflow._qualifying_sides(r) == (True, False)

    def test_edge_below_min_edge_does_not_qualify_even_with_value(self):
        r = _result(val_a=_val(edge=0.01, kelly=0.005, has_value=True))
        assert workflow._qualifying_sides(r) == (False, False)

    def test_low_sample_blocks_qualification(self):
        r = _result(val_a=_val(edge=0.05, kelly=0.02, has_value=True), low_sample=True)
        assert workflow._qualifying_sides(r) == (False, False)

    def test_suspicious_blocks_qualification(self):
        r = _result(val_a=_val(edge=0.05, kelly=0.02, has_value=True), suspicious=True)
        assert workflow._qualifying_sides(r) == (False, False)

    def test_missing_elo_blocks_qualification(self):
        r = _result(val_a=_val(edge=0.05, kelly=0.02, has_value=True), elo_ok=False)
        assert workflow._qualifying_sides(r) == (False, False)

    def test_both_sides_can_qualify_independently(self):
        r = _result(
            val_a=_val(edge=0.04, kelly=0.01, has_value=True),
            val_b=_val(edge=0.05, kelly=0.02, has_value=True),
        )
        assert workflow._qualifying_sides(r) == (True, True)


class TestWtaIsStale:
    def test_false_when_no_cache(self, monkeypatch):
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: None)
        assert workflow._wta_is_stale() is False

    def test_true_when_stale(self, monkeypatch):
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: date(2026, 1, 1))
        assert workflow._wta_is_stale() is True

    def test_false_when_fresh(self, monkeypatch):
        import src.data.timezone_utils as tzu
        today = tzu.lima_today()
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: today)
        assert workflow._wta_is_stale() is False


class TestAtpIsStale:
    def test_false_when_no_cache(self, monkeypatch):
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: None)
        assert workflow._atp_is_stale() is False

    def test_true_when_stale(self, monkeypatch):
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: date(2026, 1, 1))
        assert workflow._atp_is_stale() is True

    def test_false_when_fresh(self, monkeypatch):
        import src.data.timezone_utils as tzu
        today = tzu.lima_today()
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: today)
        assert workflow._atp_is_stale() is False


class TestStalenessLiveTournamentMode:
    """live_tournament_mode=True matches src/daily_scanner.py's behavior for
    a confirmed-live tournament: staleness is still computed and printed as
    a warning, but never excludes the tour. Default (live_tournament_mode
    omitted/False) preserves the original hard-exclude-on-any-non-OK gate,
    which exists specifically to stop auto-pushed, auto-emailed picks from
    being generated off data that could be weeks out of date with nobody
    reviewing it (see docs/superpowers/specs/2026-07-13-staleness-context-aware-design.md)."""

    def test_daily_workflow_allows_current_day_match_in_live_mode(self, monkeypatch):
        import src.data.timezone_utils as tzu
        today = tzu.lima_today()
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: today)

        assert workflow._atp_is_stale(live_tournament_mode=True) is False
        assert workflow._wta_is_stale(live_tournament_mode=True) is False

    def test_daily_workflow_still_warns_on_stale_data(self, monkeypatch, capsys):
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: date(2026, 1, 1))

        result = workflow._wta_is_stale(live_tournament_mode=True)

        assert result is False  # warns, does not block
        out = capsys.readouterr().out
        assert "dias atras" in out or "días atrás" in out

    def test_default_mode_still_blocks_stale_data(self, monkeypatch):
        # Regression guard: the live_tournament_mode=True relaxation must
        # not leak into the default (opt-out) case.
        monkeypatch.setattr(workflow, "get_last_match_date", lambda tour: date(2026, 1, 1))
        assert workflow._wta_is_stale() is True
        assert workflow._wta_is_stale(live_tournament_mode=False) is True

    def test_daily_workflow_davis_cup_staleness_ok(self, monkeypatch):
        # Davis Cup's own cached last_match_date can be genuinely old
        # (occasional fixture windows, not a weekly tour like ATP/WTA) --
        # _davis_is_stale must work off the same generic mechanism (it reads
        # data/model_cache/davis.pkl via get_last_match_date, no separate
        # data source) and must not block under live_tournament_mode either.
        monkeypatch.setattr(
            workflow, "get_last_match_date",
            lambda tour: date(2026, 2, 6) if tour == "davis" else None,
        )
        assert workflow._davis_is_stale(live_tournament_mode=True) is False


class TestRun:
    def _wire_common(self, monkeypatch, results, stale_wta=False, stale_atp=False, api_key="key123"):
        monkeypatch.setenv("ODDS_API_KEY", api_key)
        monkeypatch.setattr(workflow, "_load_models", lambda tours, retrain: {})
        monkeypatch.setattr(workflow, "_canonical_names", lambda models: {})
        # Mirrors the real _wta_is_stale/_atp_is_stale contract: stale only
        # excludes when live_tournament_mode is not set.
        monkeypatch.setattr(
            workflow, "_wta_is_stale",
            lambda live_tournament_mode=False: stale_wta and not live_tournament_mode,
        )
        monkeypatch.setattr(
            workflow, "_atp_is_stale",
            lambda live_tournament_mode=False: stale_atp and not live_tournament_mode,
        )
        monkeypatch.setattr(workflow, "discover_matches", lambda *a, **kw: [r.match for r in results])
        monkeypatch.setattr(workflow, "evaluate_matches", lambda matches, models: results)
        monkeypatch.setattr(workflow, "track_vpn_usage", lambda: {"available": False})
        self.sent = {}
        monkeypatch.setattr(
            workflow, "send_picks_email",
            lambda picks, vpn: self.sent.update(picks=picks, vpn=vpn) or True,
        )
        self.pushed = {}
        monkeypatch.setattr(
            workflow, "commit_and_push",
            lambda paths, message: self.pushed.update(paths=paths, message=message) or True,
        )
        self.logged_queries = []
        monkeypatch.setattr(
            workflow, "log_query",
            lambda *a, **kw: self.logged_queries.append((a, kw)),
        )
        self.audit_decisions = []
        monkeypatch.setattr(
            workflow, "log_prediction_audit",
            lambda *a, decision, **kw: self.audit_decisions.append(decision),
        )

    def test_aborts_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        assert workflow.run() == []

    def test_selects_only_matches_clearing_min_edge(self, monkeypatch):
        strong = _result(
            match=_match(player_a="Strong Pick"), val_a=_val(edge=0.05, kelly=0.04, has_value=True),
        )
        weak = _result(
            match=_match(player_a="Weak Pick"), val_a=_val(edge=0.01, kelly=0.005, has_value=True),
        )
        self._wire_common(monkeypatch, [strong, weak])

        selected = workflow.run()

        assert len(selected) == 1
        assert selected[0]["pick"] == "Strong Pick"

    def test_applies_quarter_kelly_to_qualifying_side(self, monkeypatch):
        r = _result(val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [r])

        selected = workflow.run()

        assert selected[0]["kelly"] == 0.01

    def test_skips_wta_matches_when_stale(self, monkeypatch):
        atp_r = _result(match=_match(player_a="ATP Pick", tour="atp"),
                         val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [atp_r], stale_wta=True)

        called_tours = {}
        original_discover = workflow.discover_matches
        monkeypatch.setattr(
            workflow, "discover_matches",
            lambda api_key, canonical, tours=("atp", "wta"): called_tours.setdefault("tours", tours) or [atp_r.match],
        )

        workflow.run()

        assert called_tours["tours"] == ("atp",)

    def test_skips_atp_matches_when_stale(self, monkeypatch):
        wta_r = _result(match=_match(player_a="WTA Pick", tour="wta"),
                         val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [wta_r], stale_atp=True)

        called_tours = {}
        monkeypatch.setattr(
            workflow, "discover_matches",
            lambda api_key, canonical, tours=("atp", "wta"): called_tours.setdefault("tours", tours) or [wta_r.match],
        )

        workflow.run()

        assert called_tours["tours"] == ("wta",)

    def test_logs_query_only_for_qualifying_matches(self, monkeypatch):
        strong = _result(val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        weak = _result(val_a=_val(edge=0.01, kelly=0.005, has_value=True))
        self._wire_common(monkeypatch, [strong, weak])

        workflow.run()

        assert len(self.logged_queries) == 1

    def test_audit_log_marks_logged_vs_passed_user_declined(self, monkeypatch):
        strong = _result(val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        weak = _result(val_a=_val(edge=0.01, kelly=0.005, has_value=True))
        self._wire_common(monkeypatch, [strong, weak])

        workflow.run()

        assert self.audit_decisions == ["logged", "passed_user_declined"]

    def test_dry_run_does_not_log_notify_or_push(self, monkeypatch):
        strong = _result(val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [strong])

        selected = workflow.run(dry_run=True)

        assert len(selected) == 1
        assert self.logged_queries == []
        assert self.audit_decisions == []
        assert self.sent == {}
        assert self.pushed == {}

    def test_pushes_only_when_something_was_selected(self, monkeypatch):
        weak = _result(val_a=_val(edge=0.01, kelly=0.005, has_value=True))
        self._wire_common(monkeypatch, [weak])

        workflow.run()

        assert self.pushed == {}

    def test_push_message_mentions_pick_count(self, monkeypatch):
        strong = _result(val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [strong])

        workflow.run()

        assert "1 value bet" in self.pushed["message"]
        assert self.pushed["paths"] == [workflow.VALUE_BETS_LOG]

    def test_sends_email_with_selected_picks_and_vpn_status(self, monkeypatch):
        strong = _result(val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [strong])

        workflow.run()

        assert len(self.sent["picks"]) == 1
        assert self.sent["vpn"] == {"available": False}

    def test_selected_picks_include_bookmaker(self, monkeypatch):
        m = DiscoveredMatch(
            tour="atp", tournament="ATP Test Open", surface="hard",
            match_date=date(2026, 8, 19), player_a="A Player", player_b="B Player",
            odds_a=2.0, odds_b=1.8, raw_home="A Player", raw_away="B Player",
            bookmaker_a="bet365", bookmaker_b="pinnacle",
        )
        r = _result(match=m, val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [r])

        selected = workflow.run()

        assert selected[0]["bookmaker"] == "bet365"

    def test_live_tournament_mode_does_not_exclude_stale_tours(self, monkeypatch):
        atp_r = _result(match=_match(player_a="ATP Pick", tour="atp"),
                         val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        wta_r = _result(match=_match(player_a="WTA Pick", tour="wta"),
                         val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [atp_r, wta_r], stale_wta=True, stale_atp=True)

        called_tours = {}
        monkeypatch.setattr(
            workflow, "discover_matches",
            lambda api_key, canonical, tours=("atp", "wta"):
                called_tours.setdefault("tours", tours) or [atp_r.match, wta_r.match],
        )

        workflow.run(live_tournament_mode=True)

        assert called_tours["tours"] == ("atp", "wta")

    def test_default_mode_still_excludes_stale_tours(self, monkeypatch):
        # Regression guard: live_tournament_mode's relaxation must be
        # strictly opt-in, matching the original hard-gate behavior when
        # the caller doesn't ask for it.
        atp_r = _result(match=_match(player_a="ATP Pick", tour="atp"),
                         val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [atp_r], stale_wta=True, stale_atp=False)

        called_tours = {}
        monkeypatch.setattr(
            workflow, "discover_matches",
            lambda api_key, canonical, tours=("atp", "wta"):
                called_tours.setdefault("tours", tours) or [atp_r.match],
        )

        workflow.run()

        assert called_tours["tours"] == ("atp",)

    def test_logs_query_with_bookmaker_from_match(self, monkeypatch):
        m = DiscoveredMatch(
            tour="atp", tournament="ATP Test Open", surface="hard",
            match_date=date(2026, 8, 19), player_a="A Player", player_b="B Player",
            odds_a=2.0, odds_b=1.8, raw_home="A Player", raw_away="B Player",
            bookmaker_a="bet365", bookmaker_b="pinnacle",
        )
        r = _result(match=m, val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [r])

        workflow.run()

        _, kwargs = self.logged_queries[0]
        assert kwargs["bookmaker_a"] == "bet365"
        assert kwargs["bookmaker_b"] == "pinnacle"

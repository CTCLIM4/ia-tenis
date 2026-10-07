"""Tests for scripts/daily_workflow.py — non-interactive daily scan + filters."""
from __future__ import annotations

import csv
from datetime import date

import scripts.daily_workflow as workflow
from src.daily_scanner import DiscoveredMatch, EvaluatedMatch
from src.features import FEATURE_COLS
from src.value_analysis import LogMatchStatus

import pytest

# Captured before the autouse fixture below swaps it out.
_PRODUCTION_EXCLUDED_TOURS = workflow.AUTO_LOG_EXCLUDED_TOURS


@pytest.fixture(autouse=True)
def _no_excluded_tours(monkeypatch):
    # Most tests exercise the logging/email/dedup mechanics with ATP picks;
    # keep them independent of which tours production currently excludes.
    # Exclusion itself is tested explicitly by overriding this again.
    monkeypatch.setattr(workflow, "AUTO_LOG_EXCLUDED_TOURS", frozenset())


_REAL_CLV_TRACKING = workflow._clv_tracking
_TRACKING_STUB = {"stub": True}


@pytest.fixture(autouse=True)
def _no_real_clv_tracking(monkeypatch):
    # The real one reads data/odds_snapshots/ and the production audit log.
    monkeypatch.setattr(workflow, "_clv_tracking", lambda: _TRACKING_STUB)


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
        monkeypatch.setattr(workflow, "_davis_is_stale", lambda live_tournament_mode=False: False)
        monkeypatch.setattr(workflow, "discover_matches", lambda *a, **kw: [r.match for r in results])
        monkeypatch.setattr(workflow, "evaluate_matches", lambda matches, models: results)
        monkeypatch.setattr(workflow, "track_vpn_usage", lambda: {"available": False})
        monkeypatch.setattr(workflow, "backup_logs", lambda **kw: None)
        monkeypatch.setattr(workflow, "check_existing_log_entry", lambda *a: (LogMatchStatus.NEW, None))
        self.sent = {}
        monkeypatch.setattr(
            workflow, "send_picks_email",
            lambda picks, vpn, signals=None, tracking=None: self.sent.update(
                picks=picks, vpn=vpn, signals=signals, tracking=tracking) or True,
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

    def test_excluded_tour_pick_is_audited_but_never_logged_or_emailed(self, monkeypatch):
        # An excluded tour (docs/metrics/2026-10-07-model-vs-market-edge.md) is
        # still scanned and audited for calibration, but no bet row, no email pick.
        wta_r = _result(match=_match(player_a="WTA Pick", tour="wta"),
                        val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        atp_r = _result(match=_match(player_a="ATP Pick", tour="atp"),
                        val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [wta_r, atp_r])
        monkeypatch.setattr(workflow, "AUTO_LOG_EXCLUDED_TOURS", frozenset({"wta"}))

        selected = workflow.run()

        assert [s["pick"] for s in selected] == ["ATP Pick"]
        assert [a[4] for a, _kw in self.logged_queries] == ["ATP Pick"]  # player_a
        assert [p["pick"] for p in self.sent["picks"]] == ["ATP Pick"]
        assert [p["pick"] for p in self.sent["signals"]] == ["WTA Pick"]  # tracked, not bet
        assert len(self.audit_decisions) == 2
        assert wta_r.val_a["kelly_fraction"] == 0.04  # untouched: never staked

    def test_production_excludes_every_tour(self):
        # No tour showed edge vs the market (2026-10-07); Davis has no odds to
        # test and the ATP model predicts its rubbers better than its own model.
        assert _PRODUCTION_EXCLUDED_TOURS == frozenset({"wta", "atp", "davis"})

    def test_production_exclusion_logs_nothing_for_any_tour(self, monkeypatch):
        picks = [
            _result(match=_match(player_a=f"{tour.upper()} Pick", tour=tour),
                    val_a=_val(edge=0.05, kelly=0.04, has_value=True))
            for tour in ("wta", "atp", "davis")
        ]
        self._wire_common(monkeypatch, picks)
        monkeypatch.setattr(workflow, "AUTO_LOG_EXCLUDED_TOURS", _PRODUCTION_EXCLUDED_TOURS)

        assert workflow.run() == []
        assert self.logged_queries == []
        assert self.sent["picks"] == []
        assert self.pushed == {}
        assert len(self.audit_decisions) == 3

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
            lambda api_key, canonical, tours=("atp", "wta"), days_ahead=1: called_tours.setdefault("tours", tours) or [atp_r.match],
        )

        workflow.run()

        assert called_tours["tours"] == ("atp", "davis")

    def test_skips_atp_matches_when_stale(self, monkeypatch):
        wta_r = _result(match=_match(player_a="WTA Pick", tour="wta"),
                         val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [wta_r], stale_atp=True)

        called_tours = {}
        monkeypatch.setattr(
            workflow, "discover_matches",
            lambda api_key, canonical, tours=("atp", "wta"), days_ahead=1: called_tours.setdefault("tours", tours) or [wta_r.match],
        )

        workflow.run()

        assert called_tours["tours"] == ("wta", "davis")

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

    def test_no_push_still_logs_and_notifies(self, monkeypatch):
        strong = _result(val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [strong])

        selected = workflow.run(push=False)

        assert len(selected) == 1
        assert selected[0]["kelly"] == 0.01
        assert len(self.logged_queries) == 1
        assert len(self.sent["picks"]) == 1
        assert self.pushed == {}

    def test_selected_tour_and_days_reach_discovery(self, monkeypatch):
        self._wire_common(monkeypatch, [])
        seen = {}

        def load_models(tours, retrain):
            seen["models"] = tours
            return {}

        monkeypatch.setattr(workflow, "_load_models", load_models)
        monkeypatch.setattr(workflow, "discover_matches", lambda *a, **kw: seen.update(kw) or [])

        workflow.run(tour="wta", days_ahead=2, dry_run=True)

        assert seen["models"] == ("wta",)
        assert seen["tours"] == ("wta",)
        assert seen["days_ahead"] == 2

    def test_stale_davis_tour_does_not_discover_matches(self, monkeypatch):
        self._wire_common(monkeypatch, [])
        monkeypatch.setattr(workflow, "_davis_is_stale", lambda live_tournament_mode=False: True)
        monkeypatch.setattr(workflow, "discover_matches", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("stale tour scanned")))

        assert workflow.run(tour="davis", dry_run=True) == []

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
            lambda api_key, canonical, tours=("atp", "wta"), days_ahead=1:
                called_tours.setdefault("tours", tours) or [atp_r.match, wta_r.match],
        )

        workflow.run(live_tournament_mode=True)

        assert called_tours["tours"] == ("atp", "wta", "davis")

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
            lambda api_key, canonical, tours=("atp", "wta"), days_ahead=1:
                called_tours.setdefault("tours", tours) or [atp_r.match],
        )

        workflow.run()

        assert called_tours["tours"] == ("atp", "davis")

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


class TestRunLogIntegrity:
    def _wire(self, monkeypatch, tmp_path, result):
        log_path = tmp_path / "value_bets_log.csv"
        monkeypatch.setattr("src.value_analysis.LOG_PATH", log_path)
        monkeypatch.setenv("ODDS_API_KEY", "test-key")
        monkeypatch.setattr(workflow, "_load_models", lambda tours, retrain: {})
        monkeypatch.setattr(workflow, "_canonical_names", lambda models: {})
        monkeypatch.setattr(workflow, "_atp_is_stale", lambda live_tournament_mode=False: False)
        monkeypatch.setattr(workflow, "_wta_is_stale", lambda live_tournament_mode=False: False)
        monkeypatch.setattr(workflow, "_davis_is_stale", lambda live_tournament_mode=False: False)
        monkeypatch.setattr(workflow, "discover_matches", lambda *a, **kw: [result.match])
        monkeypatch.setattr(workflow, "evaluate_matches", lambda matches, models: [result])
        monkeypatch.setattr(workflow, "track_vpn_usage", lambda: {})
        monkeypatch.setattr(workflow, "send_picks_email", lambda *a: True)
        monkeypatch.setattr(workflow, "commit_and_push", lambda *a: True)
        monkeypatch.setattr(workflow, "log_prediction_audit", lambda *a, **kw: None)
        backups = []
        monkeypatch.setattr(workflow, "backup_logs", lambda **kw: backups.append(log_path.exists()))
        return log_path, backups

    @staticmethod
    def _result(odds_a=3.80):
        result = _result(
            match=_match(odds_a=odds_a),
            val_a=_val(edge=0.05, kelly=0.04, has_value=True),
        )
        result.pred.update({
            "features": {col: 0.0 for col in FEATURE_COLS},
            "rank_a": None, "rank_b": None,
            "rank_a_source": "", "rank_b_source": "",
        })
        return result

    @staticmethod
    def _rows(path):
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_second_run_skips_pending_duplicate(self, monkeypatch, tmp_path):
        result = self._result()
        log_path, backups = self._wire(monkeypatch, tmp_path, result)
        pushes = []
        monkeypatch.setattr(workflow, "commit_and_push", lambda *a: pushes.append(a))

        assert len(workflow.run()) == 1
        assert workflow.run() == []

        assert len(self._rows(log_path)) == 1
        assert backups == [False, True]
        assert len(pushes) == 1

    def test_changed_odds_update_existing_row(self, monkeypatch, tmp_path):
        first = self._result()
        log_path, _ = self._wire(monkeypatch, tmp_path, first)
        workflow.run()

        updated = self._result(odds_a=3.90)
        monkeypatch.setattr(workflow, "evaluate_matches", lambda matches, models: [updated])
        assert len(workflow.run()) == 1

        rows = self._rows(log_path)
        assert len(rows) == 1
        assert rows[0]["odds_a"] == "3.9"
        assert rows[0]["kelly_a"] == "0.01"

    def test_resolved_match_is_untouched(self, monkeypatch, tmp_path):
        result = self._result()
        log_path, _ = self._wire(monkeypatch, tmp_path, result)
        workflow.run()
        rows = self._rows(log_path)
        rows[0]["result"] = "A_win"
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        assert workflow.run() == []
        assert self._rows(log_path)[0]["result"] == "A_win"
        assert len(self._rows(log_path)) == 1

    def test_backup_precedes_audit_write_even_without_picks(self, monkeypatch, tmp_path):
        result = _result()
        self._wire(monkeypatch, tmp_path, result)
        calls = []
        monkeypatch.setattr(workflow, "backup_logs", lambda **kw: calls.append("backup"))
        monkeypatch.setattr(workflow, "log_prediction_audit", lambda *a, **kw: calls.append("audit"))

        workflow.run()

        assert calls == ["backup", "audit"]

    def test_backup_copies_real_log_to_project_directory(self, monkeypatch, tmp_path):
        from scripts.backup_logs import backup_logs

        result = _result()
        log_path, _ = self._wire(monkeypatch, tmp_path, result)
        log_path.write_text("previous value log\n", encoding="utf-8")
        audit_path = tmp_path / "prediction_audit_log.csv"
        audit_path.write_text("previous audit log\n", encoding="utf-8")
        monkeypatch.setattr(workflow, "AUDIT_LOG_PATH", audit_path)
        monkeypatch.setattr(workflow, "_ROOT", tmp_path)
        monkeypatch.setattr(workflow, "backup_logs", backup_logs)

        workflow.run()

        backups = list((tmp_path / "data" / "logs").glob("backup_*"))
        assert len(backups) == 2
        assert {path.read_text(encoding="utf-8") for path in backups} == {
            "previous value log\n", "previous audit log\n",
        }


class TestClvTracking:
    def test_email_gets_tracking_status(self, monkeypatch):
        t = TestRun()
        t._wire_common(monkeypatch, [])
        workflow.run()
        assert t.sent["tracking"] is _TRACKING_STUB
        assert t.sent["signals"] == []

    def test_reporting_failure_never_breaks_the_run(self, monkeypatch, capsys):
        import scripts.clv_report as clv_report

        def boom(**kw):
            raise ValueError("corrupt snapshot")
        monkeypatch.setattr(clv_report, "summary", boom)
        assert _REAL_CLV_TRACKING() is None
        assert "corrupt snapshot" in capsys.readouterr().out

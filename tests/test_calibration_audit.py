"""Tests for src/calibration_audit.py — the prediction audit log that
captures every evaluated prediction (not just ones saved to
value_bets_log.csv), so calibration can eventually be audited against the
full, unbiased population of predictions."""
from __future__ import annotations

import csv
import itertools
from datetime import date

import pytest

import src.calibration_audit as calibration_audit
from src.value_analysis import calculate_value


class TestClassifyAuditDecision:
    def test_missing_elo_wins_over_low_sample_and_suspicious_edge(self):
        # A player never seen at all (0 matches) always also satisfies
        # low_sample, but invalid_missing_elo is the more specific, more
        # useful diagnosis and must win — checked in this order deliberately
        # (even the contradictory-in-practice combo of elo_ok=False with
        # logged=True must still resolve this way, defensively).
        result = calibration_audit.classify_audit_decision(
            low_sample=True, suspicious_edge=True, elo_ok=False, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "invalid_missing_elo"

    def test_missing_elo_wins_over_logged_and_value(self):
        result = calibration_audit.classify_audit_decision(
            low_sample=False, suspicious_edge=False, elo_ok=False, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "invalid_missing_elo"

    def test_low_sample_wins_over_suspicious_edge_when_elo_ok(self):
        # A thin-but-nonzero sample is the root cause a huge edge is not
        # trustworthy — reported as blocked_low_sample rather than the less
        # specific blocked_suspicious_edge.
        result = calibration_audit.classify_audit_decision(
            low_sample=True, suspicious_edge=True, elo_ok=True, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "blocked_low_sample"

    def test_low_sample_wins_over_logged_and_value(self):
        result = calibration_audit.classify_audit_decision(
            low_sample=True, suspicious_edge=False, elo_ok=True, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "blocked_low_sample"

    def test_suspicious_edge_when_sample_is_sufficient(self):
        result = calibration_audit.classify_audit_decision(
            low_sample=False, suspicious_edge=True, elo_ok=True, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "blocked_suspicious_edge"

    def test_logged_when_user_confirmed_save(self):
        result = calibration_audit.classify_audit_decision(
            low_sample=False, suspicious_edge=False, elo_ok=True, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "logged"

    def test_passed_low_edge_when_neither_side_has_value_and_not_logged(self):
        result = calibration_audit.classify_audit_decision(
            low_sample=False, suspicious_edge=False, elo_ok=True, logged=False,
            has_value_a=False, has_value_b=False,
        )
        assert result == "passed_low_edge"

    def test_passed_user_declined_when_value_existed_but_not_logged(self):
        result = calibration_audit.classify_audit_decision(
            low_sample=False, suspicious_edge=False, elo_ok=True, logged=False,
            has_value_a=True, has_value_b=False,
        )
        assert result == "passed_user_declined"

    def test_passed_user_declined_when_only_side_b_has_value(self):
        result = calibration_audit.classify_audit_decision(
            low_sample=False, suspicious_edge=False, elo_ok=True, logged=False,
            has_value_a=False, has_value_b=True,
        )
        assert result == "passed_user_declined"

    def test_audit_log_no_unknown_recovery_in_code(self):
        """Regression guard: 7 rows in the real prediction_audit_log.csv
        (timestamps 2026-08-21 through 2026-08-27) carry
        decision == "UNKNOWN_RECOVERY" -- a string that appears nowhere in
        this function, nowhere else in src/, and nowhere in git history as
        code (only ever as data, in the commit that first tracked the CSV).
        It was written by some uncommitted, ad-hoc script that called
        log_prediction_audit() directly with a literal decision string,
        bypassing this classifier -- not by a bug in classify_audit_decision
        itself. Exhaustively trying every boolean combination this function
        accepts proves it can never itself produce that value (or anything
        outside its known vocabulary); the real fix against recurrence is
        that no current code path calls log_prediction_audit() with a
        hardcoded decision -- both call sites (src/daily_scanner.py,
        src/value_analysis.py) always classify via this function first."""
        known_decisions = {
            "invalid_missing_elo", "blocked_low_sample", "blocked_suspicious_edge",
            "logged", "passed_low_edge", "passed_user_declined",
        }
        for low_sample, suspicious_edge, elo_ok, logged, has_value_a, has_value_b in (
            itertools.product((True, False), repeat=6)
        ):
            result = calibration_audit.classify_audit_decision(
                low_sample=low_sample, suspicious_edge=suspicious_edge, elo_ok=elo_ok,
                logged=logged, has_value_a=has_value_a, has_value_b=has_value_b,
            )
            assert result != "UNKNOWN_RECOVERY"
            assert result in known_decisions, (
                f"unexpected decision {result!r} for inputs "
                f"low_sample={low_sample}, suspicious_edge={suspicious_edge}, "
                f"elo_ok={elo_ok}, logged={logged}, "
                f"has_value_a={has_value_a}, has_value_b={has_value_b}"
            )


def _make_pred(p_a_raw=0.60, p_a_cal=0.58, elo_found_a=True, elo_found_b=True):
    return {
        "player_a":    "Player A",
        "player_b":    "Player B",
        "p_a_raw":     p_a_raw,
        "p_a_cal":     p_a_cal,
        "p_b_raw":     1.0 - p_a_raw,
        "p_b_cal":     1.0 - p_a_cal,
        "elo_found_a": elo_found_a,
        "elo_found_b": elo_found_b,
    }


def _read_audit_log(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture()
def audit_log_path(tmp_path, monkeypatch):
    path = tmp_path / "prediction_audit_log.csv"
    monkeypatch.setattr(calibration_audit, "AUDIT_LOG_PATH", path)
    return path


class TestLogPredictionAudit:
    def test_writes_one_row_with_expected_fields(self, audit_log_path, monkeypatch):
        # NOTE the patch target: src/calibration_audit.py does
        # `from src import git_utils` (module import, not
        # `from src.git_utils import current_git_commit`) and calls
        # `git_utils.current_git_commit()` — so patching the attribute on
        # the shared src.git_utils module object is what log_prediction_audit
        # actually sees. Patching a name on src.calibration_audit itself
        # would silently no-op and this test would instead exercise the
        # real subprocess call.
        monkeypatch.setattr("src.git_utils.current_git_commit", lambda: "abc1234")
        pred = _make_pred()
        val_a = calculate_value(0.58, 1.90)
        val_b = calculate_value(0.42, 2.10)

        calibration_audit.log_prediction_audit(
            "atp", "Test Open", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, val_a, val_b, 1.90, 2.10,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )

        rows = _read_audit_log(audit_log_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["tour"] == "atp"
        assert row["player_a"] == "Player A"
        assert row["decision"] == "logged"
        assert row["p_a_raw"] == "0.6"
        assert row["shrink_hi"] == "0.9"
        assert row["model_commit"] == "abc1234"
        assert row["model_snapshot_id"] == ""

    def test_model_snapshot_id_recorded_when_given(self, audit_log_path):
        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="logged",
            model_snapshot_id="2026-07-25",
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["model_snapshot_id"] == "2026-07-25"

    def test_appends_multiple_rows(self, audit_log_path):
        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        for decision in ("logged", "passed_low_edge", "blocked_suspicious_edge"):
            calibration_audit.log_prediction_audit(
                "atp", "Test", "hard", date(2026, 7, 25),
                "Player A", "Player B",
                pred, v, v, 1.90, 1.90,
                "manual", "manual",
                decision=decision,
                model_snapshot_id=None,
                shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
            )
        rows = _read_audit_log(audit_log_path)
        assert [r["decision"] for r in rows] == ["logged", "passed_low_edge", "blocked_suspicious_edge"]

    def test_records_elo_found_flags(self, audit_log_path):
        pred = _make_pred(elo_found_a=True, elo_found_b=False)
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "wta", "Test", "clay", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="invalid_missing_elo",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["elo_found_a"] == "True"
        assert row["elo_found_b"] == "False"

    def test_shrinkage_applied_true_when_cal_differs_from_raw(self, audit_log_path):
        pred = _make_pred(p_a_raw=0.95, p_a_cal=0.92)  # shrinkage compresses this
        v = calculate_value(0.92, 1.50)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.50, 1.50,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["shrinkage_applied"] == "True"

    def test_bookmaker_a_and_b_recorded(self, audit_log_path):
        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
            bookmaker_a="bet365", bookmaker_b="pinnacle",
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["bookmaker_a"] == "bet365"
        assert row["bookmaker_b"] == "pinnacle"

    def test_bookmaker_defaults_to_empty_string(self, audit_log_path):
        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["bookmaker_a"] == ""
        assert row["bookmaker_b"] == ""

    def test_migrates_old_header_and_preserves_old_row(self, audit_log_path):
        """Regression: a pre-existing audit log written under the older
        32-column header (no bookmaker_a/bookmaker_b) must be migrated in
        place the next time log_prediction_audit() appends a row, instead of
        silently misaligning every row appended after it (the bug that broke
        src.backtest_analytics' load_audit_log with a pandas ParserError)."""
        old_fields = [f for f in calibration_audit._AUDIT_LOG_FIELDS
                      if f not in ("bookmaker_a", "bookmaker_b")]
        old_row = {
            "timestamp": "2026-08-21T01:14:22",
            "tour": "atp",
            "tournament": "ATP Cincinnati Open TEST",
            "surface": "hard",
            "match_date": "2026-08-21",
            "player_a": "Lorenzo Musetti",
            "player_b": "Frances Tiafoe",
            "p_a_raw": "0.5738",
            "p_a_cal": "0.5738",
            "p_b_raw": "0.4262",
            "p_b_cal": "0.4262",
            "shrink_hi": "0.9",
            "shrink_lo": "0.1",
            "shrink_rate": "0.6",
            "shrinkage_applied": "False",
            "odds_a": "1.89",
            "odds_a_source": "auto",
            "odds_b": "2.01",
            "odds_b_source": "auto",
            "implied_a": "0.5291",
            "implied_b": "0.4975",
            "edge_a": "0.0447",
            "ev_a": "0.0845",
            "kelly_a": "0.025",
            "edge_b": "-0.0713",
            "ev_b": "-0.1434",
            "kelly_b": "0.0",
            "model_commit": "b973c0f",
            "model_snapshot_id": "",
            "elo_found_a": "True",
            "elo_found_b": "True",
            "decision": "logged",
        }
        with open(audit_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow(old_row)

        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 8, 22),
            "Player C", "Player D",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
            bookmaker_a="betus", bookmaker_b="betfair_ex_uk",
        )

        rows = _read_audit_log(audit_log_path)
        assert len(rows) == 2

        migrated = rows[0]
        assert migrated["player_a"] == "Lorenzo Musetti"
        assert migrated["odds_a"] == "1.89"
        assert migrated["odds_b"] == "2.01"
        assert migrated["bookmaker_a"] == ""
        assert migrated["bookmaker_b"] == ""

        new_row = rows[1]
        assert new_row["player_a"] == "Player C"
        assert new_row["bookmaker_a"] == "betus"
        assert new_row["bookmaker_b"] == "betfair_ex_uk"

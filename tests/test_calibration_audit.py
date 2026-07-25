"""Tests for src/calibration_audit.py — the prediction audit log that
captures every evaluated prediction (not just ones saved to
value_bets_log.csv), so calibration can eventually be audited against the
full, unbiased population of predictions."""
from __future__ import annotations

import csv
from datetime import date

import pytest

import src.calibration_audit as calibration_audit
from src.value_analysis import calculate_value


class TestClassifyAuditDecision:
    def test_suspicious_wins_over_everything(self):
        # Even if elo_ok is False and logged is True (contradictory in
        # practice, but the priority order must still hold defensively).
        result = calibration_audit.classify_audit_decision(
            suspicious=True, elo_ok=False, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "blocked_suspicious"

    def test_missing_elo_wins_over_logged_and_value(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=False, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "invalid_missing_elo"

    def test_logged_when_user_confirmed_save(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=True, logged=True,
            has_value_a=True, has_value_b=False,
        )
        assert result == "logged"

    def test_passed_low_edge_when_neither_side_has_value_and_not_logged(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=True, logged=False,
            has_value_a=False, has_value_b=False,
        )
        assert result == "passed_low_edge"

    def test_passed_user_declined_when_value_existed_but_not_logged(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=True, logged=False,
            has_value_a=True, has_value_b=False,
        )
        assert result == "passed_user_declined"

    def test_passed_user_declined_when_only_side_b_has_value(self):
        result = calibration_audit.classify_audit_decision(
            suspicious=False, elo_ok=True, logged=False,
            has_value_a=False, has_value_b=True,
        )
        assert result == "passed_user_declined"


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
        for decision in ("logged", "passed_low_edge", "blocked_suspicious"):
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
        assert [r["decision"] for r in rows] == ["logged", "passed_low_edge", "blocked_suspicious"]

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

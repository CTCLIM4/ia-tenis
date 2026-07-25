"""Tests for src/calibration_audit.py — the prediction audit log that
captures every evaluated prediction (not just ones saved to
value_bets_log.csv), so calibration can eventually be audited against the
full, unbiased population of predictions."""
from __future__ import annotations

import src.calibration_audit as calibration_audit


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

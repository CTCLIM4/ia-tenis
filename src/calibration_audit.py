"""Calibration/PASS observation audit log: records every prediction the
value-bet CLI evaluates — not just ones a human chose to save to
value_bets_log.csv — so model calibration can eventually be audited
against the full population of predictions instead of the biased subset a
human decided to bet-log. See
docs/superpowers/specs/2026-07-25-calibration-persistence-design.md.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG_PATH = _ROOT / "data" / "prediction_audit_log.csv"

_AUDIT_LOG_FIELDS = [
    "timestamp", "tour", "tournament", "surface", "match_date",
    "player_a", "player_b",
    "p_a_raw", "p_a_cal", "p_b_raw", "p_b_cal",
    "shrink_hi", "shrink_lo", "shrink_rate", "shrinkage_applied",
    "odds_a", "odds_a_source", "odds_b", "odds_b_source",
    "implied_a", "implied_b",
    "edge_a", "ev_a", "kelly_a",
    "edge_b", "ev_b", "kelly_b",
    "model_commit", "model_snapshot_id",
    "elo_found_a", "elo_found_b",
    "decision",  # logged / passed_low_edge / passed_user_declined / blocked_suspicious / invalid_missing_elo
]


def classify_audit_decision(
    suspicious: bool, elo_ok: bool, logged: bool, has_value_a: bool, has_value_b: bool,
) -> str:
    """Classify what happened to one evaluated prediction, for the audit
    log's `decision` column.

    Priority order matters and is deliberately defensive (checked in this
    order even though some combinations shouldn't occur together in
    practice): a suspicious edge always wins — the prediction was never
    trustworthy enough to act on regardless of anything else. Then missing
    Elo — the same data-quality concern log_query() itself encodes via
    status='invalid_missing_elo'. Then whether the user actually logged it
    to value_bets_log.csv. Then whether it even had positive edge on either
    side to log in the first place.
    """
    if suspicious:
        return "blocked_suspicious"
    if not elo_ok:
        return "invalid_missing_elo"
    if logged:
        return "logged"
    if not has_value_a and not has_value_b:
        return "passed_low_edge"
    return "passed_user_declined"

"""Calibration/PASS observation audit log: records every prediction the
value-bet CLI evaluates — not just ones a human chose to save to
value_bets_log.csv — so model calibration can eventually be audited
against the full population of predictions instead of the biased subset a
human decided to bet-log. See
docs/superpowers/specs/2026-07-25-calibration-persistence-design.md.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from src import git_utils

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


def log_prediction_audit(
    tour, tournament, surface, match_date,
    player_a, player_b,
    pred: dict, val_a: dict, val_b: dict, odds_a: float, odds_b: float,
    odds_a_source: str, odds_b_source: str,
    decision: str,
    model_snapshot_id: str | None,
    shrink_hi: float, shrink_lo: float, shrink_rate: float,
) -> None:
    """Append one evaluated prediction to the audit log — unconditionally,
    regardless of whether it was also saved to value_bets_log.csv. This is
    what removes the selection bias that made calibration auditing
    impossible before: every prediction the model made is captured, not
    just the ones a human chose to bet-log.

    shrink_hi/shrink_lo/shrink_rate are passed in by the caller (rather
    than imported from src.value_analysis) to avoid a circular import —
    src.value_analysis imports from this module to call it, so this module
    must not import back from src.value_analysis. git_utils.current_git_commit
    has no such risk and is imported at module top; called as
    git_utils.current_git_commit() (module-qualified, not
    `from ... import current_git_commit`) so tests can monkeypatch it at
    its source (`src.git_utils.current_git_commit`) and have this function
    see the patched version.
    """
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    shrink = abs(pred["p_a_cal"] - pred["p_a_raw"]) > 0.001

    row = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "tour":            tour,
        "tournament":      tournament,
        "surface":         surface,
        "match_date":      match_date.isoformat(),
        "player_a":        player_a,
        "player_b":        player_b,
        "p_a_raw":         round(pred["p_a_raw"], 4),
        "p_a_cal":         round(pred["p_a_cal"], 4),
        "p_b_raw":         round(pred["p_b_raw"], 4),
        "p_b_cal":         round(pred["p_b_cal"], 4),
        "shrink_hi":       shrink_hi,
        "shrink_lo":       shrink_lo,
        "shrink_rate":     shrink_rate,
        "shrinkage_applied": shrink,
        "odds_a":          odds_a,
        "odds_a_source":   odds_a_source,
        "odds_b":          odds_b,
        "odds_b_source":   odds_b_source,
        "implied_a":       round(val_a["implied_prob"], 4),
        "implied_b":       round(val_b["implied_prob"], 4),
        "edge_a":          round(val_a["edge"], 4),
        "ev_a":            round(val_a["ev"], 4),
        "kelly_a":         round(val_a["kelly_fraction"], 4),
        "edge_b":          round(val_b["edge"], 4),
        "ev_b":            round(val_b["ev"], 4),
        "kelly_b":         round(val_b["kelly_fraction"], 4),
        "model_commit":       git_utils.current_git_commit(),
        "model_snapshot_id":  model_snapshot_id or "",
        "elo_found_a":     pred.get("elo_found_a", True),
        "elo_found_b":     pred.get("elo_found_b", True),
        "decision":        decision,
    }

    write_header = not AUDIT_LOG_PATH.exists()
    with open(AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_AUDIT_LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)

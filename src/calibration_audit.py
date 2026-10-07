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
    "bookmaker_a", "bookmaker_b",
    "implied_a", "implied_b",
    "edge_a", "ev_a", "kelly_a",
    "edge_b", "ev_b", "kelly_b",
    "model_commit", "model_snapshot_id",
    "elo_found_a", "elo_found_b",
    "decision",  # logged / passed_low_edge / passed_{PASSED_REASONS} / blocked_* / invalid_missing_elo
]


PASSED_REASONS = ("user_declined", "duplicate", "tour_excluded", "below_min_edge")


def classify_audit_decision(
    low_sample: bool, suspicious_edge: bool, elo_ok: bool, logged: bool,
    has_value_a: bool, has_value_b: bool, passed_reason: str = "user_declined",
) -> str:
    """Classify what happened to one evaluated prediction, for the audit
    log's `decision` column.

    Priority order matters and is deliberately defensive (checked in this
    order even though some combinations shouldn't occur together in
    practice):

    1. Missing Elo (elo_ok=False, i.e. a player with 0 recorded matches)
       always wins — it's the most specific diagnosis available and is
       itself a special case of low_sample, so it must be checked first or
       it would never be reachable.
    2. low_sample — at least one player has fewer than
       src.config.MIN_MATCHES_THRESHOLD matches. A thin sample is often the
       root cause of an inflated edge, so it's reported ahead of the
       edge-magnitude check below even when both are true for the same
       prediction.
    3. suspicious_edge — edge exceeds src.config.MAX_SUSPICIOUS_EDGE despite
       an adequate sample; still not trustworthy enough to act on.
    4. Then whether the user actually logged it to value_bets_log.csv.
    5. Then whether it even had positive edge on either side to log in the
       first place.
    6. Otherwise it had value but wasn't logged, and `passed_reason` says
       why (PASSED_REASONS): the user said no (the default, for the
       interactive paths), it was already in the log ("duplicate"), its
       tour is excluded from auto-logging ("tour_excluded"), or its edge
       was below the workflow's MIN_EDGE ("below_min_edge"). Until
       2026-10-07 all four were recorded as passed_user_declined.
    """
    if passed_reason not in PASSED_REASONS:
        raise ValueError(f"passed_reason desconocido: {passed_reason!r}")
    if not elo_ok:
        return "invalid_missing_elo"
    if low_sample:
        return "blocked_low_sample"
    if suspicious_edge:
        return "blocked_suspicious_edge"
    if logged:
        return "logged"
    if not has_value_a and not has_value_b:
        return "passed_low_edge"
    return f"passed_{passed_reason}"


def _migrate_log_header_if_needed() -> None:
    """If AUDIT_LOG_PATH exists with an older header than _AUDIT_LOG_FIELDS
    (e.g. missing bookmaker_a/bookmaker_b), rewrite it with the current
    header so old rows stay readable instead of silently misaligning on the
    next append. Mirrors src.value_analysis._migrate_log_header_if_needed
    for the sibling value_bets_log.csv."""
    if not AUDIT_LOG_PATH.exists():
        return
    with open(AUDIT_LOG_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == _AUDIT_LOG_FIELDS:
            return
        rows = list(reader)
    for row in rows:
        row.setdefault("bookmaker_a", "")
        row.setdefault("bookmaker_b", "")
    with open(AUDIT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_AUDIT_LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _AUDIT_LOG_FIELDS})


def log_prediction_audit(
    tour, tournament, surface, match_date,
    player_a, player_b,
    pred: dict, val_a: dict, val_b: dict, odds_a: float, odds_b: float,
    odds_a_source: str, odds_b_source: str,
    decision: str,
    model_snapshot_id: str | None,
    shrink_hi: float, shrink_lo: float, shrink_rate: float,
    bookmaker_a: str = "", bookmaker_b: str = "",
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
    _migrate_log_header_if_needed()
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
        "bookmaker_a":     bookmaker_a,
        "bookmaker_b":     bookmaker_b,
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

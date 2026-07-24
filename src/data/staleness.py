"""Context-aware dataset staleness checks.

The flat 30-day threshold previously used in src/value_analysis.py missed a
dataset that was functionally stale during a live Wimbledon SF at only 16
days old (see docs/superpowers/specs/2026-07-13-staleness-context-aware-design.md)
— 16 days is fine for the normal between-tournament cadence but a real gap
once a relevant tournament is underway. This module tightens or loosens the
effective window based on calendar context instead of one fixed number.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

import pandas as pd

LIVE_WARNING_DAYS = 1
LIVE_CRITICAL_DAYS = 2
REGULAR_WARNING_DAYS = 3
REGULAR_CRITICAL_DAYS = 7
OFFSEASON_TOLERANCE_DAYS = 45

# Months in which a last-match date counts as "end of the regular season" —
# required, alongside the off-season calendar window below, before the
# 45-day grace applies. A stale dataset whose last match predates the
# season's tail end (e.g. last match in August, checked in December) is not
# a normal off-season lull, it's just stale, and should not get the grace.
_END_OF_SEASON_MONTHS = frozenset({10, 11, 12})


class StalenessLevel(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class StalenessReport:
    level: StalenessLevel
    days_stale: int
    rule: str  # "live_tournament" | "off_season" | "regular_week"
    message: str


def _is_offseason_reference(d: date) -> bool:
    """December through the first half of January."""
    return d.month == 12 or (d.month == 1 and d.day <= 15)


def _is_end_of_regular_season(d: date) -> bool:
    return d.month in _END_OF_SEASON_MONTHS


def _classify(days_stale: int, warning_days: int, critical_days: int) -> StalenessLevel:
    if days_stale > critical_days:
        return StalenessLevel.CRITICAL
    if days_stale > warning_days:
        return StalenessLevel.WARNING
    return StalenessLevel.OK


def _build_message(
    level: StalenessLevel, rule: str, days_stale: int, last_match_date: date,
) -> str:
    if level == StalenessLevel.OK:
        return (
            f"Dataset actualizado ({days_stale}d desde el ultimo partido, "
            f"regla: {rule})."
        )
    tag = "ADVERTENCIA" if level == StalenessLevel.WARNING else "CRITICO"
    return (
        f"*** {tag}: dataset desactualizado (regla: {rule}) ***\n"
        f"*** Ultimo partido en los datos: {last_match_date} ({days_stale} dias atras)."
    )


def evaluate_staleness(
    last_match_date: date,
    reference_date: date,
    live_tournament_mode: bool = False,
) -> StalenessReport:
    """Classify how stale `last_match_date` is as of `reference_date`.

    Rules, in priority order:
      1. live_tournament_mode=True: strict — WARNING >1 day, CRITICAL >2 days.
         Overrides the off-season grace even if reference_date falls inside
         the off-season calendar window (an explicit live-tracking flag from
         the caller should never be relaxed by a calendar heuristic).
      2. reference_date in the off-season window (Dec 1 - Jan 15) AND
         last_match_date is from the end of the regular season (Oct-Dec):
         lenient — OK up to OFFSEASON_TOLERANCE_DAYS (45), CRITICAL beyond
         that (no separate WARNING tier: the 45-day allowance is already
         generous, so exceeding it is treated as a clear problem rather than
         a soft warning).
      3. Otherwise (regular tournament weeks): WARNING >3 days, CRITICAL >7 days.
    """
    days_stale = (reference_date - last_match_date).days

    if live_tournament_mode:
        rule = "live_tournament"
        level = _classify(days_stale, LIVE_WARNING_DAYS, LIVE_CRITICAL_DAYS)
    elif _is_offseason_reference(reference_date) and _is_end_of_regular_season(last_match_date):
        rule = "off_season"
        level = (
            StalenessLevel.OK
            if days_stale <= OFFSEASON_TOLERANCE_DAYS
            else StalenessLevel.CRITICAL
        )
    else:
        rule = "regular_week"
        level = _classify(days_stale, REGULAR_WARNING_DAYS, REGULAR_CRITICAL_DAYS)

    message = _build_message(level, rule, days_stale, last_match_date)
    return StalenessReport(level=level, days_stale=days_stale, rule=rule, message=message)


def check_dataset_staleness(
    df: pd.DataFrame,
    reference_date: Optional[date] = None,
    live_tournament_mode: bool = False,
) -> StalenessReport:
    """Evaluate how stale `df`'s most recent match is.

    df must have a 'match_date' column (as produced by src/data/loader.py for
    both ATP and WTA) with at least one non-null value; the newest value is
    used regardless of row order. reference_date defaults to today.
    """
    if reference_date is None:
        reference_date = date.today()

    last_match = df["match_date"].max()
    if pd.isna(last_match):
        raise ValueError("check_dataset_staleness: df has no non-null match_date values")
    if hasattr(last_match, "date"):
        last_match = last_match.date()

    return evaluate_staleness(last_match, reference_date, live_tournament_mode)

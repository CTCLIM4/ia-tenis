"""Explicit America/Lima conversion for timestamps sourced from The Odds
API (UTC).

The Odds API reports commence_time in UTC. Taking .date() directly off
that UTC datetime silently mislabels matches near the UTC day boundary
(e.g. a 02:00Z match is still the previous day in Lima) — to_lima() forces
that conversion to happen explicitly, everywhere match_date is derived.
See docs/superpowers/specs/2026-07-30-timezone-lima-conversion-design.md.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

LIMA_TZ = ZoneInfo("America/Lima")


def to_lima(dt: datetime) -> datetime:
    """Convert an aware datetime to America/Lima.

    Raises ValueError on a naive datetime — its origin timezone is
    ambiguous and must never be guessed (same "never guess" convention as
    src.player_matcher and src.surface_resolver).
    """
    if dt.tzinfo is None:
        raise ValueError("to_lima() requires a timezone-aware datetime")
    return dt.astimezone(LIMA_TZ)


def lima_today() -> date:
    """Today's calendar date in America/Lima — the app's canonical 'today'."""
    return datetime.now(LIMA_TZ).date()

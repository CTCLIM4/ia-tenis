"""Age/inactivity decay features layered on top of Elo.

Motivation: a veteran's blended Elo (models/elo.py) is slow-moving by design
— it doesn't distinguish "consistently great for a decade" from "was great,
now aging/injured/rusty". These features give the LR signal to tell those
apart (e.g. Wawrinka 2026: huge career clay Elo, 41yo, 7-14 record, largely
inactive — historical Elo alone overstates him against a healthy opponent).
"""
from collections import deque
from datetime import date, timedelta
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

ROLLING_WINDOW       = 15
MIN_ROLLING_MATCHES  = 5
RUST_WINDOW_DAYS      = 60
RUST_TARGET_MATCHES   = 4
AGE_PENALTY_START     = 30
AGE_PENALTY_RATE      = 0.025
AGE_PENALTY_FLOOR     = 0.60
FATIGUE_WINDOW_SHORT_DAYS = 7
FATIGUE_WINDOW_LONG_DAYS  = 14
FATIGUE_MATCH_TARGET_7D   = 3
FATIGUE_SET_TARGET_7D     = 6
FATIGUE_MATCH_TARGET_14D  = 5
FATIGUE_SET_TARGET_14D    = 10
FATIGUE_RECENT_WEIGHT     = 0.6
FATIGUE_PENALTY_MAX       = 0.15
SURFACE_TRANSITION_WINDOW_DAYS  = 10
SURFACE_TRANSITION_PENALTY_MAX  = 0.10
SURFACE_TRANSITION_SEVERITY = {
    frozenset({"clay", "grass"}): 1.0,
    frozenset({"clay", "hard"}): 0.5,
    frozenset({"grass", "hard"}): 0.5,
}
_DEFAULT_TRANSITION_SEVERITY = 0.5   # carpet/unknown pairs — rare, safe default


class EloHistoryTracker:
    """Records each player's pre-match effective Elo, per surface, so a
    short recent window can be compared against the slow-moving blended
    rating (see rolling_elo_diff in calculate_decay_features)."""

    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
        # Plain dict, not defaultdict(lambda: ...) — a lambda closure isn't
        # picklable, and this object is persisted in data/model_cache/*.pkl.
        self._snapshots: Dict[Tuple[str, str], Deque[float]] = {}

    def record(self, player: str, surface: str, elo_value: float) -> None:
        key = (player, surface)
        if key not in self._snapshots:
            self._snapshots[key] = deque(maxlen=self.window)
        self._snapshots[key].append(elo_value)

    def rolling_elo(self, player: str, surface: str) -> Optional[float]:
        values = self._snapshots.get((player, surface))
        if values is None or len(values) < MIN_ROLLING_MATCHES:
            return None
        weights = np.linspace(0.5, 1.0, len(values))
        return float(np.average(values, weights=weights))


def age_multiplier(age: Optional[float]) -> float:
    """1.0 up to age 30, decaying 2.5pp per year after, floored at 0.60.
    None (age unknown, e.g. WTA — no birthdate in the current data source)
    is treated as neutral."""
    if age is None or age <= AGE_PENALTY_START:
        return 1.0
    return max(AGE_PENALTY_FLOOR, 1.0 - (age - AGE_PENALTY_START) * AGE_PENALTY_RATE)


def rust_factor(
    match_dates: List[date],
    current_date: date,
    window_days: int = RUST_WINDOW_DAYS,
) -> float:
    """Fraction of RUST_TARGET_MATCHES played in the last window_days.

    A brand-new player with no match history at all (match_dates == [])
    has nothing to judge inactivity against, so this returns neutral (1.0)
    rather than 0.0 — only a player with a career who has gone quiet
    recently should be penalized.
    """
    if not match_dates:
        return 1.0
    cutoff = current_date - timedelta(days=window_days)
    n_recent = sum(1 for d in match_dates if cutoff <= d < current_date)
    return min(1.0, n_recent / RUST_TARGET_MATCHES)


def _count_window(
    workload_history: List[Tuple[date, int]], current_date: date, window_days: int,
) -> Tuple[int, int]:
    """(matches, sets) played strictly before current_date, within the
    trailing window_days — same cutoff convention as rust_factor
    (cutoff <= d < current_date, never includes the current match)."""
    cutoff = current_date - timedelta(days=window_days)
    sets_in_window = [sets for d, sets in workload_history if cutoff <= d < current_date]
    return len(sets_in_window), sum(sets_in_window)


def _window_load(matches: int, sets: int, match_target: int, set_target: int) -> float:
    match_load = min(1.0, matches / match_target)
    set_load = min(1.0, sets / set_target)
    return 0.5 * match_load + 0.5 * set_load


def fatigue_multiplier(workload_history: List[Tuple[date, int]], current_date: date) -> float:
    """1.0 = fresh, down to (1 - FATIGUE_PENALTY_MAX) at maximum load in
    both the 7-day and 14-day windows. A player with no history returns
    1.0 (nothing to judge overload against — same convention as
    rust_factor for brand-new players)."""
    matches_7d, sets_7d = _count_window(workload_history, current_date, FATIGUE_WINDOW_SHORT_DAYS)
    matches_14d, sets_14d = _count_window(workload_history, current_date, FATIGUE_WINDOW_LONG_DAYS)

    load_7d = _window_load(matches_7d, sets_7d, FATIGUE_MATCH_TARGET_7D, FATIGUE_SET_TARGET_7D)
    load_14d = _window_load(matches_14d, sets_14d, FATIGUE_MATCH_TARGET_14D, FATIGUE_SET_TARGET_14D)

    fatigue_load = FATIGUE_RECENT_WEIGHT * load_7d + (1 - FATIGUE_RECENT_WEIGHT) * load_14d
    return 1.0 - FATIGUE_PENALTY_MAX * fatigue_load


def surface_transition_multiplier(
    last_surface_and_date: Optional[Tuple[date, str]],
    current_surface: str,
    current_date: date,
) -> float:
    """1.0 = no recent surface change (no history, same surface, or the
    switch happened more than SURFACE_TRANSITION_WINDOW_DAYS ago). Down to
    (1 - SURFACE_TRANSITION_PENALTY_MAX * severity) immediately after
    switching, linearly decaying back to 1.0 over the window."""
    if last_surface_and_date is None:
        return 1.0
    last_date, last_surface = last_surface_and_date
    if last_surface == current_surface:
        return 1.0
    days_since = (current_date - last_date).days
    if not (0 <= days_since < SURFACE_TRANSITION_WINDOW_DAYS):
        return 1.0
    severity = SURFACE_TRANSITION_SEVERITY.get(
        frozenset({last_surface, current_surface}), _DEFAULT_TRANSITION_SEVERITY
    )
    recency = (SURFACE_TRANSITION_WINDOW_DAYS - days_since) / SURFACE_TRANSITION_WINDOW_DAYS
    return 1.0 - SURFACE_TRANSITION_PENALTY_MAX * severity * recency


def calculate_decay_features(
    historical_elo_surface: float,
    tracker: EloHistoryTracker,
    player: str,
    surface: str,
    match_dates: List[date],
    current_date: date,
    player_age: Optional[float],
    workload_history: Optional[List[Tuple[date, int]]] = None,
) -> dict:
    """Per-player decay-adjustment features for one side of a matchup.

    historical_elo_surface: the player's pre-match effective surface Elo
        (elo.get_effective_rating(player, surface)) — passed in rather than
        recomputed here since callers already have it for elo_diff.
    match_dates: this player's own past match dates (any surface), used
        only for the rust_factor recency window.
    workload_history: this player's own (match_date, sets_played) history,
        used for fatigue_multiplier — defaults to empty (neutral fatigue)
        for callers that don't track it yet.
    """
    rolling = tracker.rolling_elo(player, surface)
    rolling_elo_diff = 0.0 if rolling is None else historical_elo_surface - rolling

    mult = age_multiplier(player_age)
    rust = rust_factor(match_dates, current_date)
    fatigue = fatigue_multiplier(workload_history or [], current_date)
    adjusted_elo_surface = historical_elo_surface * mult * rust * fatigue

    return {
        "rolling_elo_diff":     rolling_elo_diff,
        "age_multiplier":       mult,
        "rust_factor":          rust,
        "fatigue_multiplier":   fatigue,
        "adjusted_elo_surface": adjusted_elo_surface,
    }

from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

H2H_SHRINKAGE_K = 4


def _weighted_h2h_rate(matches: List[Tuple[date, bool]]) -> float:
    """Recency-weighted, sample-size-shrunk head-to-head win rate.

    matches: chronological (oldest first) list of (date, player_won) for
    this specific player-vs-opponent pairing.

    Recency weighting reuses the same linspace(0.5, 1.0, n) shape as
    EloHistoryTracker.rolling_elo, for consistency across the codebase.
    Shrinkage toward 0.5 (K=4) keeps a single meeting from swinging the
    rate as hard as a real sample would — see design doc for the exact
    algebraic proof that this still satisfies rate(B,A) == 1 - rate(A,B).
    """
    n = len(matches)
    if n == 0:
        return 0.5
    weights = np.linspace(0.5, 1.0, n)
    wins = np.array([1.0 if won else 0.0 for _, won in matches])
    weighted_rate = float(np.average(wins, weights=weights))
    shrink = n / (n + H2H_SHRINKAGE_K)
    return shrink * weighted_rate + (1 - shrink) * 0.5


class FeatureBuilder:
    def __init__(self, recent_n: int = 10, surface_n: int = 10):
        self.recent_n = recent_n
        self.surface_n = surface_n
        # player -> [(match_date, surface, won)]
        self._history: Dict[str, List[Tuple[date, str, bool]]] = defaultdict(list)
        # (player, opponent) -> [(match_date, player_won)], both directions stored
        self._h2h_matches: Dict[Tuple[str, str], List[Tuple[date, bool]]] = defaultdict(list)
        # player -> [(match_date, sets_played)] — for fatigue/workload
        self._workload_history: Dict[str, List[Tuple[date, int]]] = defaultdict(list)
        self._last_match_date: Dict[str, Optional[date]] = {}

    def _win_rate(self, results: List[bool]) -> float:
        if not results:
            return 0.5
        return sum(results) / len(results)

    def _trend(self, results: List[bool], min_n: int = 4) -> float:
        """Recent-half win rate minus earlier-half win rate: positive means
        improving, negative means declining. 0.0 (neutral) with fewer than
        min_n results to split into two meaningfully-sized halves."""
        n = len(results)
        if n < min_n:
            return 0.0
        half = n // 2
        recent_half = results[-half:]
        earlier_half = results[:-half]
        return self._win_rate(recent_half) - self._win_rate(earlier_half)

    def get_features(self, player: str, opponent: str, surface: str, match_date: date) -> dict:
        history = self._history[player]

        recent = [won for _, _, won in history[-self.recent_n :]]
        recent_surface = [won for _, s, won in history if s == surface][-self.surface_n :]

        h2h_matches = self._h2h_matches[(player, opponent)]
        h2h_rate = _weighted_h2h_rate(h2h_matches)
        h2h_trend = self._trend([won for _, won in h2h_matches])

        last = self._last_match_date.get(player)
        rest_days = (match_date - last).days if last is not None else 14

        all_results = [won for _, _, won in history]
        surface_results = [won for _, s, won in history if s == surface]

        return {
            "recent_win_rate": self._win_rate(recent),
            "recent_win_rate_surface": self._win_rate(recent_surface) if recent_surface else 0.5,
            "h2h_win_rate": h2h_rate,
            "h2h_matches": len(h2h_matches),
            "h2h_trend": h2h_trend,
            "rest_days": float(rest_days),
            "momentum_3": self._win_rate(all_results[-3:]),
            "momentum_5": self._win_rate(all_results[-5:]),
            "surface_win_rate_trend": self._trend(surface_results),
        }

    def match_dates(self, player: str) -> List[date]:
        """This player's own past match dates (any surface), oldest first."""
        return [d for d, _, _ in self._history[player]]

    def workload_history(self, player: str) -> List[Tuple[date, int]]:
        """This player's own (match_date, sets_played) history, oldest first."""
        return list(self._workload_history[player])

    def last_surface_and_date(self, player: str) -> Optional[Tuple[date, str]]:
        """This player's most recent match's (date, surface), or None if
        they have no history yet."""
        history = self._history[player]
        if not history:
            return None
        d, s, _ = history[-1]
        return (d, s)

    def update(self, winner: str, loser: str, surface: str, match_date: date, sets_played: int = 0) -> None:
        self._history[winner].append((match_date, surface, True))
        self._history[loser].append((match_date, surface, False))
        self._h2h_matches[(winner, loser)].append((match_date, True))
        self._h2h_matches[(loser, winner)].append((match_date, False))
        self._workload_history[winner].append((match_date, sets_played))
        self._workload_history[loser].append((match_date, sets_played))
        self._last_match_date[winner] = match_date
        self._last_match_date[loser] = match_date

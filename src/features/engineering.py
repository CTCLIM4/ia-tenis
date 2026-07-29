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
        self._last_match_date: Dict[str, Optional[date]] = {}

    def _win_rate(self, results: List[bool]) -> float:
        if not results:
            return 0.5
        return sum(results) / len(results)

    def get_features(self, player: str, opponent: str, surface: str, match_date: date) -> dict:
        history = self._history[player]

        recent = [won for _, _, won in history[-self.recent_n :]]
        recent_surface = [won for _, s, won in history if s == surface][-self.surface_n :]

        h2h_matches = self._h2h_matches[(player, opponent)]
        h2h_rate = _weighted_h2h_rate(h2h_matches)

        last = self._last_match_date.get(player)
        rest_days = (match_date - last).days if last is not None else 14

        return {
            "recent_win_rate": self._win_rate(recent),
            "recent_win_rate_surface": self._win_rate(recent_surface) if recent_surface else 0.5,
            "h2h_win_rate": h2h_rate,
            "h2h_matches": len(h2h_matches),
            "rest_days": float(rest_days),
        }

    def match_dates(self, player: str) -> List[date]:
        """This player's own past match dates (any surface), oldest first."""
        return [d for d, _, _ in self._history[player]]

    def update(self, winner: str, loser: str, surface: str, match_date: date) -> None:
        self._history[winner].append((match_date, surface, True))
        self._history[loser].append((match_date, surface, False))
        self._h2h_matches[(winner, loser)].append((match_date, True))
        self._h2h_matches[(loser, winner)].append((match_date, False))
        self._last_match_date[winner] = match_date
        self._last_match_date[loser] = match_date

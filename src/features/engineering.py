from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple


class FeatureBuilder:
    def __init__(self, recent_n: int = 10, surface_n: int = 10):
        self.recent_n = recent_n
        self.surface_n = surface_n
        # player -> [(match_date, surface, won)]
        self._history: Dict[str, List[Tuple[date, str, bool]]] = defaultdict(list)
        # (winner, loser) -> count of times winner beat loser
        self._h2h_wins: Dict[Tuple[str, str], int] = defaultdict(int)
        self._last_match_date: Dict[str, Optional[date]] = {}

    def _win_rate(self, results: List[bool]) -> float:
        if not results:
            return 0.5
        return sum(results) / len(results)

    def get_features(self, player: str, opponent: str, surface: str, match_date: date) -> dict:
        history = self._history[player]

        recent = [won for _, _, won in history[-self.recent_n :]]
        recent_surface = [won for _, s, won in history if s == surface][-self.surface_n :]

        p_wins = self._h2h_wins[(player, opponent)]
        o_wins = self._h2h_wins[(opponent, player)]
        total_h2h = p_wins + o_wins
        h2h_rate = p_wins / total_h2h if total_h2h > 0 else 0.5

        last = self._last_match_date.get(player)
        rest_days = (match_date - last).days if last is not None else 14

        return {
            "recent_win_rate": self._win_rate(recent),
            "recent_win_rate_surface": self._win_rate(recent_surface) if recent_surface else 0.5,
            "h2h_win_rate": h2h_rate,
            "h2h_matches": total_h2h,
            "rest_days": float(rest_days),
        }

    def update(self, winner: str, loser: str, surface: str, match_date: date) -> None:
        self._history[winner].append((match_date, surface, True))
        self._history[loser].append((match_date, surface, False))
        self._h2h_wins[(winner, loser)] += 1
        self._last_match_date[winner] = match_date
        self._last_match_date[loser] = match_date

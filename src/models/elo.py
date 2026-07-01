from datetime import date
from typing import Dict, Optional, Tuple


class EloSystem:
    def __init__(
        self,
        k: float = 32.0,
        initial_rating: float = 1500.0,
        decay_factor: float = 0.75,
        surface_blend_cap: int = 250,
    ):
        self.k = k
        self.initial_rating = initial_rating
        self.decay_factor = decay_factor          # fraction of rating retained at year boundary
        self.surface_blend_cap = surface_blend_cap  # w = n / (n + cap)

        self.general_ratings: Dict[str, float] = {}
        self.surface_ratings: Dict[Tuple[str, str], float] = {}
        self.match_counts: Dict[str, int] = {}
        self.surface_match_counts: Dict[Tuple[str, str], int] = {}
        self.current_year: Dict[str, Optional[int]] = {}

    # ── accessors ──────────────────────────────────────────────────────────────

    def get_general_rating(self, player: str) -> float:
        return self.general_ratings.get(player, self.initial_rating)

    def get_surface_rating(self, player: str, surface: str) -> float:
        return self.surface_ratings.get((player, surface), self.initial_rating)

    def get_blend_weight(self, player: str, surface: str) -> float:
        n = self.surface_match_counts.get((player, surface), 0)
        return n / (n + self.surface_blend_cap)

    def get_effective_rating(self, player: str, surface: str) -> float:
        w = self.get_blend_weight(player, surface)
        return w * self.get_surface_rating(player, surface) + (1 - w) * self.get_general_rating(player)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def get_k(self, player: str) -> float:
        n = self.match_counts.get(player, 0)
        if n < 30:
            return self.k
        if n < 100:
            return self.k * 0.75
        return self.k * 0.5

    # ── internal ───────────────────────────────────────────────────────────────

    def _apply_yearly_decay(self, player: str, match_year: int) -> None:
        prev = self.current_year.get(player)
        if prev is not None and match_year > prev:
            r = self.general_ratings.get(player, self.initial_rating)
            self.general_ratings[player] = (
                self.decay_factor * r + (1 - self.decay_factor) * self.initial_rating
            )
            for key in list(self.surface_ratings):
                if key[0] == player:
                    sr = self.surface_ratings[key]
                    self.surface_ratings[key] = (
                        self.decay_factor * sr + (1 - self.decay_factor) * self.initial_rating
                    )
        self.current_year[player] = match_year

    # ── public update ──────────────────────────────────────────────────────────

    def update(self, winner: str, loser: str, surface: str, match_date: date) -> float:
        """Process one match. Returns Elo win-probability for the winner (pre-update)."""
        self._apply_yearly_decay(winner, match_date.year)
        self._apply_yearly_decay(loser, match_date.year)

        # Pre-match prediction
        ra = self.get_effective_rating(winner, surface)
        rb = self.get_effective_rating(loser, surface)
        prob_winner = self.expected_score(ra, rb)

        k_w = self.get_k(winner)
        k_l = self.get_k(loser)

        # Update general ratings
        ga = self.get_general_rating(winner)
        gb = self.get_general_rating(loser)
        ea = self.expected_score(ga, gb)
        self.general_ratings[winner] = ga + k_w * (1.0 - ea)
        self.general_ratings[loser] = gb + k_l * (ea - 1.0)   # = k_l * (0 - (1-ea))

        # Update surface ratings
        sa = self.get_surface_rating(winner, surface)
        sb = self.get_surface_rating(loser, surface)
        ea_s = self.expected_score(sa, sb)
        self.surface_ratings[(winner, surface)] = sa + k_w * (1.0 - ea_s)
        self.surface_ratings[(loser, surface)] = sb + k_l * (ea_s - 1.0)

        # Update counters
        self.match_counts[winner] = self.match_counts.get(winner, 0) + 1
        self.match_counts[loser] = self.match_counts.get(loser, 0) + 1
        self.surface_match_counts[(winner, surface)] = (
            self.surface_match_counts.get((winner, surface), 0) + 1
        )
        self.surface_match_counts[(loser, surface)] = (
            self.surface_match_counts.get((loser, surface), 0) + 1
        )

        return prob_winner

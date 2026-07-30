"""Módulo 4: Simulación Monte Carlo de Banca (Kelly).

Proyecta trayectorias futuras de banca bajo distintas fracciones de Kelly
(completo/medio/cuarto), calibradas con el perfil de edge/odds de las
apuestas ya resueltas en data/value_bets_log.csv, para estimar banca
esperada (P10/P50/P90) y probabilidad de ruina a N apuestas.

Uso:
  python -m src.bankroll_simulation
  python -m src.bankroll_simulation --bets 100 --simulations 20000
  python -m src.bankroll_simulation --mean-edge 0.05 --std-edge 0.02 \
      --mean-odds 2.0 --std-odds 0.3   # perfil manual, sin leer el CSV

Ver docs/superpowers/specs/2026-07-29-bankroll-simulation-design.md para el
diseño completo (fórmulas y rationale).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.value_analysis import KELLY_CAP


def estimate_bet_profile(df: pd.DataFrame) -> dict:
    """Derive an edge/odds sampling profile from resolved bets.

    Expects the DataFrame shape produced by
    backtest_analytics.load_resolved_bets (has bet_side and odds_taken
    already derived, plus the raw edge_a/edge_b columns). std falls back to
    0.0 when there's only one row (nothing to estimate dispersion from).
    """
    if df.empty:
        raise ValueError("No hay apuestas resueltas de donde estimar el perfil.")

    edge_taken = df["edge_a"].where(df["bet_side"] == "a", df["edge_b"])
    n = len(df)
    return {
        "mean_edge": edge_taken.mean(),
        "std_edge": edge_taken.std(ddof=1) if n > 1 else 0.0,
        "mean_odds": df["odds_taken"].mean(),
        "std_odds": df["odds_taken"].std(ddof=1) if n > 1 else 0.0,
        "n": n,
    }


def sample_bet(rng: np.random.Generator, profile: dict) -> tuple[float, float]:
    """Draw one synthetic (p_win, odds) pair from the calibrated profile.

    edge ~ Normal(mean_edge, std_edge), floored above 0 (the historical log
    only ever contains positive-edge bets by construction of
    value_analysis.calculate_value, so a non-positive draw isn't a
    realistic sample — it's floored rather than treated as "skip this bet",
    which would complicate comparing Kelly fractions on equal footing).

    odds ~ Lognormal fit by method of moments to have the given mean/std
    (collapses to exactly mean_odds when std_odds == 0, since sigma
    becomes 0). p_win is implied_prob (1/odds) + edge, clipped away from
    0/1 so Kelly can't degenerate.
    """
    edge = rng.normal(profile["mean_edge"], profile["std_edge"])
    edge = max(edge, 0.001)

    mean_odds = profile["mean_odds"]
    std_odds = profile["std_odds"]
    sigma2 = math.log(1 + (std_odds / mean_odds) ** 2)
    mu = math.log(mean_odds) - sigma2 / 2
    odds = rng.lognormal(mu, math.sqrt(sigma2))
    odds = max(odds, 1.01)

    implied_prob = 1 / odds
    p_win = min(max(implied_prob + edge, 0.01), 0.99)
    return p_win, odds


def kelly_stake(p_win: float, odds: float, kelly_multiplier: float) -> float:
    """Same convention as value_analysis.calculate_value's Kelly formula,
    with an extra multiplier applied before the cap (so half/quarter Kelly
    only differ from full Kelly when the cap isn't already binding)."""
    implied_prob = 1 / odds
    edge = p_win - implied_prob
    raw_kelly = edge / (odds - 1) if edge > 0 else 0.0
    return min(kelly_multiplier * raw_kelly, KELLY_CAP)

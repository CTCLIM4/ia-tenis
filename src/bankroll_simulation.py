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


def simulate_path(
    rng: np.random.Generator,
    profile: dict,
    kelly_multiplier: float,
    n_bets: int,
    initial_bankroll: float,
) -> np.ndarray:
    """Simulate one bankroll trajectory of n_bets synthetic bets.

    Stakes a fraction of the CURRENT bankroll each bet (compounding), not
    a fraction of the fixed initial bankroll — this is the whole point of
    the Kelly criterion (geometric growth) and is why half-Kelly reduces
    ruin risk so much in practice. This differs deliberately from Módulo
    2's linear equity curve, which is correct for reporting a fixed
    historical sequence but not for a forward-looking Kelly simulation.
    """
    bankroll = initial_bankroll
    path = [bankroll]
    for _ in range(n_bets):
        p_win, odds = sample_bet(rng, profile)
        stake_frac = kelly_stake(p_win, odds, kelly_multiplier)
        stake_usd = stake_frac * bankroll
        if rng.random() < p_win:
            bankroll += stake_usd * (odds - 1)
        else:
            bankroll -= stake_usd
        bankroll = max(bankroll, 0.0)
        path.append(bankroll)
    return np.array(path)


def run_monte_carlo(
    profile: dict,
    kelly_multipliers: list[float],
    n_simulations: int,
    n_bets: int,
    initial_bankroll: float,
    ruin_threshold: float,
    seed: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for km in kelly_multipliers:
        finals = []
        max_drawdowns_pct = []
        ruined = 0
        for _ in range(n_simulations):
            path = simulate_path(rng, profile, km, n_bets, initial_bankroll)
            finals.append(path[-1])
            running_max = np.maximum.accumulate(path)
            drawdown_pct = np.where(running_max > 0, (path - running_max) / running_max, 0.0)
            max_drawdowns_pct.append(drawdown_pct.min())
            if path.min() < ruin_threshold * initial_bankroll:
                ruined += 1
        finals = np.array(finals)
        results[km] = {
            "p10": float(np.percentile(finals, 10)),
            "p50": float(np.percentile(finals, 50)),
            "p90": float(np.percentile(finals, 90)),
            "ruin_probability": ruined / n_simulations,
            "median_max_drawdown_pct": float(np.median(max_drawdowns_pct)) * 100,
        }

    return {
        "profile": profile,
        "results": results,
        "n_simulations": n_simulations,
        "n_bets": n_bets,
        "initial_bankroll": initial_bankroll,
        "ruin_threshold": ruin_threshold,
    }

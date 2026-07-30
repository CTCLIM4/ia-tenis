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

import pandas as pd


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

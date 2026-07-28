"""Módulo 2: Dashboard de Rendimiento y Análisis de Banca.

Lee el histórico acumulado de data/value_bets_log.csv (apuestas logueadas)
y data/prediction_audit_log.csv (población completa de predicciones
evaluadas) y genera un reporte de consola con métricas financieras, de
riesgo, y una comparación de calibración (EV teórico vs beneficio real).

Uso:
  python -m src.backtest_analytics                    # banca $1000, ambos tours
  python -m src.backtest_analytics --tour atp
  python -m src.backtest_analytics --bankroll 5000

Ver docs/superpowers/specs/2026-07-28-backtest-analytics-design.md para el
diseño completo (fórmulas, filtros, y por qué).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

DEFAULT_BANKROLL = 1000.0


def load_resolved_bets(path: str, tour: Optional[str] = None) -> pd.DataFrame:
    """Load value_bets_log.csv and filter down to real, resolved bets.

    Keeps only status=='ok' and result in {'A_win','B_win'} — excludes test
    rows, missing-Elo rows, and unresolved/discarded rows, which would
    otherwise contaminate financial/risk metrics with data that isn't a real
    settled bet. Also excludes resolved rows where neither side has a
    positive Kelly stake (kelly_a == kelly_b == 0): a no-edge match logged
    without a real bet — reachable via the interactive CLI's save prompt.

    Derives per-row bet_side/stake/odds_taken/ev_theoretical/win from
    whichever side (a/b) actually has kelly_<side> > 0 — deterministic since
    edge_a + edge_b <= 0 always (bookmaker vig), so at most one side can have
    positive Kelly for a given row.
    """
    df = pd.read_csv(path)
    df = df[(df["status"] == "ok") & (df["result"].isin(["A_win", "B_win"]))].copy()
    if tour and tour != "both":
        df = df[df["tour"] == tour]

    # Exclude rows where neither side has a positive Kelly stake: no real bet
    # was placed (e.g. the interactive CLI's "Guardar en log?" prompt can log
    # a no-edge match), so there's nothing to attribute a bet_side/stake to.
    df = df[(df["kelly_a"] > 0) | (df["kelly_b"] > 0)].copy()

    bet_side = df["kelly_a"].gt(0).map({True: "a", False: "b"})
    df["bet_side"] = bet_side
    df["stake"] = df["kelly_a"].where(bet_side == "a", df["kelly_b"])
    df["odds_taken"] = df["odds_a"].where(bet_side == "a", df["odds_b"])
    df["ev_theoretical"] = df["ev_a"].where(bet_side == "a", df["ev_b"])
    df["win"] = df["profit"] > 0
    df["match_date"] = pd.to_datetime(df["match_date"])

    return df.sort_values("match_date").reset_index(drop=True)

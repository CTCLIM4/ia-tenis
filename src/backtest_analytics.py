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


def compute_financial_metrics(df: pd.DataFrame, bankroll: float) -> dict:
    """Compute headline financial metrics from a resolved-bets DataFrame.

    ROI es sum(profit)/sum(stake) — un ratio de fracciones de Kelly, así que
    la banca se cancela y no afecta el resultado; solo se usa para convertir
    a montos en USD (net_profit_usd, avg_stake_usd).
    """
    total_profit = df["profit"].sum()
    total_stake = df["stake"].sum()
    wins = int(df["win"].sum())
    return {
        "roi_pct": (total_profit / total_stake * 100) if total_stake else 0.0,
        "win_rate_pct": df["win"].mean() * 100,
        "wins": wins,
        "losses": len(df) - wins,
        "net_profit_usd": total_profit * bankroll,
        "avg_stake_usd": df["stake"].mean() * bankroll,
        "avg_stake_pct": df["stake"].mean() * 100,
        "num_bets": len(df),
    }


def _max_streak(values: list, target: bool) -> int:
    best = current = 0
    for v in values:
        if v == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def compute_risk_metrics(df: pd.DataFrame, bankroll: float) -> dict:
    """Compute equity-curve-derived risk metrics from a resolved-bets DataFrame.

    Builds the equity curve as bankroll + cumsum(profit)*bankroll, ordered
    chronologically by match_date, then derives max drawdown (both in USD
    and as a % of the running peak) and the longest win/loss streaks.
    """
    ordered = df.sort_values("match_date")
    equity = bankroll + ordered["profit"].cumsum() * bankroll
    running_max = equity.cummax()
    drawdown_usd = equity - running_max

    max_drawdown_usd = drawdown_usd.min()
    trough_idx = drawdown_usd.idxmin()
    peak_at_trough = running_max.loc[trough_idx]
    max_drawdown_pct = (max_drawdown_usd / peak_at_trough * 100) if peak_at_trough else 0.0

    variance = ordered["profit"].var(ddof=1) if len(ordered) > 1 else 0.0

    win_sequence = ordered["win"].tolist()
    return {
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "variance": 0.0 if pd.isna(variance) else variance,
        "max_win_streak": _max_streak(win_sequence, True),
        "max_loss_streak": _max_streak(win_sequence, False),
    }

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

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest_analytics import SMALL_SAMPLE_THRESHOLD, load_resolved_bets
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


_KELLY_LABELS = {1.0: "Kelly completo", 0.5: "1/2 Kelly", 0.25: "1/4 Kelly"}


def _kelly_label(km: float) -> str:
    return _KELLY_LABELS.get(km, f"{km:g}x Kelly")


def print_report(mc_result: dict) -> None:
    profile = mc_result["profile"]
    results = mc_result["results"]
    kellys = list(results.keys())

    print("=" * 60)
    print(" SIMULACION MONTE CARLO DE BANCA - Modulo 4")
    print(f" Banca inicial: ${mc_result['initial_bankroll']:,.2f} | "
          f"{mc_result['n_simulations']} simulaciones x {mc_result['n_bets']} apuestas")
    print("=" * 60)

    n = profile["n"]
    n_suffix = f" | N={n}" if n != "manual" else ""
    print(f"\n Perfil: edge medio {profile['mean_edge'] * 100:.1f}% "
          f"(sd {profile['std_edge'] * 100:.1f}%) | odds medias "
          f"{profile['mean_odds']:.2f} (sd {profile['std_odds']:.2f}){n_suffix}")

    if n != "manual" and n < SMALL_SAMPLE_THRESHOLD:
        print(f"\n  *** Perfil estimado de muestra chica (N={n} < "
              f"{SMALL_SAMPLE_THRESHOLD}) - usar con cautela. ***")

    labels = [_kelly_label(km) for km in kellys]
    col_width = max(14, max(len(l) for l in labels) + 2)

    header = " " * 28 + "".join(f"{l:>{col_width}}" for l in labels)
    print("\n" + header)

    def _row(title, fmt_fn):
        cells = "".join(f"{fmt_fn(results[km]):>{col_width}}" for km in kellys)
        print(f"  {title:<26}{cells}")

    _row("Banca final P10", lambda r: f"${r['p10']:,.2f}")
    _row("Banca final P50 (mediana)", lambda r: f"${r['p50']:,.2f}")
    _row("Banca final P90", lambda r: f"${r['p90']:,.2f}")
    _row("Prob. de ruina", lambda r: f"{r['ruin_probability'] * 100:.1f}%")
    _row("Drawdown maximo esperado", lambda r: f"{r['median_max_drawdown_pct']:.1f}%")


_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BETS_PATH = str(_ROOT / "data" / "value_bets_log.csv")
DEFAULT_BANKROLL = 1000.0
DEFAULT_N_BETS = 50
DEFAULT_N_SIMULATIONS = 10000
DEFAULT_RUIN_THRESHOLD = 0.5
DEFAULT_KELLY_MULTIPLIERS = [1.0, 0.5, 0.25]


def run_simulation(
    bets_path: str = DEFAULT_BETS_PATH,
    tour: str | None = None,
    bankroll: float = DEFAULT_BANKROLL,
    n_bets: int = DEFAULT_N_BETS,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    kelly_multipliers: list[float] | None = None,
    ruin_threshold: float = DEFAULT_RUIN_THRESHOLD,
    seed: int | None = None,
    mean_edge: float | None = None,
    std_edge: float | None = None,
    mean_odds: float | None = None,
    std_odds: float | None = None,
) -> None:
    if kelly_multipliers is None:
        kelly_multipliers = list(DEFAULT_KELLY_MULTIPLIERS)

    if n_bets <= 0:
        raise ValueError(f"n_bets debe ser > 0, recibido {n_bets}")
    if n_simulations <= 0:
        raise ValueError(f"n_simulations debe ser > 0, recibido {n_simulations}")
    if not (0 < ruin_threshold < 1):
        raise ValueError(f"ruin_threshold debe estar en (0, 1), recibido {ruin_threshold}")
    if any(km <= 0 for km in kelly_multipliers):
        raise ValueError(f"kelly_multipliers deben ser > 0, recibido {kelly_multipliers}")
    if bankroll <= 0:
        raise ValueError(f"bankroll debe ser > 0, recibido {bankroll}")

    overrides = [mean_edge, std_edge, mean_odds, std_odds]
    n_overrides = sum(o is not None for o in overrides)
    if n_overrides not in (0, 4):
        print("Debes dar los 4 overrides (--mean-edge --std-edge --mean-odds "
              "--std-odds) juntos, o ninguno.")
        return

    if n_overrides == 4:
        if mean_odds <= 1:
            raise ValueError(f"mean_odds debe ser > 1, recibido {mean_odds}")
        if std_edge < 0:
            raise ValueError(f"std_edge debe ser >= 0, recibido {std_edge}")
        if std_odds < 0:
            raise ValueError(f"std_odds debe ser >= 0, recibido {std_odds}")
        profile = {
            "mean_edge": mean_edge, "std_edge": std_edge,
            "mean_odds": mean_odds, "std_odds": std_odds, "n": "manual",
        }
    else:
        if not Path(bets_path).exists():
            print(f"No se encontro {bets_path} y no diste overrides manuales "
                  "(--mean-edge/--std-edge/--mean-odds/--std-odds).")
            return
        df = load_resolved_bets(bets_path, tour)
        if df.empty:
            print("No hay apuestas resueltas todavia (status='ok' y result en "
                  "A_win/B_win) para estimar el perfil; usa los overrides "
                  "manuales o registra mas apuestas primero.")
            return
        profile = estimate_bet_profile(df)

    mc_result = run_monte_carlo(
        profile, kelly_multipliers, n_simulations, n_bets, bankroll,
        ruin_threshold, seed,
    )
    print_report(mc_result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulacion Monte Carlo de banca bajo Kelly (Modulo 4)"
    )
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--tour", choices=["atp", "wta", "both"], default="both")
    parser.add_argument("--bets-file", default=DEFAULT_BETS_PATH)
    parser.add_argument("--bets", type=int, default=DEFAULT_N_BETS)
    parser.add_argument("--simulations", type=int, default=DEFAULT_N_SIMULATIONS)
    parser.add_argument("--kelly-fractions", default="1.0,0.5,0.25")
    parser.add_argument("--ruin-threshold", type=float, default=DEFAULT_RUIN_THRESHOLD)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mean-edge", type=float, default=None)
    parser.add_argument("--std-edge", type=float, default=None)
    parser.add_argument("--mean-odds", type=float, default=None)
    parser.add_argument("--std-odds", type=float, default=None)
    args = parser.parse_args()

    tour = None if args.tour == "both" else args.tour
    try:
        kelly_multipliers = [float(x) for x in args.kelly_fractions.split(",")]
    except ValueError:
        parser.error(
            f"--kelly-fractions invalido: '{args.kelly_fractions}' "
            "(debe ser una lista de numeros separados por comas, ej. '1.0,0.5,0.25')"
        )

    run_simulation(
        bets_path=args.bets_file,
        tour=tour,
        bankroll=args.bankroll,
        n_bets=args.bets,
        n_simulations=args.simulations,
        kelly_multipliers=kelly_multipliers,
        ruin_threshold=args.ruin_threshold,
        seed=args.seed,
        mean_edge=args.mean_edge,
        std_edge=args.std_edge,
        mean_odds=args.mean_odds,
        std_odds=args.std_odds,
    )


if __name__ == "__main__":
    main()

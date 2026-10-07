"""Nightly settlement: for each pending bet on a given date in
data/value_bets_log.csv, ask for the real result, recompute profit/PnL,
write a jornada report, and push.

Uso:
  python scripts/settle_workflow.py                    # liquida pendientes de hoy
  python scripts/settle_workflow.py --date 2026-08-19
  python scripts/settle_workflow.py --dry-run           # no escribe, no hace git push

Este script NO obtiene resultados de ninguna fuente automatica — el proyecto
no tiene un modulo de resultados confiable, y las jornadas liquidadas a mano
hasta ahora (ver docs/metrics/2026-08-19-jornada-cincinnati-round4.md) ya
mostraron que una sola fuente scrapeada puede alucinar un marcador. Por eso
pide confirmacion explicita del ganador por partido (a/b/skip) y automatiza
todo lo demas: calculo de profit (ver [[feedback-profit-convention]]:
kelly*(odds-1) en victoria, -kelly en derrota), reescritura del CSV, reporte
en docs/metrics/, commit y push.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date as date_cls
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.backtest_analytics import (
    DEFAULT_BANKROLL,
    compute_calibration_metrics,
    compute_financial_metrics,
    compute_risk_metrics,
    load_resolved_bets,
)
from src.git_utils import commit_and_push

VALUE_BETS_LOG = "data/value_bets_log.csv"
DEFAULT_REPORT_DIR = "docs/metrics"


def _parse_date(s: str | None) -> date_cls:
    return date_cls.fromisoformat(s) if s else date_cls.today()


def _read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_rows(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bet_side(row: dict) -> str:
    """Which side ('a' or 'b') actually carries the Kelly stake on this row —
    see src.backtest_analytics.load_resolved_bets, same convention.

    Picks the larger of the two: normally only one side has a positive
    Kelly fraction, but different bookmaker sources for odds_a/odds_b can
    each show their own (small) edge on the same match — the dominant
    position is whichever side actually got the bigger stake."""
    kelly_a = float(row.get("kelly_a") or 0)
    kelly_b = float(row.get("kelly_b") or 0)
    return "a" if kelly_a >= kelly_b else "b"


def _compute_profit(row: dict, winner_side: str) -> tuple[str, float]:
    """profit = kelly*(odds-1) on a win, -kelly on a loss — Kelly-scaled, not
    a flat +-1 unit (see [[feedback-profit-convention]])."""
    bet_side = _bet_side(row)
    kelly = float(row[f"kelly_{bet_side}"])
    odds = float(row[f"odds_{bet_side}"])
    won = winner_side == bet_side
    result = f"{winner_side.upper()}_win"
    profit = kelly * (odds - 1) if won else -kelly
    return result, round(profit, 6)


def _ask_winner(row: dict) -> str | None:
    """Prompt for the real winner. Returns 'a', 'b', or None to skip (leave
    the row pending) — e.g. when the match hasn't been confirmed yet."""
    prompt = (
        f"\n  {row['player_a']} (A, odds {row['odds_a']}) vs "
        f"{row['player_b']} (B, odds {row['odds_b']}) "
        f"[{row['tournament']}] - ganador? (a/b/skip): "
    )
    answer = input(prompt).strip().lower()
    return answer if answer in ("a", "b") else None


def _report_slug(rows: list[dict]) -> str:
    """Filename slug from the tournaments actually settled, e.g. 'ATP US Open'
    and 'WTA US Open' -> 'us-open'. Was hardcoded to 'cincinnati', which
    mislabeled every later jornada (Monterrey, US Open...)."""
    slugs: list[str] = []
    for row in rows:
        name = re.sub(r"^(ATP|WTA|Davis Cup|Davis)\s+", "", row.get("tournament", "").strip(), flags=re.I)
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if slug and slug not in slugs:
            slugs.append(slug)
    return "-".join(slugs)


def _report_filename(target: str, rows: list[dict]) -> str:
    slug = _report_slug(rows)
    return f"{target}-jornada-{slug}.md" if slug else f"{target}-jornada.md"


def _build_report(target: str, settled: list[dict], log_path: str, bankroll: float = DEFAULT_BANKROLL) -> str:
    lines = [f"# Jornada {target} — liquidacion\n"]
    lines.append(f"{len(settled)} apuesta(s) liquidada(s) en `data/value_bets_log.csv`.\n")
    lines.append("| Partido | Pick | Bookmaker | Odds | Kelly | Resultado | Profit |")
    lines.append("|---|---|---|---|---|---|---|")

    stake_total = profit_total = 0.0
    wins = 0
    for row in settled:
        side = _bet_side(row)
        odds = float(row[f"odds_{side}"])
        kelly = float(row[f"kelly_{side}"])
        pick = row["player_a"] if side == "a" else row["player_b"]
        opponent = row["player_b"] if side == "a" else row["player_a"]
        bookmaker = row.get(f"bookmaker_{side}", "")
        profit = float(row["profit"])
        stake_total += kelly
        profit_total += profit
        wins += 1 if profit > 0 else 0
        lines.append(
            f"| {pick} vs {opponent} | {pick} | {bookmaker} | {odds:.2f} | {kelly * 100:.2f}% | "
            f"{row['result']} | {profit:+.4f} |"
        )

    n = len(settled)
    roi_jornada = (profit_total / stake_total * 100) if stake_total else 0.0
    lines.append("")
    lines.append(
        f"**Resumen de la jornada**: {wins}W/{n - wins}L, stake total {stake_total * 100:.2f}% banca, "
        f"beneficio {profit_total * 100:+.3f}% banca, ROI jornada {roi_jornada:+.2f}%."
    )

    df = load_resolved_bets(log_path)
    if not df.empty:
        fin = compute_financial_metrics(df, bankroll)
        risk = compute_risk_metrics(df, bankroll)
        cal = compute_calibration_metrics(df)
        lines.append("")
        lines.append(
            f"**Acumulado del proyecto tras esta jornada** (banca base "
            f"${bankroll:,.0f}, N={fin['num_bets']}):"
        )
        lines.append(
            f"- ROI acumulado: {fin['roi_pct']:+.2f}% | Win rate: {fin['win_rate_pct']:.1f}% "
            f"({fin['wins']}W/{fin['losses']}L)"
        )
        lines.append(
            f"- Beneficio neto: {fin['net_profit_usd']:+.2f}$ | Stake promedio: "
            f"{fin['avg_stake_usd']:.2f}$ ({fin['avg_stake_pct']:.2f}% banca)"
        )
        lines.append(
            f"- Drawdown maximo: {risk['max_drawdown_usd']:.2f}$ ({risk['max_drawdown_pct']:.1f}%) | "
            f"Varianza (profit): {risk['variance']:.5f}"
        )
        lines.append(
            f"- Racha ganadora maxima: {risk['max_win_streak']} | "
            f"Racha perdedora maxima: {risk['max_loss_streak']}"
        )
        lines.append(
            f"- Calibracion: EV teorico {cal['ev_theoretical_avg'] * 100:+.1f}% vs "
            f"retorno real {cal['real_return_avg'] * 100:+.1f}% (diff {cal['difference'] * 100:+.1f} pp)"
        )

    return "\n".join(lines) + "\n"


def run(
    target_date: date_cls,
    dry_run: bool = False,
    log_path: str = VALUE_BETS_LOG,
    report_dir: str = DEFAULT_REPORT_DIR,
) -> list[dict]:
    rows = _read_rows(log_path)
    fieldnames = list(rows[0].keys()) if rows else []
    target = target_date.isoformat()

    settled: list[dict] = []
    for row in rows:
        if row.get("match_date") != target or row.get("status") != "ok" or row.get("result") != "pending":
            continue
        winner = _ask_winner(row)
        if winner is None:
            print(f"    Sin confirmar, se deja pending: {row['player_a']} vs {row['player_b']}")
            continue
        result, profit = _compute_profit(row, winner)
        row["result"] = result
        row["profit"] = profit
        settled.append(row)

    if not settled:
        print(f"\nNo hay apuestas liquidadas para {target}.")
        return []

    if not dry_run:
        _write_rows(log_path, rows, fieldnames)

    # The report reflects every bet resolved for this match_date, not
    # just the ones settled in this particular run — otherwise settling
    # the same date in two batches (e.g. a rain-suspended match
    # confirmed a day later) would silently overwrite the earlier
    # batch's report instead of extending it.
    all_settled_for_date = [
        row for row in rows
        if row.get("match_date") == target and row.get("status") == "ok"
        and row.get("result") not in ("", "pending")
    ]
    report_path = str(Path(report_dir) / _report_filename(target, all_settled_for_date))
    if not dry_run:
        report = _build_report(target, all_settled_for_date, log_path)
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(report, encoding="utf-8")
        print(f"\nReporte escrito en {report_path}")
        commit_and_push(
            [log_path, report_path],
            f"feat(data): settle {len(settled)} bet(s) for {target}, log jornada report",
        )

    return settled


def main() -> None:
    parser = argparse.ArgumentParser(description="Liquidacion nocturna de apuestas pendientes")
    parser.add_argument("--date", default=None, help="Fecha a liquidar (YYYY-MM-DD), default hoy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(_parse_date(args.date), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

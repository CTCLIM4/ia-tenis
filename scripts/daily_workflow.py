"""Daily workflow: scan for value bets, apply the project's filters
non-interactively, log the qualifying picks, notify by email, and push.

Uso:
  python scripts/daily_workflow.py
  python scripts/daily_workflow.py --dry-run   # no escribe CSVs, no envia correo, no hace git push

Reemplaza el flujo manual usado hasta ahora (correr src.daily_scanner,
revisar la tabla en consola, elegir un subconjunto de picks a mano, aplicar
1/2 Kelly) por una corrida no interactiva con los mismos criterios que se
venian aplicando a mano en las ultimas jornadas:

  - WTA se ignora por completo si el dataset esta desactualizado (mismo
    criterio de src.data.staleness que ya imprime la advertencia en
    src.daily_scanner — aqui se usa para decidir, no solo para avisar).
  - Solo se registran picks con edge >= MIN_EDGE (3%) — descarta el "ruido"
    de edges marginales que se venian declinando a mano.
  - El Kelly registrado es la mitad del calculado (1/2 Kelly), igual que las
    jornadas registradas manualmente hasta ahora.

No liquida apuestas ni decide resultados de partidos — eso es
scripts/settle_workflow.py, y sigue pidiendo confirmacion humana porque este
proyecto no tiene ninguna fuente de resultados automatizada confiable (ver
docs/metrics/2026-08-19-jornada-cincinnati-round4.md).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.config  # noqa: F401  (loads .env as a side effect, before ODDS_API_KEY is read)
from src.calibration_audit import classify_audit_decision, log_prediction_audit
from src.daily_scanner import _canonical_names, _load_models, discover_matches, evaluate_matches
from src.data.staleness import StalenessLevel, evaluate_staleness
from src.data.timezone_utils import lima_today
from src.git_utils import commit_and_push
from src.utils.notifier import send_picks_email
from src.utils.vpn_tracker import track_vpn_usage
from src.value_analysis import _SHRINK_HI, _SHRINK_LO, _SHRINK_RATE, get_last_match_date, log_query

MIN_EDGE = 0.03
KELLY_DIVISOR = 2
VALUE_BETS_LOG = "data/value_bets_log.csv"


def _wta_is_stale() -> bool:
    """True when the cached WTA dataset is stale enough to skip WTA picks
    entirely for this run. Requires _load_models(("atp","wta"), ...) to have
    already populated the WTA cache — returns False (don't skip) if there's
    no cache to check yet, letting the normal load_model() staleness print
    speak for itself instead of silently excluding WTA."""
    last_match = get_last_match_date("wta")
    if last_match is None:
        return False
    report = evaluate_staleness(last_match, lima_today())
    return report.level != StalenessLevel.OK


def _qualifying_sides(r) -> tuple[bool, bool]:
    """Which side(s) of an evaluated match pass the project's auto-logging
    gate: elo found, adequate sample, not a suspicious edge, and edge >= MIN_EDGE
    on that side specifically (has_value alone isn't enough — a 0.5% edge
    has_value=True but doesn't clear the bar this workflow applies)."""
    if not (r.elo_ok and not r.low_sample and not r.suspicious):
        return False, False
    qualifies_a = r.val_a["has_value"] and r.val_a["edge"] >= MIN_EDGE
    qualifies_b = r.val_b["has_value"] and r.val_b["edge"] >= MIN_EDGE
    return qualifies_a, qualifies_b


def run(dry_run: bool = False) -> list[dict]:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY no configurada. Abortando.")
        return []

    print("Cargando modelos (ATP, WTA)...")
    models = _load_models(("atp", "wta"), retrain=False)
    canonical_names = _canonical_names(models)

    tours = ("atp", "wta")
    if _wta_is_stale():
        print("  WTA desactualizado: se ignoran picks WTA de esta jornada.")
        tours = ("atp",)

    print("Descubriendo partidos programados...")
    matches = discover_matches(api_key, canonical_names, tours=tours)
    results = evaluate_matches(matches, models)

    # One qualification decision per match, computed once (kelly gets
    # mutated in-place right after — recomputing later would still be
    # correct since qualification only looks at has_value/edge, but keeping
    # a single decision per match avoids relying on that non-obvious fact).
    graded = [(r, *_qualifying_sides(r)) for r in results]

    selected: list[dict] = []
    for r, qualifies_a, qualifies_b in graded:
        if not (qualifies_a or qualifies_b):
            continue

        r.val_a["kelly_fraction"] = (r.val_a["kelly_fraction"] / KELLY_DIVISOR) if qualifies_a else 0.0
        r.val_b["kelly_fraction"] = (r.val_b["kelly_fraction"] / KELLY_DIVISOR) if qualifies_b else 0.0

        match_label = f"{r.match.player_a} vs {r.match.player_b}"
        for qualifies, val, odds, name, bookmaker in (
            (qualifies_a, r.val_a, r.match.odds_a, r.match.player_a, r.match.bookmaker_a),
            (qualifies_b, r.val_b, r.match.odds_b, r.match.player_b, r.match.bookmaker_b),
        ):
            if qualifies:
                selected.append({
                    "match": match_label, "pick": name, "odds": odds,
                    "edge": val["edge"], "kelly": val["kelly_fraction"],
                    "bookmaker": bookmaker,
                })

        if not dry_run:
            log_query(
                r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                r.match.player_a, r.match.player_b, r.pred, r.val_a, r.val_b,
                r.match.odds_a, r.match.odds_b, odds_a_source="auto", odds_b_source="auto",
                bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
            )

    if not dry_run:
        for r, qualifies_a, qualifies_b in graded:
            try:
                decision = classify_audit_decision(
                    r.low_sample, r.suspicious, r.elo_ok, qualifies_a or qualifies_b,
                    r.val_a["has_value"], r.val_b["has_value"],
                )
                log_prediction_audit(
                    r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                    r.match.player_a, r.match.player_b, r.pred, r.val_a, r.val_b,
                    r.match.odds_a, r.match.odds_b, odds_a_source="auto", odds_b_source="auto",
                    decision=decision, model_snapshot_id=None,
                    shrink_hi=_SHRINK_HI, shrink_lo=_SHRINK_LO, shrink_rate=_SHRINK_RATE,
                    bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
                )
            except Exception as e:
                print(f"  Aviso: no se pudo escribir en el audit log ({e}). Continuando.")

    print(f"\n{len(selected)} pick(s) seleccionados (edge >= {MIN_EDGE * 100:.0f}%, 1/2 Kelly):")
    for p in selected:
        print(f"  {p['match']} -> {p['pick']} @ {p['odds']:.2f} "
              f"(edge {p['edge'] * 100:+.1f}%, kelly {p['kelly'] * 100:.2f}%)")

    vpn_status = track_vpn_usage()
    print(f"\nVPN: {vpn_status}")

    if not dry_run:
        send_picks_email(selected, vpn_status)
        if selected:
            commit_and_push(
                [VALUE_BETS_LOG],
                f"feat(bets): auto-log {len(selected)} value bet(s) for {lima_today().isoformat()} (1/2 Kelly)",
            )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Escaneo diario automatizado: filtra, registra, notifica y pushea"
    )
    parser.add_argument("--dry-run", action="store_true",
                         help="No escribe CSVs, no envia correo, no hace git push")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

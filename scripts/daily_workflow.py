"""Daily workflow: scan for value bets, apply the project's filters
non-interactively, log the qualifying picks, notify by email, and optionally push.

Uso:
  python scripts/daily_workflow.py
  python scripts/daily_workflow.py --dry-run   # no escribe CSVs, no envia correo, no hace git push
  python scripts/daily_workflow.py --no-push   # para el runner programado

Reemplaza el flujo manual usado hasta ahora (correr src.daily_scanner,
revisar la tabla en consola, elegir un subconjunto de picks a mano, aplicar
1/2 Kelly) por una corrida no interactiva con los mismos criterios que se
venian aplicando a mano en las ultimas jornadas:

  - ATP, WTA y Davis se ignoran por completo, cada uno independientemente, si su
    dataset esta desactualizado (mismo criterio de src.data.staleness que ya
    imprime la advertencia en src.daily_scanner — aqui se usa para decidir,
    no solo para avisar). Simetria ATP/WTA agregada el 2026-08-28: antes solo
    WTA se auto-excluia; el aviso de staleness de ATP se imprimia pero no
    bloqueaba nada.
  - Solo se registran picks con edge >= MIN_EDGE (3%) — descarta el "ruido"
    de edges marginales que se venian declinando a mano.
  - El Kelly registrado es un cuarto del calculado (1/4 Kelly). Bajado desde
    1/2 Kelly el 2026-08-27 tras el diagnostico de calibracion de la gira de
    pista dura (Cincinnati+Monterrey): la banca acumulaba drawdown de -11.8%
    y el Monte Carlo de Modulo 4 mostro que 1/4 Kelly recorta el drawdown
    mediano esperado de -18.9% a -7.5% sin aumentar el riesgo de ruina (ya
    era ~0% gracias al KELLY_CAP). No se toco apply_shrinkage: el sesgo de
    sobreconfianza detectado en la racha reciente (odds 1.5-2.0, N=28) no es
    estadisticamente significativo (p=0.08) y cae justo en el rango 40-80%
    que el estudio de calibracion de referencia (Section 13, ATP+WTA 2016-23,
    muestra mucho mayor) encontro bien calibrado — ajustar el shrinkage con
    N=28 arriesgaria sobreajustar ruido de muestra chica.

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
from src import value_analysis
from scripts.backup_logs import backup_logs
from src.calibration_audit import AUDIT_LOG_PATH, classify_audit_decision, log_prediction_audit
from src.daily_scanner import _canonical_names, _load_models, discover_matches, evaluate_matches
from src.data.staleness import StalenessLevel, evaluate_staleness
from src.data.timezone_utils import lima_today
from src.git_utils import commit_and_push
from src.utils.notifier import send_picks_email
from src.utils.vpn_tracker import track_vpn_usage
from src.value_analysis import (
    _SHRINK_HI, _SHRINK_LO, _SHRINK_RATE, LogMatchStatus,
    check_existing_log_entry, get_last_match_date, log_query, update_log_entry,
)

MIN_EDGE = 0.03
KELLY_DIVISOR = 4
# Tours still scanned and written to the prediction audit log, but never
# auto-logged as bets or emailed. Both excluded since 2026-10-07 after
# replaying this exact rule on out-of-sample predictions against historical
# market odds (docs/metrics/2026-10-07-model-vs-market-edge.md): WTA lost
# -12.3% flat over 9,038 bets, worse than betting every side; ATP lost -5.0%
# at average odds over 13,645 bets, positive only at the best available odds,
# with what edge there was eroding since 2018. Davis Cup has no odds source
# to test against. Remove a tour only with new evidence of edge from
# scripts/market_edge_backtest.py.
AUTO_LOG_EXCLUDED_TOURS = frozenset({"wta", "atp"})
VALUE_BETS_LOG = "data/value_bets_log.csv"


def _is_stale(tour: str, live_tournament_mode: bool = False) -> bool:
    """True when `tour`'s cached dataset is stale enough that its picks
    should be excluded from this run. Requires _load_models(...) to have
    already populated the cache for `tour` — returns False (don't skip) if
    there's no cache to check yet, letting the normal load_model()
    staleness print speak for itself instead of silently excluding it.

    live_tournament_mode=True matches src/daily_scanner.py's behavior for a
    confirmed-live tournament: the staleness level is still computed and
    printed as a warning, but never excludes the tour — a few days of
    archive publication lag during an actually-live tournament week is a
    known, accepted limitation of the data source, not a sign the whole
    dataset is stale/abandoned. Default (False) preserves the original
    hard-exclude-on-any-non-OK gate, which exists specifically to stop
    auto-pushed, auto-emailed picks from being generated off data that
    could be weeks out of date with nobody reviewing it (see
    docs/superpowers/specs/2026-07-13-staleness-context-aware-design.md).
    """
    last_match = get_last_match_date(tour)
    if last_match is None:
        return False
    report = evaluate_staleness(last_match, lima_today(), live_tournament_mode)
    if report.level == StalenessLevel.OK:
        return False
    print(f"  {tour.upper()}: {report.message}")
    return not live_tournament_mode


def _wta_is_stale(live_tournament_mode: bool = False) -> bool:
    """True when the cached WTA dataset is stale enough to skip WTA picks
    entirely for this run — see _is_stale()'s docstring for the
    live_tournament_mode semantics."""
    return _is_stale("wta", live_tournament_mode)


def _atp_is_stale(live_tournament_mode: bool = False) -> bool:
    """True when the cached ATP dataset is stale enough to skip ATP picks
    entirely for this run. Symmetric to _wta_is_stale() — see
    _is_stale()'s docstring."""
    return _is_stale("atp", live_tournament_mode)


def _davis_is_stale(live_tournament_mode: bool = False) -> bool:
    """True when the cached Davis Cup dataset is stale enough to skip Davis
    Cup picks for this run. Symmetric to _wta_is_stale()/_atp_is_stale() —
    Davis Cup has no separate raw source (see src/data/loader.py's
    load_davis_cup_matches), so this reads the same
    data/model_cache/davis.pkl via get_last_match_date("davis")."""
    return _is_stale("davis", live_tournament_mode)


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


def run(
    dry_run: bool = False, retrain: bool = False, live_tournament_mode: bool = False,
    tour: str = "both", days_ahead: int = 1, push: bool = True,
) -> list[dict]:
    if days_ahead < 1:
        raise ValueError("days_ahead debe ser al menos 1")
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY no configurada. Abortando.")
        return []

    tours = ("atp", "wta", "davis") if tour == "both" else (tour,)
    print(f"Cargando modelos ({', '.join(t.upper() for t in tours)})...")
    models = _load_models(tours, retrain=retrain)
    canonical_names = _canonical_names(models)

    if "atp" in tours and _atp_is_stale(live_tournament_mode):
        print("  ATP desactualizado: se ignoran picks ATP de esta jornada.")
        tours = tuple(t for t in tours if t != "atp")
    if "wta" in tours and _wta_is_stale(live_tournament_mode):
        print("  WTA desactualizado: se ignoran picks WTA de esta jornada.")
        tours = tuple(t for t in tours if t != "wta")
    if "davis" in tours and _davis_is_stale(live_tournament_mode):
        print("  Davis Cup desactualizado: se ignoran picks Davis de esta jornada.")
        tours = tuple(t for t in tours if t != "davis")

    print("Descubriendo partidos programados...")
    matches = discover_matches(api_key, canonical_names, tours=tours, days_ahead=days_ahead) if tours else []
    results = evaluate_matches(matches, models)

    # One qualification decision per match, computed once (kelly gets
    # mutated in-place right after — recomputing later would still be
    # correct since qualification only looks at has_value/edge, but keeping
    # a single decision per match avoids relying on that non-obvious fact).
    graded = [(r, *_qualifying_sides(r)) for r in results]

    selected: list[dict] = []
    logged_results: set[int] = set()
    if results and not dry_run:
        backup_logs(files=(AUDIT_LOG_PATH, value_analysis.LOG_PATH), backup_dir=_ROOT / "data" / "logs")
    for r, qualifies_a, qualifies_b in graded:
        if not (qualifies_a or qualifies_b):
            continue
        if r.match.tour in AUTO_LOG_EXCLUDED_TOURS:
            print(f"  {r.match.player_a} vs {r.match.player_b}: pick {r.match.tour.upper()} no registrado "
                  f"(tour excluido del auto-registro: sin edge demostrado frente al mercado).")
            continue

        status = LogMatchStatus.NEW
        if not dry_run:
            status, _existing = check_existing_log_entry(
                r.match.tour, r.match.tournament, r.match.player_a, r.match.player_b,
                r.match.match_date, r.match.odds_a, r.match.odds_b,
            )
            if status in (LogMatchStatus.DUPLICATE, LogMatchStatus.RESOLVED):
                print(f"  {r.match.player_a} vs {r.match.player_b}: ya registrado "
                      f"({status.value}); no se vuelve a guardar.")
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
            save = update_log_entry if status == LogMatchStatus.UPDATE else log_query
            saved = save(
                r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                r.match.player_a, r.match.player_b, r.pred, r.val_a, r.val_b,
                r.match.odds_a, r.match.odds_b, odds_a_source="auto", odds_b_source="auto",
                bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
            )
            if status == LogMatchStatus.UPDATE and not saved:
                raise RuntimeError("El partido ya no existe en el log; se cancela la actualización.")
            logged_results.add(id(r))

    if not dry_run:
        for r, qualifies_a, qualifies_b in graded:
            try:
                decision = classify_audit_decision(
                    r.low_sample, r.suspicious, r.elo_ok, id(r) in logged_results,
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

    print(f"\n{len(selected)} pick(s) seleccionados (edge >= {MIN_EDGE * 100:.0f}%, 1/4 Kelly):")
    for p in selected:
        print(f"  {p['match']} -> {p['pick']} @ {p['odds']:.2f} "
              f"(edge {p['edge'] * 100:+.1f}%, kelly {p['kelly'] * 100:.2f}%)")

    vpn_status = track_vpn_usage()
    print(f"\nVPN: {vpn_status}")

    if not dry_run:
        send_picks_email(selected, vpn_status)
        if selected and push:
            commit_and_push(
                [VALUE_BETS_LOG],
                f"feat(bets): auto-log {len(selected)} value bet(s) for {lima_today().isoformat()} (1/4 Kelly)",
            )

    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Escaneo diario automatizado: filtra, registra y notifica"
    )
    parser.add_argument("--dry-run", action="store_true",
                         help="No escribe CSVs, no envia correo, no hace git push")
    parser.add_argument("--retrain", action="store_true",
                         help="Reconstruye el cache del modelo desde data/processed/ en vez de reusar el existente")
    parser.add_argument("--tour", choices=["atp", "wta", "davis", "both"], default="both")
    parser.add_argument("--days-ahead", type=int, default=1)
    parser.add_argument("--no-push", action="store_true",
                        help="Guarda y notifica sin crear commit ni hacer push")
    parser.add_argument("--live-tournament", action="store_true", dest="live_tournament_mode",
                         help="Torneo activo confirmado: no excluir un tour por staleness, solo advertir "
                              "(igual que src/daily_scanner.py) -- el gate estricto por defecto sigue activo sin esta flag")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.days_ahead < 1:
        parser.error("--days-ahead debe ser al menos 1")
    run(dry_run=args.dry_run, retrain=args.retrain, live_tournament_mode=args.live_tournament_mode,
        tour=args.tour, days_ahead=args.days_ahead, push=not args.no_push)


if __name__ == "__main__":
    main()

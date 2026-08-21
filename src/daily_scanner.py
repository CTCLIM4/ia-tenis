"""Módulo 1: Automatización y Descubrimiento de Partidos del Día.

Escanea los torneos ATP/WTA activos vía The Odds API, cruza los partidos
con el dataset interno de jugadores/Elo, corre el pipeline de inferencia en
lote (el mismo predict_match/calculate_value que usa la CLI interactiva de
src/value_analysis.py) y muestra solo las apuestas con valor (+EV y
Kelly > 0).

Uso:
  python -m src.daily_scanner                  # ATP + WTA, ventana de 1 dia
  python -m src.daily_scanner --tour atp        # solo ATP
  python -m src.daily_scanner --days-ahead 2    # incluye partidos hasta 2 dias vista

Requiere ODDS_API_KEY en el entorno (o en un archivo .env en la raiz del
repo, ver .env.example — cargado automaticamente por src.config) — a
diferencia de la CLI interactiva, este modulo no tiene fallback manual (no
hay humano tipeando cuotas para cada partido descubierto).

Descubrimiento: The Odds API publica el tenis con un sport_key por torneo
activo (ej. 'tennis_atp_wimbledon'), no un key generico por tour. Este
modulo llama primero a /v4/sports para listar los torneos de tenis activos
(src.odds_api.list_tennis_sport_keys), y luego pide las cuotas de cada uno
por separado. Este supuesto sobre el formato de sport_key aun no se ha
verificado contra una respuesta real de la API (pendiente de una
ODDS_API_KEY real) — ver docs/superpowers/specs/2026-07-07-odds-api-integration-design.md.

Superficie: resuelta via src.surface_resolver.resolve_surface a partir del
titulo/key del torneo. Partidos sin superficie resoluble se excluyen.

Emparejamiento de jugadores: los nombres que reporta The Odds API pueden
diferir en formato de los nombres canonicos del dataset ("A. Zverev" vs
"Alexander Zverev") — resuelto via src.player_matcher.match_player_name.
Partidos donde no se puede emparejar alguno de los dos jugadores se
excluyen (nunca se adivina).

Registro: al final se pregunta una sola vez si se quieren guardar las
value bets encontradas en data/value_bets_log.csv (igual que la CLI
interactiva, opt-in). El audit log (data/prediction_audit_log.csv) se
escribe SIEMPRE, para cada partido evaluado, independientemente de esa
respuesta — igual que hace interactive_cli — porque existe justamente para
capturar la poblacion completa de predicciones sin el sesgo de seleccion
humana (ver src/calibration_audit.py).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from src.calibration_audit import classify_audit_decision, log_prediction_audit
from src.config import MAX_SUSPICIOUS_EDGE, MIN_MATCHES_THRESHOLD  # also loads .env (side effect) before ODDS_API_KEY is read below
from src.data.timezone_utils import to_lima
from src.odds_api import (
    DEFAULT_BOOKMAKER,
    _best_price,
    fetch_odds_events_by_key,
    fetch_sports_index,
    list_tennis_sport_keys,
    resolve_allowed_bookmakers,
)
from src.player_matcher import match_player_name
from src.surface_resolver import resolve_surface
from src.value_analysis import (
    _SHRINK_HI,
    _SHRINK_LO,
    _SHRINK_RATE,
    calculate_value,
    load_model,
    log_query,
    predict_match,
)

DEFAULT_DAYS_AHEAD = 1


@dataclass
class DiscoveredMatch:
    tour: str
    tournament: str
    surface: str
    match_date: date
    player_a: str   # canonical dataset name (post player_matcher resolution)
    player_b: str
    odds_a: float
    odds_b: float
    raw_home: str   # name as reported by The Odds API, for display/debugging
    raw_away: str
    bookmaker_a: str = ""
    bookmaker_b: str = ""


@dataclass
class EvaluatedMatch:
    match: DiscoveredMatch
    pred: dict
    val_a: dict
    val_b: dict
    elo_ok: bool
    suspicious: bool
    low_sample: bool = False

    @property
    def has_value(self) -> bool:
        return self.val_a["has_value"] or self.val_b["has_value"]

    @property
    def is_value_bet(self) -> bool:
        return self.elo_ok and not self.low_sample and not self.suspicious and self.has_value


def _now() -> datetime:
    return datetime.now(timezone.utc)


# _now()/_is_within_window compare timezone-aware instants, not calendar
# dates — that comparison is correct regardless of which tz the operands
# are expressed in (Python normalizes internally), so no Lima conversion
# is needed here. Only the calendar-date extraction below needs it.
def _is_within_window(commence_time: str, days_ahead: int, now: Optional[datetime] = None) -> bool:
    """True when commence_time (ISO8601, e.g. '2026-07-27T18:00:00Z') falls
    between now and now + days_ahead days — the "today/next matchday" window."""
    now = now or _now()
    ts = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    return now <= ts <= now + timedelta(days=days_ahead)


def _extract_h2h_odds(
    event: dict, allowed_bookmakers: Optional[set[str]]
) -> Optional[tuple[float, float, str, str]]:
    """Return (odds_home, odds_away, bookmaker_home, bookmaker_away) from the
    best-priced allowed bookmaker's h2h market, or None if no allowed
    bookmaker quoted both sides of this event."""
    odds_home, odds_away, bk_home, bk_away = _best_price(event, allowed_bookmakers)
    if odds_home is None or odds_away is None:
        return None
    return odds_home, odds_away, bk_home, bk_away


def discover_matches(
    api_key: str,
    canonical_names_by_tour: dict[str, list[str]],
    bookmaker: Optional[str] = None,
    days_ahead: int = DEFAULT_DAYS_AHEAD,
    tours: tuple[str, ...] = ("atp", "wta"),
) -> list[DiscoveredMatch]:
    """Discover today's/next-matchday's matches with bettable odds, matched
    to canonical dataset player names.

    `bookmaker` is a backward-compatible override (ODDS_API_BOOKMAKER env
    var or --bookmaker CLI flag) that collapses best-price selection to a
    single fixed bookmaker — see resolve_allowed_bookmakers.

    Skips (with a printed reason) any event where the tournament's surface
    can't be resolved, either player can't be matched to the dataset, or no
    allowed bookmaker quoted h2h odds for both sides — never raises on a
    per-event problem, since one bad event must not abort the whole scan.
    """
    allowed_bookmakers = resolve_allowed_bookmakers(bookmaker)
    sports_index = fetch_sports_index(api_key)
    tennis_sports = [s for s in list_tennis_sport_keys(sports_index) if s["tour"] in tours]

    matches: list[DiscoveredMatch] = []
    for sport in tennis_sports:
        surface = resolve_surface(sport["title"], sport["key"])
        if surface is None:
            continue  # resolve_surface already printed the warning

        events = fetch_odds_events_by_key(sport["key"], api_key)
        for event in events:
            commence_time = event.get("commence_time")
            if not commence_time or not _is_within_window(commence_time, days_ahead):
                continue

            odds = _extract_h2h_odds(event, allowed_bookmakers)
            if odds is None:
                continue
            odds_home, odds_away, bookmaker_home, bookmaker_away = odds

            home, away = event.get("home_team", ""), event.get("away_team", "")
            canonical = canonical_names_by_tour.get(sport["tour"], [])
            player_a = match_player_name(home, canonical)
            player_b = match_player_name(away, canonical)
            if player_a is None or player_b is None:
                unmatched = [n for n, p in ((home, player_a), (away, player_b)) if p is None]
                print(f"  Aviso: no se pudo emparejar jugador(es) {', '.join(unmatched)} "
                      f"({sport['title']}). Partido excluido.")
                continue

            match_date = to_lima(datetime.fromisoformat(commence_time.replace("Z", "+00:00"))).date()
            matches.append(DiscoveredMatch(
                tour=sport["tour"], tournament=sport["title"], surface=surface,
                match_date=match_date, player_a=player_a, player_b=player_b,
                odds_a=odds_home, odds_b=odds_away,
                raw_home=home, raw_away=away,
                bookmaker_a=bookmaker_home, bookmaker_b=bookmaker_away,
            ))

    return matches


def evaluate_matches(
    matches: list[DiscoveredMatch],
    models: dict[str, tuple],
    halt_on_suspicious: bool = True,
) -> list[EvaluatedMatch]:
    """Run batch inference over discovered matches, reusing the same
    predict_match/calculate_value pipeline as the interactive CLI.

    Returns one EvaluatedMatch per input match, evaluated regardless of
    quality (missing Elo, suspicious edge, no value) — callers filter via
    `.is_value_bet` for display/logging. This keeps the full population
    available for the audit log, matching interactive_cli's semantics.
    """
    results = []
    for m in matches:
        elo, fb, clf, rank_lookup, elo_tracker, age_lookup = models[m.tour]
        pred = predict_match(
            elo, fb, clf, m.player_a, m.player_b, m.surface, m.match_date,
            rank_lookup=rank_lookup, elo_tracker=elo_tracker, age_lookup=age_lookup,
        )
        val_a = calculate_value(pred["p_a_cal"], m.odds_a)
        val_b = calculate_value(pred["p_b_cal"], m.odds_b)
        elo_ok = pred.get("elo_found_a", True) and pred.get("elo_found_b", True)
        suspicious = halt_on_suspicious and max(val_a["edge"], val_b["edge"]) > MAX_SUSPICIOUS_EDGE
        low_sample = (
            pred.get("matches_a", MIN_MATCHES_THRESHOLD) < MIN_MATCHES_THRESHOLD
            or pred.get("matches_b", MIN_MATCHES_THRESHOLD) < MIN_MATCHES_THRESHOLD
        )
        results.append(EvaluatedMatch(m, pred, val_a, val_b, elo_ok, suspicious, low_sample))
    return results


def print_value_bets_table(value_bets: list[EvaluatedMatch]) -> None:
    if not value_bets:
        print("\n  No se encontraron value bets (+EV, Kelly>0) en el escaneo actual.")
        return

    print(f"\n  {'Torneo':<18} {'Partido':<38} {'Lado':<20} {'Cuota':>6} {'Edge':>7} {'EV':>7} {'Kelly':>6}")
    print("  " + "-" * 104)
    for r in value_bets:
        match_label = f"{r.match.player_a} vs {r.match.player_b}"
        for val, odds, name in (
            (r.val_a, r.match.odds_a, r.match.player_a),
            (r.val_b, r.match.odds_b, r.match.player_b),
        ):
            if not val["has_value"]:
                continue
            print(
                f"  {r.match.tournament:<18.18} {match_label:<38.38} {name:<20.20} "
                f"{odds:>6.2f} {val['edge']*100:>+6.1f}% {val['ev']:>+7.3f} "
                f"{val['kelly_fraction']*100:>5.1f}%"
            )


def _load_models(tours: tuple[str, ...], retrain: bool) -> dict[str, tuple]:
    return {tour: load_model(tour, retrain=retrain) for tour in tours}


def _canonical_names(models: dict[str, tuple]) -> dict[str, list[str]]:
    return {tour: list(model[0].general_ratings.keys()) for tour, model in models.items()}


def run_scan(
    tours: tuple[str, ...] = ("atp", "wta"),
    bookmaker: str = DEFAULT_BOOKMAKER,
    days_ahead: int = DEFAULT_DAYS_AHEAD,
    halt_on_suspicious: bool = True,
    retrain: bool = False,
) -> None:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY no configurada. El escaneo automatico requiere una key de "
              "The Odds API (no hay fallback manual en este modo). "
              "Define la variable de entorno e intenta de nuevo.")
        return

    print(f"Cargando modelos ({', '.join(t.upper() for t in tours)})...")
    models = _load_models(tours, retrain)
    canonical_names = _canonical_names(models)

    print("Descubriendo torneos activos y partidos programados...")
    matches = discover_matches(api_key, canonical_names, bookmaker, days_ahead, tours)
    if not matches:
        print("No se encontraron partidos programados en la ventana configurada.")
        return
    print(f"{len(matches)} partido(s) descubiertos. Evaluando...")

    results = evaluate_matches(matches, models, halt_on_suspicious)
    value_bets = [r for r in results if r.is_value_bet]

    print_value_bets_table(value_bets)

    save_all = False
    if value_bets:
        answer = input(
            f"\n  Guardar estas {len(value_bets)} value bet(s) en value_bets_log.csv? (s/n) [s]: "
        ).strip().lower()
        save_all = answer in ("s", "si", "y", "yes", "")

    for r in results:
        logged = save_all and r.is_value_bet
        if logged:
            log_query(
                r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                r.match.player_a, r.match.player_b,
                r.pred, r.val_a, r.val_b, r.match.odds_a, r.match.odds_b,
                odds_a_source="auto", odds_b_source="auto",
                bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
            )
        try:
            decision = classify_audit_decision(
                r.low_sample, r.suspicious, r.elo_ok, logged,
                r.val_a["has_value"], r.val_b["has_value"],
            )
            log_prediction_audit(
                r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                r.match.player_a, r.match.player_b,
                r.pred, r.val_a, r.val_b, r.match.odds_a, r.match.odds_b,
                odds_a_source="auto", odds_b_source="auto",
                decision=decision, model_snapshot_id=None,
                shrink_hi=_SHRINK_HI, shrink_lo=_SHRINK_LO, shrink_rate=_SHRINK_RATE,
                bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
            )
        except Exception as e:
            print(f"  Aviso: no se pudo escribir en el audit log ({e}). Continuando.")

    print("\nEscaneo completo.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Escaneo automatico de partidos del dia con value bets (+EV, Kelly>0)"
    )
    parser.add_argument("--tour", choices=["atp", "wta", "both"], default="both")
    parser.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER)
    parser.add_argument(
        "--days-ahead", type=int, default=DEFAULT_DAYS_AHEAD,
        help="Ventana de partidos a incluir, en dias desde ahora (default: 1)",
    )
    parser.add_argument(
        "--no-halt-on-suspicious", dest="halt_on_suspicious", action="store_false",
        help=f"No excluir edges sospechosos (>{MAX_SUSPICIOUS_EDGE*100:.0f}%%) — "
             "por defecto SI se excluyen, a diferencia de la CLI interactiva.",
    )
    parser.add_argument("--retrain", action="store_true")
    parser.set_defaults(halt_on_suspicious=True)
    args = parser.parse_args()

    tours = ("atp", "wta") if args.tour == "both" else (args.tour,)
    run_scan(
        tours=tours, bookmaker=args.bookmaker, days_ahead=args.days_ahead,
        halt_on_suspicious=args.halt_on_suspicious, retrain=args.retrain,
    )


if __name__ == "__main__":
    main()

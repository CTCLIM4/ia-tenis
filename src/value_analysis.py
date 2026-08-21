"""
Tennis value-bet analysis — auto-fetched odds with manual fallback.

Calcula Edge / EV / Kelly para un partido dado, usando el modelo
LR+features entrenado sobre el histórico completo de ATP o WTA.

Uso:
  python -m src.value_analysis          # ATP (default)
  python -m src.value_analysis --wta    # WTA
  python -m src.value_analysis --retrain # forzar reentrenamiento

El modelo se reconstruye desde cero la primera vez (~30-60 s ATP),
y luego queda cacheado 7 días en data/model_cache/{tour}.pkl.

Cuotas: si ODDS_API_KEY esta configurada en el entorno, se intenta
autocompletar la cuota de cada jugador via The Odds API (bookmaker fijo,
default "pinnacle" — configurable con ODDS_API_BOOKMAKER, cache de eventos
configurable con ODDS_API_CACHE_MINUTES). Si no hay key, no hay match, o
falla la llamada, se pide la cuota a mano igual que antes — el auto-fetch
nunca bloquea el flujo.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MAX_SUSPICIOUS_EDGE, MIN_MATCHES_THRESHOLD  # also loads .env (side effect) before ODDS_API_KEY is read below
from src.backtest.walkforward import _MIRROR_FLIP_COLS, load_features_with_mirror
from src.calibration_audit import classify_audit_decision, log_prediction_audit
from src.data.snapshots import load_snapshot_metadata, resolve_snapshot_path
from src.data.staleness import StalenessLevel, evaluate_staleness
from src.data.timezone_utils import lima_today
from src.features import FEATURE_COLS
from src.features.decay import EloHistoryTracker, calculate_decay_features
from src.odds_api import DEFAULT_BOOKMAKER, DEFAULT_CACHE_MINUTES, MatchOdds, get_match_odds

# ── paths ────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
_DATA_DIR  = _ROOT / "data"
_CACHE_DIR = _DATA_DIR / "model_cache"
LOG_PATH   = _DATA_DIR / "value_bets_log.csv"

# ── model constants ──────────────────────────────────────────────────────────
KELLY_CAP         = 0.05   # max 5% of bankroll (conservative)
CACHE_MAX_AGE_DAYS = 7     # rebuild if cache older than this

_odds_warned = False       # print the auto-fetch failure warning once per session

# ── calibration shrinkage (Section 13: reliability analysis, ATP+WTA 2016-23) ─
# LR with 7 features shows overconfidence at p > 0.90 (~7pp bias) and
# underconfidence at p < 0.10 (~7pp bias). Compress 60% of the excess
# toward the boundary before computing edge/EV/Kelly.
_SHRINK_HI   = 0.90
_SHRINK_LO   = 0.10
_SHRINK_RATE = 0.60


# ── core value functions ──────────────────────────────────────────────────────

def apply_shrinkage(p: float) -> float:
    """Calibration correction for extreme probabilities.

    Shrinks predictions outside [0.10, 0.90] toward the boundary.
    The 40-80% range is well-calibrated (|error| < 3pp) — unchanged.
    """
    if p > _SHRINK_HI:
        return _SHRINK_HI + (p - _SHRINK_HI) * (1 - _SHRINK_RATE)
    if p < _SHRINK_LO:
        return _SHRINK_LO - (_SHRINK_LO - p) * (1 - _SHRINK_RATE)
    return p


def calculate_value(model_prob: float, odds_decimal: float) -> dict:
    """Compute betting metrics for one side of a match.

    Args:
        model_prob:   calibrated win probability from the LR model
        odds_decimal: bookmaker decimal odds (e.g. 1.85)

    Returns dict with:
        implied_prob   – bookmaker's implied probability (1/odds)
        edge           – model_prob minus implied_prob
        ev             – expected value per unit staked (positive = value bet)
        kelly_fraction – conservative Kelly stake (capped at KELLY_CAP)
        has_value      – True when edge > 0
    """
    if odds_decimal <= 1.0:
        raise ValueError(f"Cuota invalida: {odds_decimal}. Debe ser > 1.")
    if not (0 < model_prob < 1):
        raise ValueError(f"Probabilidad invalida: {model_prob}. Debe estar en (0, 1).")

    implied_prob = 1.0 / odds_decimal
    edge         = model_prob - implied_prob
    ev           = model_prob * (odds_decimal - 1) - (1 - model_prob)
    raw_kelly    = edge / (odds_decimal - 1) if edge > 0 else 0.0
    kelly        = min(raw_kelly, KELLY_CAP)

    return {
        "implied_prob":   implied_prob,
        "edge":           edge,
        "ev":             ev,
        "kelly_fraction": kelly,
        "has_value":      edge > 0,
    }


def _should_halt_on_suspicious_edge(val_a: dict, val_b: dict, halt_on_suspicious: bool) -> bool:
    """True when --halt-on-suspicious is active and either side's edge exceeds
    MAX_SUSPICIOUS_EDGE — a signal the prediction may be based on stale
    data or a name-matching error rather than genuine market inefficiency."""
    return halt_on_suspicious and max(val_a["edge"], val_b["edge"]) > MAX_SUSPICIOUS_EDGE


def _should_log_prediction(suspicious: bool, save_response: str) -> bool:
    """True when the CLI should persist a prediction via log_query().

    Suspicious edges are always blocked, regardless of the answer. A
    missing-Elo prediction is NOT blocked here — log_query() itself
    downgrades it to status='invalid_missing_elo', preserving traceability
    instead of discarding the row outright. (Previously the CLI had a
    dedicated `elif not elo_ok` branch that printed a message and never
    called log_query at all, making that documented status unreachable.)
    """
    if suspicious:
        return False
    return save_response.strip().lower() in ("s", "si", "y", "yes", "")


# ── rank lookup ───────────────────────────────────────────────────────────────

def _build_rank_lookup(df_raw: pd.DataFrame) -> Dict[str, int]:
    """Extract most-recent ranking for every player from raw match data.

    Iterates matches in chronological order; the last write per player
    wins, so the result is each player's rank as of their most recent match.
    Both winner and loser perspectives are used.

    Works for both ATP (winner_name/winner_rank) and WTA after _clean_wta
    normalises the column names to the same schema.
    """
    lookup: Dict[str, int] = {}
    for _, row in df_raw.iterrows():
        w_rank = row.get("winner_rank")
        l_rank = row.get("loser_rank")
        if pd.notna(w_rank) and w_rank > 0:
            lookup[row["winner_name"]] = int(w_rank)
        if pd.notna(l_rank) and l_rank > 0:
            lookup[row["loser_name"]] = int(l_rank)
    return lookup


# ── age lookup (ATP only — WTA's tennis-data.co.uk source has no age column) ─

def _build_age_lookup(df_raw: pd.DataFrame) -> Dict[str, Tuple[float, date]]:
    """Extract each player's most recently observed age + the date it was
    observed, so a later prediction can extrapolate their current age.

    Returns {} for tours without an age column (WTA) — callers must treat a
    missing entry as "age unknown", not an error; age_multiplier() already
    treats None as neutral (1.0).
    """
    lookup: Dict[str, Tuple[float, date]] = {}
    if "winner_age" not in df_raw.columns or "loser_age" not in df_raw.columns:
        return lookup
    for _, row in df_raw.iterrows():
        match_date = row["match_date"]
        if hasattr(match_date, "date"):
            match_date = match_date.date()
        w_age = row.get("winner_age")
        l_age = row.get("loser_age")
        if pd.notna(w_age):
            lookup[row["winner_name"]] = (float(w_age), match_date)
        if pd.notna(l_age):
            lookup[row["loser_name"]] = (float(l_age), match_date)
    return lookup


def _current_age(
    player: str, age_lookup: Dict[str, Tuple[float, date]], as_of_date: date,
) -> Optional[float]:
    """Extrapolate a player's age forward from their last observed match."""
    entry = age_lookup.get(player)
    if entry is None:
        return None
    observed_age, observed_date = entry
    return observed_age + (as_of_date - observed_date).days / 365.25


def _to_wta_key(full_name: str) -> str:
    """Convert 'Aryna Sabalenka' → 'Sabalenka A.' for WTA lookup."""
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]} {parts[0][0]}."
    return full_name


def _resolve_player_name(player: str, elo) -> str:
    """Return the internal EloSystem/FeatureBuilder key for a player.

    WTA data uses abbreviated names like 'Sabalenka A.' or 'Pliskova Ka.'
    Users type full names like 'Aryna Sabalenka' or 'Karolina Pliskova'.

    When siblings share a surname (e.g. Karolina vs Kristyna Pliskova),
    tennis-data.co.uk disambiguates with longer prefixes ('Ka.' vs 'Kr.',
    and sometimes legacy variants like 'Kar.' with only 3 entries).

    Among all 'Lastname X.' keys where first_name.startswith(X), we pick
    the one with the MOST career matches — the primary/canonical key for
    that player, not a legacy three-match disambiguation artifact.
    """
    if player in elo.general_ratings:
        return player

    parts = player.strip().split()
    if len(parts) < 2:
        return player

    first, last = parts[0], parts[-1]

    # Collect all 'Lastname X.' candidates where first_name starts with X,
    # ranked by career match count (most matches = canonical key).
    best_key, best_count = None, -1
    for key in elo.general_ratings:
        key_parts = key.split()
        if len(key_parts) != 2 or key_parts[0] != last:
            continue
        abbrev = key_parts[1].rstrip(".")
        if not first.lower().startswith(abbrev.lower()):
            continue
        count = elo.match_counts.get(key, 0)
        if count > best_count:
            best_key, best_count = key, count

    return best_key if best_key is not None else player


def _is_elo_known(player: str, elo) -> bool:
    """True when the player has a trained Elo rating (not the 1500 default)."""
    resolved = _resolve_player_name(player, elo)
    return resolved in elo.general_ratings


def _elo_candidates(player: str, elo, fb) -> list:
    """Return all valid 'Lastname X.' candidates for a typed name.

    Each element: (key, match_count, year_span_str), sorted by match_count desc.
    Returns [] when the name matches exactly (no ambiguity possible) or is unknown.
    Returns 1 element when unambiguous after WTA translation.
    Returns 2+ elements when multiple abbreviations match the first name — the
    caller should warn the user and confirm which one is being used.
    """
    if player in elo.general_ratings:
        return []  # exact key, nothing to resolve

    parts = player.strip().split()
    if len(parts) < 2:
        return []

    first, last = parts[0], parts[-1]
    candidates = []
    for key in elo.general_ratings:
        key_parts = key.split()
        if len(key_parts) != 2 or key_parts[0] != last:
            continue
        abbrev = key_parts[1].rstrip(".")
        if not first.lower().startswith(abbrev.lower()):
            continue
        count = elo.match_counts.get(key, 0)
        hist  = fb._history.get(key, [])
        if hist:
            years = [e[0].year for e in hist]
            span  = f"{min(years)}-{max(years)}"
        else:
            span = "sin historial"
        candidates.append((key, count, span))

    candidates.sort(key=lambda x: -x[1])
    return candidates


def lookup_rank(player: str, rank_lookup: Dict[str, int]) -> Optional[int]:
    """Return most-recent known rank for player, or None if unknown.

    Tries two formats:
      1. Direct match — works for ATP (full names) and any exact WTA key.
      2. WTA 'Lastname I.' — converts 'Aryna Sabalenka' → 'Sabalenka A.'
         to match tennis-data.co.uk WTA encoding.
    """
    if player in rank_lookup:
        return rank_lookup[player]
    wta_key = _to_wta_key(player)
    return rank_lookup.get(wta_key)


# ── cache ─────────────────────────────────────────────────────────────────────

def _cache_path(tour: str, snapshot: Optional[str] = None) -> Path:
    if snapshot is not None:
        return _CACHE_DIR / f"{tour}__snapshot-{snapshot}.pkl"
    return _CACHE_DIR / f"{tour}.pkl"


def _check_staleness(
    tour: str,
    last_match_date: date,
    live_tournament_mode: bool = False,
    reference_date: Optional[date] = None,
) -> None:
    """Warn when the dataset's newest match is stale, using context-aware
    thresholds (see src/data/staleness.py) instead of one flat number:
    lenient during the Dec-early-Jan off-season, strict in --live-tournament
    scenarios, and a tighter default (3d warning / 7d critical) otherwise
    than the old flat 30-day rule — that flat rule missed a dataset that was
    functionally stale mid-Slam at only 16 days old (see
    docs/superpowers/specs/2026-07-13-staleness-context-aware-design.md).

    Runs on every load_model() call that isn't pinned to a snapshot,
    regardless of cache hit/miss, so the warning reflects true data age (how
    recent is the underlying match data) rather than cache age (how old is
    the pickle file) — those are different things and a week-old cache can
    still wrap multi-month-old match data. load_model(snapshot=...) skips
    this entirely — a deliberately old, pinned dataset isn't "stale."

    last_match_date must be a plain date (not datetime/Timestamp) — callers
    deriving this from a pandas column should call .date() first.
    """
    report = evaluate_staleness(
        last_match_date,
        reference_date or lima_today(),
        live_tournament_mode,
    )
    if report.level == StalenessLevel.OK:
        return
    tag = "ADVERTENCIA" if report.level == StalenessLevel.WARNING else "CRITICO"
    print(f"\n  *** {tag}: dataset {tour.upper()} desactualizado (regla: {report.rule}) ***")
    print(f"  *** Ultimo partido en los datos: {last_match_date} ({report.days_stale} dias atras).")
    print("  *** Las predicciones no incorporan resultados posteriores a esa fecha.")


def get_last_match_date(tour: str) -> Optional[date]:
    """Last match date backing the currently cached model for `tour`, or None
    if no cache exists yet — call load_model(tour) at least once first.

    Lets callers (e.g. scripts/daily_workflow.py, deciding whether to skip a
    stale WTA scan) check dataset staleness themselves via
    src.data.staleness.evaluate_staleness without duplicating _check_staleness's
    print-only logic here.
    """
    cached = _load_cache(tour)
    return cached[4] if cached is not None else None


def _save_cache(
    tour: str, elo, fb, clf, rank_lookup: dict, last_match_date: date,
    elo_tracker: EloHistoryTracker, age_lookup: dict, snapshot: Optional[str] = None,
) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "elo":              elo,
        "fb":               fb,
        "clf":              clf,
        "rank_lookup":      rank_lookup,
        "timestamp":        datetime.now(),
        "tour":             tour,
        "last_match_date":  last_match_date,
        "elo_tracker":      elo_tracker,
        "age_lookup":       age_lookup,
    }
    path = _cache_path(tour, snapshot)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache guardado en {path}")


def _load_cache(tour: str, snapshot: Optional[str] = None):
    """Return (elo, fb, clf, rank_lookup, last_match_date, elo_tracker,
    age_lookup) from cache, or None if stale/missing.

    When snapshot is set, the cache never expires (CACHE_MAX_AGE_DAYS is
    ignored) — a snapshot-pinned cache reflects an immutable dataset, so
    there's nothing for it to go stale relative to.
    """
    path = _cache_path(tour, snapshot)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        print(f"  Advertencia: no se pudo leer cache ({e}). Reentrenando...")
        return None

    if "last_match_date" not in payload:
        print("  Cache de formato antiguo (sin last_match_date). Reentrenando...")
        return None

    if snapshot is None:
        age = datetime.now() - payload["timestamp"]
        if age >= timedelta(days=CACHE_MAX_AGE_DAYS):
            print(f"  Cache expirado ({age.days}d {age.seconds//3600}h). Reentrenando...")
            return None
        age_str = (f"{age.days}d " if age.days else "") + f"{age.seconds//3600}h {(age.seconds%3600)//60}m"
        print(f"  Cache cargado ({age_str} de antiguedad — maximo {CACHE_MAX_AGE_DAYS}d).")
    else:
        print(f"  Cache de snapshot '{snapshot}' cargado (sin expiracion).")

    return (
        payload["elo"], payload["fb"], payload["clf"],
        payload["rank_lookup"], payload["last_match_date"],
        payload.get("elo_tracker") or EloHistoryTracker(),
        payload.get("age_lookup") or {},
    )


# ── model loading ─────────────────────────────────────────────────────────────

def _build_elo_fb(tour: str, raw_dir_override: Optional[Path] = None):
    """Rebuild EloSystem + FeatureBuilder from raw matches.

    Returns (elo, fb, rank_lookup, last_match_date, elo_tracker, age_lookup)
    where rank_lookup maps player names to their most recently observed
    ATP/WTA ranking, last_match_date is the most recent match_date seen in
    the raw data, elo_tracker holds each player's rolling pre-match Elo
    snapshots (see src/features/decay.py), and age_lookup maps player names
    to their most recently observed age + the date it was observed (empty
    for WTA — tennis-data.co.uk has no age column).

    raw_dir_override: read raw matches from here instead of the live
    data/raw/{tour dir} — used for snapshot-pinned loading
    (load_model(snapshot=...)).
    """
    from src.data.loader import load_atp_matches, load_wta_matches
    from src.features.engineering import FeatureBuilder
    from src.models.elo import EloSystem
    from src.backtest.walkforward import build_match_features

    print(f"  Reconstruyendo Elo + FeatureBuilder ({tour.upper()})...")
    loader   = load_atp_matches if tour == "atp" else load_wta_matches
    start    = 1990 if tour == "atp" else 2007
    end      = date.today().year
    df_raw   = loader(start, end, raw_dir_override=raw_dir_override)
    # Sort by date so rank lookup iteration is chronological
    df_raw   = df_raw.sort_values("match_date").reset_index(drop=True)
    rank_lkp = _build_rank_lookup(df_raw)
    age_lkp  = _build_age_lookup(df_raw)
    last_match_date = df_raw["match_date"].max().date()
    elo         = EloSystem()
    fb          = FeatureBuilder()
    elo_tracker = EloHistoryTracker()
    build_match_features(df_raw, elo, fb, elo_tracker=elo_tracker)   # mutates in-place
    print(f"  Listo: {len(elo.general_ratings):,} jugadores, "
          f"{len(rank_lkp):,} con ranking conocido.")
    return elo, fb, rank_lkp, last_match_date, elo_tracker, age_lkp


def _train_lr(tour: str, features_path: Optional[Path] = None) -> Pipeline:
    """Load a features CSV and train a scaled LR on the full history.

    StandardScaler + LogisticRegression, matching walk_forward_backtest's
    per-year scaling (src/backtest/walkforward.py) — before this, the
    backtest reported metrics for a scaled-feature LR while this function
    (which trains the model actually serving live predictions) fit on raw,
    unscaled features. Fitting the scaler here uses the exact same full
    training set the LR itself sees, same as the backtest's per-year
    fit-on-train-only scaler.

    features_path: read from here instead of the default
    data/processed/{tour}_features.csv — used for snapshot-pinned training
    (src/data/snapshots.py's resolve_snapshot_path).
    """
    path = features_path or (_DATA_DIR / "processed" / f"{tour}_features.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. "
            f"Corre primero: python -m src.pipeline {tour}"
        )
    full = load_features_with_mirror(path)
    n_original = int((~full["is_mirror"]).sum())

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )
    # .values strips DataFrame column names before fit, matching the plain
    # ndarray predict_match() passes to predict_proba() later — fitting on a
    # named DataFrame and predicting on an unnamed array triggers a sklearn
    # UserWarning otherwise.
    clf.fit(full[FEATURE_COLS].fillna(0).values, full["outcome"].values)
    print(f"  LR entrenado sobre {n_original:,} partidos ({tour.upper()}).")
    return clf


def load_model(tour: str = "atp", retrain: bool = False, snapshot: Optional[str] = None):
    """Load or rebuild model, with transparent cache management.

    snapshot: when set, pin loading to data/snapshots/{snapshot}/ instead of
    live data/raw/ + data/processed/. Uses its own cache file
    ({tour}__snapshot-{id}.pkl, no expiry — an immutable snapshot can't go
    stale relative to a cache-age clock) and skips the staleness check
    entirely, since a deliberately old, pinned dataset isn't "stale" — that
    it's old relative to today is the whole point of asking for it.

    Returns:
        elo          – EloSystem with full historical state
        fb           – FeatureBuilder with full historical state
        clf          – Pipeline(StandardScaler, LogisticRegression) trained on all available data
        rank_lookup  – dict {player_name: most_recent_rank}
        elo_tracker  – EloHistoryTracker (rolling pre-match Elo snapshots)
        age_lookup   – dict {player_name: (age, observed_date)}, empty for WTA
    """
    if not retrain:
        cached = _load_cache(tour, snapshot)
        if cached is not None:
            elo, fb, clf, rank_lookup, last_match_date, elo_tracker, age_lookup = cached
            if snapshot is None:
                _check_staleness(tour, last_match_date)
            return elo, fb, clf, rank_lookup, elo_tracker, age_lookup

    if snapshot is not None:
        print(f"Construyendo modelo pinned a snapshot '{snapshot}' ({tour.upper()})...")
        raw_dir       = resolve_snapshot_path(snapshot, tour, "raw")
        features_path = resolve_snapshot_path(snapshot, tour, "processed")
    else:
        print(f"Construyendo modelo desde cero ({tour.upper()}) — primera vez ~30-60 s...")
        raw_dir       = None
        features_path = None

    elo, fb, rank_lkp, last_match_date, elo_tracker, age_lkp = _build_elo_fb(tour, raw_dir_override=raw_dir)
    clf = _train_lr(tour, features_path=features_path)
    _save_cache(tour, elo, fb, clf, rank_lkp, last_match_date, elo_tracker, age_lkp, snapshot=snapshot)
    if snapshot is None:
        _check_staleness(tour, last_match_date)
    return elo, fb, clf, rank_lkp, elo_tracker, age_lkp


# ── feature builder for a new match ──────────────────────────────────────────

def build_prediction_features(
    elo,
    fb,
    player_a: str,
    player_b: str,
    surface: str,
    match_date: date,
    rank_a: Optional[float] = None,
    rank_b: Optional[float] = None,
    elo_tracker: Optional[EloHistoryTracker] = None,
    age_a: Optional[float] = None,
    age_b: Optional[float] = None,
) -> dict:
    """Compute the feature values for a future match from player_a's perspective.

    elo_tracker: rolling pre-match Elo history (see src/features/decay.py).
        When omitted, an empty ephemeral tracker is used, which makes
        rolling_elo_diff neutral (0.0) for both players — matches the
        "insufficient history" fallback in calculate_decay_features.
    age_a / age_b: current age of each player, or None if unknown (always
        the case for WTA — no birthdate in the current data source) —
        None makes age_multiplier neutral (1.0) for that player.
    """
    # Resolve to internal keys (WTA uses 'Lastname I.' format internally)
    pa_key = _resolve_player_name(player_a, elo)
    pb_key = _resolve_player_name(player_b, elo)

    elo_a    = elo.get_effective_rating(pa_key, surface)
    elo_b    = elo.get_effective_rating(pb_key, surface)
    p_elo    = elo.expected_score(elo_a, elo_b)
    elo_diff = elo_a - elo_b

    fa  = fb.get_features(pa_key, pb_key, surface, match_date)
    fb_ = fb.get_features(pb_key, pa_key, surface, match_date)

    # rank_diff = loser_rank - winner_rank (see walkforward.py convention)
    if rank_a is not None and rank_b is not None:
        rank_diff = float(rank_b) - float(rank_a)
    else:
        rank_diff = 0.0

    tracker = elo_tracker if elo_tracker is not None else EloHistoryTracker()
    decay_a = calculate_decay_features(
        elo_a, tracker, pa_key, surface, fb.match_dates(pa_key), match_date, age_a,
        fb.workload_history(pa_key), fb.last_surface_and_date(pa_key),
    )
    decay_b = calculate_decay_features(
        elo_b, tracker, pb_key, surface, fb.match_dates(pb_key), match_date, age_b,
        fb.workload_history(pb_key), fb.last_surface_and_date(pb_key),
    )

    return {
        "elo_diff":          elo_diff,
        "elo_prob":          p_elo,
        "rank_diff":         rank_diff,
        "form_diff":         fa["recent_win_rate"]         - fb_["recent_win_rate"],
        "surface_form_diff": fa["recent_win_rate_surface"] - fb_["recent_win_rate_surface"],
        "h2h_rate":          fa["h2h_win_rate"],
        "rest_diff":         fa["rest_days"]               - fb_["rest_days"],
        "rolling_elo_diff":    decay_a["rolling_elo_diff"]     - decay_b["rolling_elo_diff"],
        "age_multiplier_diff": decay_a["age_multiplier"]       - decay_b["age_multiplier"],
        "rust_factor_diff":    decay_a["rust_factor"]          - decay_b["rust_factor"],
        "fatigue_multiplier_diff": decay_a["fatigue_multiplier"] - decay_b["fatigue_multiplier"],
        "surface_transition_multiplier_diff": decay_a["surface_transition_multiplier"] - decay_b["surface_transition_multiplier"],
        "adjusted_elo_diff":   decay_a["adjusted_elo_surface"] - decay_b["adjusted_elo_surface"],
    }


def predict_match(
    elo, fb, clf,
    player_a: str,
    player_b: str,
    surface: str,
    match_date: date,
    rank_a: Optional[float] = None,
    rank_b: Optional[float] = None,
    rank_lookup: Optional[Dict[str, int]] = None,
    elo_tracker: Optional[EloHistoryTracker] = None,
    age_lookup: Optional[Dict[str, Tuple[float, date]]] = None,
) -> dict:
    """Return win probabilities for both players (raw + calibrated).

    If rank_a or rank_b is None and rank_lookup is provided, the lookup
    is used to fill in the most recently observed ranking automatically.
    Falls back to rank_diff=0 only when both sources are unavailable.

    elo_tracker / age_lookup: passed straight through to
    build_prediction_features for the decay features (rolling_elo_diff,
    age_multiplier_diff, rust_factor_diff, adjusted_elo_diff). age_lookup
    entries are extrapolated to match_date via _current_age(); omitted for
    a player (or the whole tour, e.g. WTA) means age_multiplier stays
    neutral (1.0) for that side.
    """
    # Auto-fill ranks from lookup when caller did not supply them
    rank_a_used = rank_a
    rank_b_used = rank_b
    rank_a_source = "manual" if rank_a is not None else None
    rank_b_source = "manual" if rank_b is not None else None

    if rank_lookup is not None:
        if rank_a_used is None:
            found = lookup_rank(player_a, rank_lookup)
            if found is not None:
                rank_a_used   = found
                rank_a_source = "auto"
        if rank_b_used is None:
            found = lookup_rank(player_b, rank_lookup)
            if found is not None:
                rank_b_used   = found
                rank_b_source = "auto"

    age_a = _current_age(player_a, age_lookup, match_date) if age_lookup is not None else None
    age_b = _current_age(player_b, age_lookup, match_date) if age_lookup is not None else None

    feats = build_prediction_features(
        elo, fb, player_a, player_b, surface, match_date,
        rank_a_used, rank_b_used,
        elo_tracker=elo_tracker, age_a=age_a, age_b=age_b,
    )
    X       = np.array([[feats[c] for c in FEATURE_COLS]])
    p_a_raw = float(clf.predict_proba(X)[0, 1])
    p_a_cal = apply_shrinkage(p_a_raw)

    pa_key = _resolve_player_name(player_a, elo)
    pb_key = _resolve_player_name(player_b, elo)
    return {
        "player_a":      player_a,
        "player_b":      player_b,
        "p_a_raw":       p_a_raw,
        "p_a_cal":       p_a_cal,
        "p_b_raw":       1.0 - p_a_raw,
        "p_b_cal":       1.0 - p_a_cal,
        "rank_a":        rank_a_used,
        "rank_b":        rank_b_used,
        "rank_a_source": rank_a_source,
        "rank_b_source": rank_b_source,
        "features":      feats,
        "elo_a":         elo.get_effective_rating(pa_key, surface),
        "elo_b":         elo.get_effective_rating(pb_key, surface),
        "elo_found_a":   _is_elo_known(player_a, elo),
        "elo_found_b":   _is_elo_known(player_b, elo),
        "matches_a":     elo.match_counts.get(pa_key, 0),
        "matches_b":     elo.match_counts.get(pb_key, 0),
    }


# ── CSV log ───────────────────────────────────────────────────────────────────

_LOG_FIELDS = [
    "timestamp", "tour", "tournament", "surface", "match_date",
    "player_a", "player_b",
    "rank_a", "rank_a_source", "rank_b", "rank_b_source",
    "p_a_raw", "p_a_cal", "p_b_raw", "p_b_cal",
    "odds_a", "odds_a_source", "odds_b", "odds_b_source",
    "bookmaker_a", "bookmaker_b",
    "implied_a", "implied_b",
    "edge_a", "ev_a", "kelly_a",
    "edge_b", "ev_b", "kelly_b",
    "shrinkage_applied",
    "status",    # ok / invalid_missing_elo / test_never_played
    "result",    # pending / A_win / B_win / excluded (test_never_played rows)
    "profit",    # filled in later
    "elo_diff", "elo_prob", "rank_diff", "form_diff",
    "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff", "surface_transition_multiplier_diff", "adjusted_elo_diff",
]


def _migrate_log_header_if_needed() -> None:
    """If LOG_PATH exists with an older header than _LOG_FIELDS (e.g. missing
    odds_a_source/odds_b_source or bookmaker_a/bookmaker_b), rewrite it with the
    current header so old rows stay readable instead of silently misaligning on
    the next append."""
    if not LOG_PATH.exists():
        return
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == _LOG_FIELDS:
            return
        rows = list(reader)
    for row in rows:
        row.setdefault("odds_a_source", "manual")
        row.setdefault("odds_b_source", "manual")
        row.setdefault("bookmaker_a", "pinnacle")
        row.setdefault("bookmaker_b", "pinnacle")
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _LOG_FIELDS})


def log_query(tour, tournament, surface, match_date,
              player_a, player_b,
              pred, val_a, val_b, odds_a, odds_b,
              odds_a_source: str = "manual", odds_b_source: str = "manual",
              bookmaker_a: str = "", bookmaker_b: str = "") -> None:
    """Append one match prediction + odds to the CSV log.

    Rows where either player's Elo was not found (default 1500) are saved
    with status='invalid_missing_elo' instead of 'ok', so they can be
    filtered out during analysis without contaminating the pick history.
    """
    _migrate_log_header_if_needed()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats  = pred["features"]
    shrink = abs(pred["p_a_cal"] - pred["p_a_raw"]) > 0.001

    elo_ok = pred.get("elo_found_a", True) and pred.get("elo_found_b", True)
    status = "ok" if elo_ok else "invalid_missing_elo"

    row = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "tour":            tour,
        "tournament":      tournament,
        "surface":         surface,
        "match_date":      match_date.isoformat(),
        "player_a":        player_a,
        "player_b":        player_b,
        "rank_a":          pred["rank_a"] if pred["rank_a"] is not None else "",
        "rank_a_source":   pred["rank_a_source"] or "none",
        "rank_b":          pred["rank_b"] if pred["rank_b"] is not None else "",
        "rank_b_source":   pred["rank_b_source"] or "none",
        "p_a_raw":         round(pred["p_a_raw"], 4),
        "p_a_cal":         round(pred["p_a_cal"], 4),
        "p_b_raw":         round(pred["p_b_raw"], 4),
        "p_b_cal":         round(pred["p_b_cal"], 4),
        "odds_a":          odds_a,
        "odds_a_source":   odds_a_source,
        "odds_b":          odds_b,
        "odds_b_source":   odds_b_source,
        "bookmaker_a":     bookmaker_a,
        "bookmaker_b":     bookmaker_b,
        "implied_a":       round(val_a["implied_prob"], 4),
        "implied_b":       round(val_b["implied_prob"], 4),
        "edge_a":          round(val_a["edge"], 4),
        "ev_a":            round(val_a["ev"], 4),
        "kelly_a":         round(val_a["kelly_fraction"], 4),
        "edge_b":          round(val_b["edge"], 4),
        "ev_b":            round(val_b["ev"], 4),
        "kelly_b":         round(val_b["kelly_fraction"], 4),
        "shrinkage_applied": shrink,
        "status":          status,
        "result":          "pending",
        "profit":          "",
        **{k: round(feats[k], 4) for k in FEATURE_COLS},
    }

    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)

    if elo_ok:
        print(f"  Guardado en {LOG_PATH}")
    else:
        missing = [player_a] * (not pred.get("elo_found_a", True)) \
                + [player_b] * (not pred.get("elo_found_b", True))
        print(f"  Guardado como INVALIDO (Elo faltante: {', '.join(missing)}) en {LOG_PATH}")


# ── display helpers ───────────────────────────────────────────────────────────

_SURFACES = {"hard", "clay", "grass", "carpet"}


def _fmt_pct(v: float, decimals: int = 1) -> str:
    return f"{v * 100:.{decimals}f}%"


def _rank_str(pred: dict, side: str) -> str:
    """Format rank + source tag for display (e.g. '3 (auto)' or '5 (manual)')."""
    rank   = pred[f"rank_{side}"]
    source = pred[f"rank_{side}_source"]
    if rank is None:
        return "N/A (rank_diff=0)"
    tag = f" ({source})" if source else ""
    return f"{int(rank)}{tag}"


def _print_prediction(pred, val_a, val_b, odds_a, odds_b) -> None:
    pa     = pred["player_a"]
    pb     = pred["player_b"]
    shrunk = abs(pred["p_a_cal"] - pred["p_a_raw"]) > 0.001

    print()
    print("=" * 64)
    print(f"  {pa}  vs  {pb}")
    print("=" * 64)

    # Elo not-found warning
    missing_elo = []
    if not pred.get("elo_found_a", True):
        missing_elo.append(pa)
    if not pred.get("elo_found_b", True):
        missing_elo.append(pb)
    if missing_elo:
        print(f"\n  *** ADVERTENCIA: sin historial Elo para: {', '.join(missing_elo)}")
        print(f"  *** elo_diff / form_diff / h2h usan valores neutros.")
        print(f"  *** Probabilidades poco fiables — verifica el nombre exacto.")

    # Elo + rankings
    def _elo_str(val: float, found: bool) -> str:
        return f"{val:>6.1f}" + ("  [?]" if not found else "")

    print(f"\n  Elo efectivo (superficie)  |  Ranking")
    print(f"    {pa:<28}  {_elo_str(pred['elo_a'], pred.get('elo_found_a', True))}  |  {_rank_str(pred, 'a')}")
    print(f"    {pb:<28}  {_elo_str(pred['elo_b'], pred.get('elo_found_b', True))}  |  {_rank_str(pred, 'b')}")

    # Probabilidades
    print(f"\n  Probabilidades del modelo:")
    print(f"    {'Jugador':<28}  {'LR raw':>7}  {'Calibrada':>9}")
    print(f"    {'-'*50}")
    suffix = "  *" if shrunk else ""
    print(f"    {pa:<28}  {_fmt_pct(pred['p_a_raw']):>7}  {_fmt_pct(pred['p_a_cal']):>9}{suffix}")
    print(f"    {pb:<28}  {_fmt_pct(pred['p_b_raw']):>7}  {_fmt_pct(pred['p_b_cal']):>9}{suffix}")
    if shrunk:
        print("    (* shrinkage aplicado: prob > 90% o < 10%)")

    # Features
    f = pred["features"]
    print(f"\n  Features:")
    print(f"    elo_diff={f['elo_diff']:+.1f}  form_diff={f['form_diff']:+.3f}"
          f"  surf_form={f['surface_form_diff']:+.3f}")
    print(f"    h2h={_fmt_pct(f['h2h_rate'])}  rest_diff={f['rest_diff']:+.0f}d"
          f"  rank_diff={f['rank_diff']:+.0f}")

    # Value table
    print(f"\n  {'Jugador':<28}  {'Cuota':>6}  {'Impl.%':>6}  "
          f"{'Edge':>7}  {'EV':>7}  {'Kelly':>6}  {'Value?':>7}")
    print(f"  {'-'*77}")
    for player, val, odds in [(pa, val_a, odds_a), (pb, val_b, odds_b)]:
        mark = " YES" if val["has_value"] else "  no"
        print(
            f"  {player:<28}  {odds:>6.2f}  "
            f"{_fmt_pct(val['implied_prob']):>6}  "
            f"{val['edge'] * 100:>+6.1f}%  "
            f"{val['ev']:>+6.3f}  "
            f"{val['kelly_fraction'] * 100:>5.1f}%  "
            f"{mark:>7}"
        )
    print("=" * 64)

    bets = [(p, v, o) for p, v, o in [(pa, val_a, odds_a), (pb, val_b, odds_b)]
            if v["has_value"]]
    if not bets:
        print("\n  Sin value en ninguno de los lados. Pasar este partido.")
    else:
        for p, v, o in bets:
            print(f"\n  VALUE BET: {p} @ {o:.2f}")
            print(f"    Edge {v['edge']*100:+.1f}pp  |  EV {v['ev']:+.3f}"
                  f"  |  Kelly {v['kelly_fraction']*100:.1f}% bankroll"
                  f" (cap: {KELLY_CAP*100:.0f}%)")


# ── interactive CLI ───────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val if val else default


def try_auto_odds(tour: str, player_a: str, player_b: str) -> Optional[MatchOdds]:
    """Best-effort odds auto-fill via The Odds API. Never raises, never blocks.

    Returns None silently when ODDS_API_KEY isn't set or no match/bookmaker
    quote was found. Prints a one-time-per-session warning on genuine
    failures (network error, bad key, rate limit) so the user knows why
    it's not filling in — the manual flow always still works.
    """
    global _odds_warned
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        return None
    try:
        bookmaker = os.environ.get("ODDS_API_BOOKMAKER", DEFAULT_BOOKMAKER)
        cache_minutes = int(os.environ.get("ODDS_API_CACHE_MINUTES", DEFAULT_CACHE_MINUTES))
        return get_match_odds(tour, player_a, player_b, api_key, bookmaker, cache_minutes)
    except Exception as e:
        if not _odds_warned:
            print(f"\n  Aviso: no se pudieron obtener cuotas automaticas ({e}). "
                  f"Se pediran las cuotas a mano el resto de la sesion.")
            _odds_warned = True
        return None


def _ask_odds(player: str, default: Optional[float] = None) -> tuple[float, str]:
    """Ask for decimal odds. Returns (value, source); source is 'auto' when the
    caller-supplied default (from the Odds API) was accepted via Enter."""
    # 2dp display rounding; bookmakers already quote at ~2dp precision
    dflt_str = f"{default:.2f}" if default is not None else ""
    while True:
        raw = _ask(f"Cuota decimal para {player}", dflt_str)
        if raw.lower() == "q":
            raise KeyboardInterrupt
        if not raw:
            print("    Cuota requerida (> 1.0).")
            continue
        source = "auto" if (default is not None and raw == dflt_str) else "manual"
        try:
            v = float(raw)
            if 1.0 < v <= 100.0:
                return v, source
            print("    Cuota fuera de rango (1.0 < v <= 100.0). Reintenta.")
        except ValueError:
            print("    Valor invalido. Ingresa un numero.")


def _ask_rank_with_hint(player: str, auto_rank: Optional[int]) -> Optional[float]:
    """Ask for a player's rank, showing the auto-found value as default."""
    if auto_rank is not None:
        hint = f"Enter = usar {auto_rank} (historico)"
        raw = _ask(f"Ranking {player} [{hint}]").strip()
        if not raw:
            return float(auto_rank)
        if raw.lower() == "q":
            raise KeyboardInterrupt
        try:
            v = float(raw)
            if 0 < v <= 9999:
                return v
            print("    Valor fuera de rango. Usando ranking historico.")
            return float(auto_rank)
        except ValueError:
            print("    Valor invalido. Usando ranking historico.")
            return float(auto_rank)
    else:
        # No history — ask without default, allow empty for 0 fallback
        raw = _ask(f"Ranking {player} [Enter = sin ranking]").strip()
        if not raw or raw.lower() == "q":
            return None
        try:
            v = float(raw)
            return v if 0 < v <= 9999 else None
        except ValueError:
            return None


def interactive_cli(
    tour: str = "atp", retrain: bool = False, halt_on_suspicious: bool = False,
    snapshot: Optional[str] = None,
) -> None:
    """Run the interactive CLI session."""
    elo, fb, clf, rank_lookup, elo_tracker, age_lookup = load_model(tour, retrain=retrain, snapshot=snapshot)

    if snapshot is not None:
        meta = load_snapshot_metadata(snapshot)
        last_match = meta["tours"][tour]["last_match_date"]
        print(f"\n  *** Usando snapshot pinned '{snapshot}' — datos como al {last_match}. ***")

    print()
    print("=" * 64)
    print(f"  ia-tenis  Value Bet Analyzer  ({tour.upper()})")
    print("=" * 64)
    print("  Escribe 'q' en cualquier prompt para salir.")
    print()

    while True:
        print("\n--- NUEVO PARTIDO ---")

        try:
            tournament = _ask("Torneo")
            if tournament.lower() == "q":
                break

            surface_raw = _ask("Superficie (hard/clay/grass/carpet)", "hard").lower()
            if surface_raw == "q":
                break
            surface = surface_raw if surface_raw in _SURFACES else "hard"
            if surface != surface_raw:
                print("    Superficie no reconocida; usando 'hard'.")

            date_str = _ask("Fecha del partido (YYYY-MM-DD)", date.today().isoformat())
            if date_str.lower() == "q":
                break
            try:
                match_date = date.fromisoformat(date_str)
            except ValueError:
                match_date = date.today()
                print(f"    Fecha invalida; usando hoy: {match_date}")

            player_a = _ask("Jugador A (ej. Jannik Sinner)")
            if player_a.lower() == "q":
                break
            player_b = _ask("Jugador B")
            if player_b.lower() == "q":
                break

            # Ambiguity warning when multiple 'Lastname X.' keys match
            for _pname in (player_a, player_b):
                _cands = _elo_candidates(_pname, elo, fb)
                if len(_cands) > 1:
                    chosen   = _cands[0]
                    alt_strs = " o ".join(
                        f"{k} ({n} partidos, {s})" for k, n, s in _cands[1:]
                    )
                    # Pick the candidate with the most different first-name prefix as example
                    example_key = max(_cands[1:], key=lambda x: len(x[0]))[0]
                    print(f"\n  Nombre ambiguo: '{_pname}' podria ser "
                          f"{chosen[0]} ({chosen[1]} partidos, {chosen[2]}) "
                          f"o {alt_strs}.")
                    print(f"  Usando {chosen[0][:-1] if chosen[0].endswith('.') else chosen[0]}. "
                          f"Si esto es incorrecto, escribe el nombre exacto del indice "
                          f"(ej. '{example_key}').")

            # Cuotas automaticas (best-effort, nunca bloquea el flujo)
            auto_odds = try_auto_odds(tour, player_a, player_b)
            if auto_odds is not None:
                bookmaker_label = os.environ.get("ODDS_API_BOOKMAKER", DEFAULT_BOOKMAKER)
                print(f"\n  Cuotas encontradas ({bookmaker_label}): "
                      f"{auto_odds.matched_home} vs {auto_odds.matched_away}")

            # Rankings — show auto-found values as defaults
            auto_a = lookup_rank(player_a, rank_lookup)
            auto_b = lookup_rank(player_b, rank_lookup)

            print(f"\n  Rankings (historico encontrado: "
                  f"{player_a}={auto_a or 'N/A'}, {player_b}={auto_b or 'N/A'}):")
            print("  Presiona Enter para usar el ranking historico, o escribe uno nuevo.")

            rank_a = _ask_rank_with_hint(player_a, auto_a)
            rank_b = _ask_rank_with_hint(player_b, auto_b)

            # Need both ranks or neither for rank_diff to be meaningful
            if (rank_a is None) != (rank_b is None):
                print("  Advertencia: se omite ranking de un jugador — usando rank_diff=0.")
                rank_a = rank_b = None

        except KeyboardInterrupt:
            break

        # Prediction (ranks already resolved, skip auto-lookup inside predict_match)
        try:
            pred = predict_match(
                elo, fb, clf,
                player_a, player_b,
                surface, match_date,
                rank_a=rank_a,
                rank_b=rank_b,
                rank_lookup=None,   # already resolved above
                elo_tracker=elo_tracker,
                age_lookup=age_lookup,
            )
            # Store source info for display and logging
            pred["rank_a_source"] = (
                "manual" if rank_a is not None and rank_a != auto_a
                else ("auto" if auto_a is not None and rank_a is not None else None)
            )
            pred["rank_b_source"] = (
                "manual" if rank_b is not None and rank_b != auto_b
                else ("auto" if auto_b is not None and rank_b is not None else None)
            )
        except Exception as e:
            print(f"\n  Error al calcular prediccion: {e}")
            continue

        # Odds input
        try:
            print(f"\n  Ingresa las cuotas decimales:")
            odds_a, odds_a_source = _ask_odds(
                player_a, default=auto_odds.odds_a if auto_odds else None
            )
            odds_b, odds_b_source = _ask_odds(
                player_b, default=auto_odds.odds_b if auto_odds else None
            )
        except KeyboardInterrupt:
            break

        val_a = calculate_value(pred["p_a_cal"], odds_a)
        val_b = calculate_value(pred["p_b_cal"], odds_b)

        _print_prediction(pred, val_a, val_b, odds_a, odds_b)

        elo_ok = pred.get("elo_found_a", True) and pred.get("elo_found_b", True)
        suspicious = _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious)
        low_sample = (
            pred.get("matches_a", MIN_MATCHES_THRESHOLD) < MIN_MATCHES_THRESHOLD
            or pred.get("matches_b", MIN_MATCHES_THRESHOLD) < MIN_MATCHES_THRESHOLD
        )

        if suspicious:
            best_edge = max(val_a["edge"], val_b["edge"])
            print(f"\n  *** BLOQUEADO: edge sospechoso ({best_edge*100:.1f}% > "
                  f"{MAX_SUSPICIOUS_EDGE*100:.0f}%) ***")
            print("  *** Posible dato stale o error de matching. Revisa manualmente.")
            print("  *** No se guarda en esta sesion. Corre sin --halt-on-suspicious para loguear igual.")
            save = ""
        elif low_sample:
            print(f"\n  *** BLOQUEADO: muestra insuficiente (matches_a={pred.get('matches_a', 0)}, "
                  f"matches_b={pred.get('matches_b', 0)}, minimo {MIN_MATCHES_THRESHOLD}) ***")
            print("  *** El Elo de al menos un jugador aun no es confiable — revisa manualmente.")
            save = ""
        elif not elo_ok:
            print("\n  *** Elo faltante para uno o ambos jugadores — la prediccion no es confiable.")
            save = _ask("  Guardar como invalida para mantener trazabilidad? (s/n)", "n").lower()
        else:
            save = _ask("\n  Guardar en log? (s/n)", "s").lower()

        logged = _should_log_prediction(suspicious or low_sample, save)
        if logged:
            log_query(
                tour, tournament, surface, match_date,
                player_a, player_b,
                pred, val_a, val_b, odds_a, odds_b,
                odds_a_source, odds_b_source,
            )

        # Audit log: every evaluated prediction, unconditionally — not just
        # ones saved above. This is what makes future calibration/reliability
        # analysis possible (the full population, not a human-selected
        # subset). See docs/superpowers/specs/2026-07-25-calibration-persistence-design.md.
        # Best-effort, like try_auto_odds: this fires on every iteration now
        # (not just on human opt-in like log_query), so a write failure here
        # (disk full, permissions) must never crash the rest of the session
        # or lose the value_bets_log.csv save that just happened above.
        try:
            decision = classify_audit_decision(
                low_sample, suspicious, elo_ok, logged,
                val_a["has_value"], val_b["has_value"],
            )
            log_prediction_audit(
                tour, tournament, surface, match_date,
                player_a, player_b,
                pred, val_a, val_b, odds_a, odds_b,
                odds_a_source, odds_b_source,
                decision=decision,
                model_snapshot_id=snapshot,
                shrink_hi=_SHRINK_HI, shrink_lo=_SHRINK_LO, shrink_rate=_SHRINK_RATE,
            )
        except Exception as e:
            print(f"\n  Aviso: no se pudo escribir en el audit log ({e}). Continuando sesion.")

        again = _ask("  Analizar otro partido? (s/n)", "s").lower()
        if again not in ("s", "si", "y", "yes", ""):
            break

    print("\n  Sesion terminada.")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tennis value-bet analyzer — manual odds input"
    )
    parser.add_argument("--wta", action="store_true",
                        help="Usar modelo WTA en lugar de ATP (default)")
    parser.add_argument("--retrain", action="store_true",
                        help="Ignorar cache y reentrenar modelo desde cero")
    parser.add_argument("--halt-on-suspicious", action="store_true",
                        dest="halt_on_suspicious",
                        help="Bloquea el guardado en log si el edge supera "
                             f"{MAX_SUSPICIOUS_EDGE*100:.0f}%% (posible dato stale)")
    parser.add_argument("--snapshot", type=str, default=None,
                        help="Usar un snapshot pinned (data/snapshots/{id}/) en vez de datos en vivo")
    args = parser.parse_args()
    tour = "wta" if args.wta else "atp"
    interactive_cli(
        tour, retrain=args.retrain, halt_on_suspicious=args.halt_on_suspicious,
        snapshot=args.snapshot,
    )


if __name__ == "__main__":
    main()

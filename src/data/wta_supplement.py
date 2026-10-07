"""Conservative WTA main-draw supplement from Valuebetennis.

Valuebetennis, Résultats et cotes de tennis depuis 2021,
https://www.valuebetennis.com/donnees.htm (CC BY 4.0).

Only settled matches after the latest tennis-data.co.uk match are eligible.
Unknown or ambiguous players are omitted rather than silently given a new
Elo identity. A minimum match-coverage gate prevents a partial supplement
from making an incomplete WTA history appear fresh.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "match_id", "date", "tournoi", "categorie", "genre", "surface", "tour",
    "joueur1", "joueur1_id", "joueur2", "joueur2_id", "vainqueur_id", "score",
}
MAIN_TOUR_CATEGORIES = {"Main tour", "Masters", "Grand Chelem"}
MIN_MAIN_DRAW_ROUND = 4
MIN_MATCH_COVERAGE = 0.90
MAX_CARRIED_RANK_AGE_DAYS = 35
SURFACE_MAP = {"dur": "hard", "terre battue": "clay", "gazon": "grass", "moquette": "carpet"}
_SET_SCORE_PATTERN = re.compile(r"\d+-\d+(?:\(\d+\))?")


def _tokens(name: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", name)
    without_accents = "".join(c for c in normalized if not unicodedata.combining(c))
    return re.findall(r"[a-z]+", without_accents.lower())


def _candidate_parts(name: str) -> tuple[list[str], str] | None:
    """Parse a canonical WTA key such as 'Osorio Serrano M.C.'."""
    parts = name.strip().split()
    i = len(parts) - 1
    while i >= 1 and parts[i].endswith("."):
        i -= 1
    if i == len(parts) - 1 or i < 0:
        return None
    surname = _tokens(" ".join(parts[: i + 1]))
    abbreviated_first = "".join(_tokens(" ".join(parts[i + 1 :])))
    if not surname or not abbreviated_first:
        return None
    return surname, abbreviated_first


def resolve_player_name(full_name: str, name_stats: dict[str, tuple[date, int]]) -> str | None:
    """Match full source name to the most recently used compatible WTA key.

    Longer first-name fragments (e.g. Wang Xin. vs Wang Xiy.) must match the
    actual given name, while multi-token given names match initials (e.g.
    Elena Gabriela -> E.G.). Exact ties remain unresolved.
    """
    full = _tokens(full_name)
    if len(full) < 2:
        return None
    matches = []
    for canonical, (last_seen, count) in name_stats.items():
        if _tokens(canonical) == full:
            return canonical
        parts = _candidate_parts(canonical)
        if parts is None:
            continue
        surname, abbreviated_first = parts
        for start in range(1, len(full) - len(surname) + 1):
            if full[start : start + len(surname)] != surname:
                continue
            given = full[:start]
            initials = "".join(token[0] for token in given)
            if given[0].startswith(abbreviated_first) or initials.startswith(abbreviated_first):
                # A multi-letter given-name prefix is more identifying than
                # a one-letter legacy key (Wang Xin. vs Wang X.).
                strong_prefix = len(abbreviated_first) > 1 and given[0].startswith(abbreviated_first)
                matches.append((int(strong_prefix), last_seen, count, canonical))
                break
    if not matches:
        return None
    matches.sort(reverse=True)
    if len(matches) > 1 and matches[0][:3] == matches[1][:3]:
        return None
    return matches[0][3]


def _name_stats(primary: pd.DataFrame) -> dict[str, tuple[date, int]]:
    appearances = pd.concat(
        [
            primary[["match_date", "winner_name"]].rename(columns={"winner_name": "name"}),
            primary[["match_date", "loser_name"]].rename(columns={"loser_name": "name"}),
        ],
        ignore_index=True,
    )
    grouped = appearances.groupby("name")["match_date"].agg(["max", "count"])
    return {name: (row["max"].date(), int(row["count"])) for name, row in grouped.iterrows()}


def _latest_ranks(primary: pd.DataFrame) -> dict[str, tuple[float, date]]:
    ranks: dict[str, tuple[float, date]] = {}
    observations = []
    for name_col, rank_col in (("winner_name", "winner_rank"), ("loser_name", "loser_rank")):
        known = primary[["match_date", name_col, rank_col]].dropna(subset=[rank_col])
        observations.append(known.rename(columns={name_col: "name", rank_col: "rank"}))
    combined = pd.concat(observations, ignore_index=True).sort_values("match_date", kind="stable")
    for name, rank, observed in zip(combined["name"], combined["rank"], combined["match_date"]):
        ranks[name] = (float(rank), observed.date())
    return ranks


def _carried_rank(name: str, match_date: date, ranks: dict[str, tuple[float, date]]) -> float:
    entry = ranks.get(name)
    if entry is None or not 0 <= (match_date - entry[1]).days <= MAX_CARRIED_RANK_AGE_DAYS:
        return float("nan")
    return entry[0]


def load_supplement(
    path: Path, primary: pd.DataFrame, year: int, reference_date: date,
) -> tuple[pd.DataFrame, int, int]:
    """Return (accepted rows, eligible rows, unresolved rows) for one year.

    A zero-row result when coverage is below MIN_MATCH_COVERAGE is deliberate:
    it leaves the original dataset and its staleness warning intact.
    """
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing Valuebetennis columns: {sorted(missing)}")

    current_primary = primary[primary["year"] == year]
    if current_primary.empty:
        return pd.DataFrame(), 0, 0
    last_primary = current_primary["match_date"].max()
    raw = raw.copy()
    raw["match_date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    round_number = pd.to_numeric(raw["tour"], errors="coerce")
    raw = raw[
        (raw["genre"] == "wta")
        & raw["categorie"].isin(MAIN_TOUR_CATEGORIES)
        & (round_number >= MIN_MAIN_DRAW_ROUND)
        & (raw["match_date"].dt.year == year)
        & (raw["match_date"] > last_primary)
        & (raw["match_date"] <= pd.Timestamp(reference_date + timedelta(days=1)))
    ].drop_duplicates(subset=["match_id"], keep="last")
    eligible = len(raw)
    if not eligible:
        return pd.DataFrame(), 0, 0

    history = primary[primary["match_date"] <= last_primary]
    stats = _name_stats(history)
    names = set(raw["joueur1"].dropna()) | set(raw["joueur2"].dropna())
    resolved = {name: resolve_player_name(name, stats) for name in names}
    ranks = _latest_ranks(history)
    records = []
    for _, row in raw.iterrows():
        player1 = resolved.get(row["joueur1"])
        player2 = resolved.get(row["joueur2"])
        surface = SURFACE_MAP.get(str(row["surface"]).strip().lower())
        if not player1 or not player2 or player1 == player2 or not surface:
            continue
        ids = (row["joueur1_id"], row["joueur2_id"])
        if pd.isna(row["vainqueur_id"]) or ids[0] == ids[1] or row["vainqueur_id"] not in ids:
            continue
        winner, loser = (player1, player2) if row["vainqueur_id"] == ids[0] else (player2, player1)
        score = row["score"] if isinstance(row["score"], str) else ""
        match_date = row["match_date"].date()
        records.append({
            "match_date": row["match_date"], "year": year, "tour": "wta",
            "winner_name": winner, "loser_name": loser,
            "winner_rank": _carried_rank(winner, match_date, ranks),
            "loser_rank": _carried_rank(loser, match_date, ranks),
            "surface": surface, "sets_played": len(_SET_SCORE_PATTERN.findall(score)),
            "Tournament": row["tournoi"], "score": score,
            "source": "valuebetennis", "source_match_id": row["match_id"],
        })
    accepted = len(records)
    if accepted / eligible < MIN_MATCH_COVERAGE:
        return pd.DataFrame(), eligible, eligible - accepted
    return pd.DataFrame(records).sort_values("match_date", kind="stable").reset_index(drop=True), eligible, eligible - accepted

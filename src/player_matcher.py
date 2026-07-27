"""Match a player name as reported by an external source (The Odds API) to
the canonical name used internally in the Elo/features dataset.

Odds API names and dataset names can differ in format ("A. Zverev" vs
"Alexander Zverev" vs "Zverev A."). This reuses the same surname +
first-initial heuristic already proven in src/odds_api.py (see
src/name_matching.py) rather than adding a fuzzy-matching dependency —
no format seen in this project needs more than that.
"""
from __future__ import annotations

from typing import Iterable, Optional

from src.name_matching import first_initial, surname


def match_player_name(discovered_name: str, canonical_names: Iterable[str]) -> Optional[str]:
    """Return the canonical name matching discovered_name, or None.

    None is returned both when nothing matches and when more than one
    canonical name is equally compatible (same surname, same or unknown
    first initial) — guessing wrong is worse than not matching, so
    ambiguous cases are left for the caller to handle (e.g. skip the match
    and warn) rather than silently picking one.
    """
    canonical_names = list(canonical_names)
    if discovered_name in canonical_names:
        return discovered_name

    target_surname = surname(discovered_name)
    target_initial = first_initial(discovered_name)

    matches = []
    for candidate in canonical_names:
        if surname(candidate) != target_surname:
            continue
        candidate_initial = first_initial(candidate)
        if not target_initial or not candidate_initial or target_initial == candidate_initial:
            matches.append(candidate)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        # Ambiguous unless the initials actually distinguish them (e.g. one
        # candidate had no initial to compare) — re-check with exact initial
        # equality only, dropping the "unknown initial" leniency.
        strict = [c for c in matches if target_initial and first_initial(c) == target_initial]
        if len(strict) == 1:
            return strict[0]
        return None

    return None

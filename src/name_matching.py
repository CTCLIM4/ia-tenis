"""Shared name-normalization helpers for matching player names across sources
that format them differently: full name ("Aryna Sabalenka"), abbreviated
surname-first ("Sabalenka A."), or abbreviated first-name ("A. Sabalenka").

Extracted from src/odds_api.py so src/player_matcher.py can reuse the exact
same, already-tested normalization instead of a third divergent
implementation. No fuzzy-matching dependency (e.g. rapidfuzz) — surname +
first-initial comparison is sufficient for the name formats every source in
this project actually produces.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/periods, collapse whitespace for name matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.replace(".", "")
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


def surname(name: str) -> str:
    parts = normalize_name(name).split(" ")
    # Drop single-character tokens (initials like "a" in "Sabalenka A.")
    # so the surname is identified consistently regardless of whether the
    # source lists it first ("Sabalenka A.") or last ("Aryna Sabalenka").
    surname_parts = [p for p in parts if len(p) > 1]
    return surname_parts[-1] if surname_parts else (parts[-1] if parts else "")


def first_initial(name: str) -> str:
    """First letter of the player's first name, independent of whether the
    source lists it as 'Sabalenka A.' (surname-first, abbreviated), 'A.
    Sabalenka' (first-initial-first, abbreviated) or 'Aryna Sabalenka'
    (first-last, full).

    Returns "" when the name has no separate first-name token (e.g. a bare
    surname) — callers must treat that as "unknown" and not reject a match
    on it, since there's nothing to compare.

    Does NOT disambiguate siblings whose first names also share an initial
    (e.g. Karolina/Kristyna Pliskova, both "K") — that needs the fuller
    2-letter-prefix heuristic used elsewhere for WTA
    (_resolve_player_name in value_analysis.py), which isn't available here
    since callers of this module only carry single-letter initials.
    """
    parts = [p for p in normalize_name(name).split(" ") if p]
    if len(parts) < 2:
        return ""
    sn = surname(name)
    first_name_token = next((p for p in parts if p != sn), parts[0])
    return first_name_token[0]

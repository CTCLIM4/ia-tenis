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


def _tagged_tokens(name: str) -> list[tuple[str, bool]]:
    """Split into normalized tokens, each paired with whether the ORIGINAL
    (pre-normalize) token ended with a period. normalize_name() strips
    periods, so this has to run on the raw string first -- a trailing
    period is the only reliable signal that a token is a truncated first
    name (e.g. "Xin." disambiguating "Wang Xin." from another "Wang X."),
    which can be more than one letter and would otherwise be
    indistinguishable from a real surname."""
    tagged = []
    for raw in name.split():
        had_period = raw.endswith(".")
        norm = normalize_name(raw)
        if norm:
            tagged.append((norm, had_period))
    return tagged


def surname(name: str) -> str:
    tokens = _tagged_tokens(name)
    if not tokens:
        return ""
    # A token is an abbreviated first-name fragment -- never the surname --
    # if the original ended with a period ("A." or the longer "Xin.") or
    # it's a bare single letter even without one. Whatever's left is the
    # surname; when the source lists it first ("Sabalenka A.") that's the
    # sole remaining token, when last ("Aryna Sabalenka") it's the last one.
    non_abbrev = [tok for tok, had_period in tokens if not (had_period or len(tok) <= 1)]
    return non_abbrev[-1] if non_abbrev else tokens[-1][0]


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

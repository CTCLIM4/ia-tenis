"""Resolve a tennis match's playing surface from Odds API event metadata.

Batch/automated prediction (src/daily_scanner.py) has no human in the loop
to type the surface the way interactive_cli() does, so it must be inferred
from what The Odds API gives us: sport_title (e.g. "Wimbledon") and sport_key
(e.g. "tennis_atp_wimbledon").

Three-tier fallback, in priority order:
  1. Explicit "clay"/"hard"/"grass" keyword in the title or key text.
  2. Known tournament name in TOURNAMENT_SURFACES.
  3. Unknown -> None. Callers must exclude that match from automated
     prediction rather than guessing a default surface (an Elo lookup on
     the wrong surface silently biases the prediction).
"""
from __future__ import annotations

from typing import Optional

_SURFACE_KEYWORDS = ("clay", "hard", "grass")

# Common ATP/WTA tournaments not identifiable by keyword alone. Extend as new
# tournaments show up in the discovered event feed.
TOURNAMENT_SURFACES = {
    "australian open": "hard",
    "roland garros": "clay",
    "french open": "clay",
    "wimbledon": "grass",
    "us open": "hard",
    "indian wells": "hard",
    "miami open": "hard",
    "monte carlo": "clay",
    "madrid": "clay",
    "italian open": "clay",
    "rome": "clay",
    "canadian open": "hard",
    "cincinnati": "hard",
    "shanghai": "hard",
    "paris masters": "hard",
    "kitzbuhel": "clay",
    "estoril": "clay",
    "hamburg": "clay",
    "prague": "hard",
    "washington": "hard",
    "monterrey": "hard",
}


def resolve_surface(sport_title: str, sport_key: str) -> Optional[str]:
    """Return 'hard' / 'clay' / 'grass', or None if it can't be determined."""
    haystack = f"{sport_title} {sport_key}".lower()
    for surface in _SURFACE_KEYWORDS:
        if surface in haystack:
            return surface

    title_lower = sport_title.lower()
    for tournament, surface in TOURNAMENT_SURFACES.items():
        if tournament in title_lower:
            return surface

    print(f"  Aviso: no se pudo determinar la superficie para '{sport_title}' "
          f"({sport_key}). Partido excluido del escaneo automatico.")
    return None

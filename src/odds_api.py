"""The Odds API client: fetch, cache, and match tennis odds by player name."""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "data" / "odds_cache"

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
DEFAULT_BOOKMAKER = "bet365"
DEFAULT_CACHE_MINUTES = 15


@dataclass
class MatchOdds:
    odds_a: float
    odds_b: float
    matched_home: str
    matched_away: str


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents/periods, collapse whitespace for name matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.replace(".", "")
    return re.sub(r"\s+", " ", ascii_name).strip().lower()


def _surname(name: str) -> str:
    parts = _normalize_name(name).split(" ")
    # Drop single-character tokens (initials like "a" in "Sabalenka A.")
    # so the surname is identified consistently regardless of whether the
    # source lists it first ("Sabalenka A.") or last ("Aryna Sabalenka").
    surname_parts = [p for p in parts if len(p) > 1]
    return surname_parts[-1] if surname_parts else (parts[-1] if parts else "")


def find_match_odds(
    events: list[dict],
    player_a: str,
    player_b: str,
    bookmaker: str = DEFAULT_BOOKMAKER,
) -> Optional[MatchOdds]:
    """Find the event matching player_a/player_b (either order) and extract
    that bookmaker's h2h prices.

    Returns None when no event's participants match, or when a matching
    event exists but the configured bookmaker didn't quote it — callers
    must not substitute a different bookmaker.
    """
    surname_a = _surname(player_a)
    surname_b = _surname(player_b)

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        surname_home = _surname(home)
        surname_away = _surname(away)

        if surname_home == surname_a and surname_away == surname_b:
            order = (home, away)
        elif surname_home == surname_b and surname_away == surname_a:
            order = (away, home)
        else:
            continue

        for bk in event.get("bookmakers", []):
            if bk.get("key") != bookmaker:
                continue
            for market in bk.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                if order[0] in prices and order[1] in prices:
                    return MatchOdds(
                        odds_a=prices[order[0]],
                        odds_b=prices[order[1]],
                        matched_home=home,
                        matched_away=away,
                    )
        return None  # event matched but this bookmaker didn't quote it

    return None


def fetch_odds_events(tour: str, api_key: str) -> list[dict]:
    """Fetch raw upcoming h2h odds events for a tour from The Odds API."""
    sport_key = f"tennis_{tour}"
    url = (
        f"{ODDS_API_BASE}/{sport_key}/odds/"
        f"?apiKey={api_key}&regions=eu&markets=h2h&oddsFormat=decimal"
    )
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read())


def _cache_path(tour: str) -> Path:
    return CACHE_DIR / f"{tour}.json"


def _load_cache(tour: str, max_age_minutes: int) -> Optional[list[dict]]:
    path = _cache_path(tour)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    fetched_at = datetime.fromisoformat(payload["fetched_at"])
    if datetime.now(timezone.utc) - fetched_at >= timedelta(minutes=max_age_minutes):
        return None
    return payload["events"]


def _save_cache(tour: str, events: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"fetched_at": datetime.now(timezone.utc).isoformat(), "events": events}
    _cache_path(tour).write_text(json.dumps(payload), encoding="utf-8")


def get_events(tour: str, api_key: str, cache_minutes: int = DEFAULT_CACHE_MINUTES) -> list[dict]:
    """Return cached events if fresh, otherwise fetch and cache."""
    cached = _load_cache(tour, cache_minutes)
    if cached is not None:
        return cached
    events = fetch_odds_events(tour, api_key)
    _save_cache(tour, events)
    return events


def get_match_odds(
    tour: str,
    player_a: str,
    player_b: str,
    api_key: str,
    bookmaker: str = DEFAULT_BOOKMAKER,
    cache_minutes: int = DEFAULT_CACHE_MINUTES,
) -> Optional[MatchOdds]:
    """Top-level lookup: cached/fetched events -> matched odds for this pairing."""
    events = get_events(tour, api_key, cache_minutes)
    return find_match_odds(events, player_a, player_b, bookmaker)

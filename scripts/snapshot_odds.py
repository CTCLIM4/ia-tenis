"""Record pre-match ATP/WTA odds snapshots from The Odds API, to later
measure whether opening lines carry an edge that closing lines don't.

Uso:
  python -m scripts.snapshot_odds
  python -m scripts.snapshot_odds --regions eu --min-remaining 150

Tennis-Data's historical odds are closing prices only, so "bet early, before
the line moves" can't be backtested from them (see
docs/metrics/2026-10-07-model-vs-market-edge.md). This builds that history
going forward: each run fetches h2h odds for every active ATP/WTA
tournament and appends to data/odds_snapshots/{YYYY-MM}.csv one row per
(event, bookmaker) whose prices changed since the last stored row. Only
matches that haven't started are recorded -- in-play prices would pollute
the "closing" line. Every run, even one that writes nothing new, is logged
in data/odds_snapshots/runs.csv: with change-only rows, that is what says a
price was still current at a given time.

Credits: each tournament costs one credit per region (h2h only). The free
plan has 500/month, shared with the daily scanner (~3 credits per active
tournament per day). A run stops fetching once the remaining credits would
drop below --min-remaining, so snapshots never starve the daily scan.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.config  # noqa: F401  (loads .env before ODDS_API_KEY is read)
from src.odds_api import ODDS_API_BASE, list_tennis_sport_keys

DEFAULT_OUT_DIR = _ROOT / "data" / "odds_snapshots"
DEFAULT_REGIONS = "eu"
DEFAULT_MIN_REMAINING = 150
SNAPSHOT_COLS = [
    "snapshot_utc", "sport_key", "event_id", "commence_time", "home", "away",
    "bookmaker", "bookmaker_last_update", "price_home", "price_away",
]
RUN_COLS = [
    "snapshot_utc", "regions", "tournaments_fetched", "tournaments_skipped_budget",
    "events_prematch", "rows_written", "credits_remaining",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_json(url: str) -> tuple[list, Optional[int]]:
    """GET url -> (parsed JSON, x-requests-remaining or None)."""
    with urllib.request.urlopen(url, timeout=20) as resp:
        remaining = resp.headers.get("x-requests-remaining")
        return json.loads(resp.read()), (int(float(remaining)) if remaining else None)


def _fetch_index(api_key: str, retries: int = 2) -> tuple[list, Optional[int]]:
    """The tournament list is free and required: retry transient failures, then raise."""
    for attempt in range(retries + 1):
        try:
            return fetch_json(f"{ODDS_API_BASE}?apiKey={api_key}")
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(attempt + 1)
    raise RuntimeError("unreachable")


def event_rows(event: dict, sport_key: str, snapshot_utc: str) -> list[dict]:
    home, away = event.get("home_team", ""), event.get("away_team", "")
    rows = []
    for bk in event.get("bookmakers", []):
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            prices = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
            if prices.get(home) is None or prices.get(away) is None:
                continue
            rows.append({
                "snapshot_utc": snapshot_utc, "sport_key": sport_key, "event_id": event.get("id", ""),
                "commence_time": event.get("commence_time", ""), "home": home, "away": away,
                "bookmaker": bk.get("key", ""), "bookmaker_last_update": market.get("last_update", ""),
                "price_home": float(prices[home]), "price_away": float(prices[away]),
            })
    return rows


def _month_files(out_dir: Path, now: datetime) -> list[Path]:
    prev = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
    return [out_dir / f"{prev[0]:04d}-{prev[1]:02d}.csv", out_dir / f"{now.year:04d}-{now.month:02d}.csv"]


def last_prices(out_dir: Path, now: datetime) -> dict[tuple[str, str], tuple[float, float]]:
    """(event_id, bookmaker) -> latest stored (price_home, price_away), from
    this month's and last month's files (an event never spans more)."""
    latest: dict[tuple[str, str], tuple[float, float]] = {}
    for path in _month_files(out_dir, now):
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                latest[(r["event_id"], r["bookmaker"])] = (float(r["price_home"]), float(r["price_away"]))
    return latest


def _append(path: Path, cols: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def take_snapshot(
    api_key: str, out_dir: Path = DEFAULT_OUT_DIR, regions: str = DEFAULT_REGIONS,
    min_remaining: int = DEFAULT_MIN_REMAINING, now: Optional[datetime] = None,
) -> dict:
    now = now or _now()
    stamp = _iso(now)
    cost = len([r for r in regions.split(",") if r.strip()])
    index, remaining = _fetch_index(api_key)
    sports = list_tennis_sport_keys(index)
    previous = last_prices(out_dir, now)

    new_rows, fetched, skipped, prematch = [], 0, 0, 0
    for sport in sports:
        if remaining is not None and remaining - cost < min_remaining:
            skipped += 1
            continue
        url = (f"{ODDS_API_BASE}/{sport['key']}/odds/?apiKey={api_key}"
               f"&regions={regions}&markets=h2h&oddsFormat=decimal")
        try:
            events, rem = fetch_json(url)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  Aviso: no se pudieron obtener cuotas para {sport['key']} ({exc}). Saltando.")
            continue
        fetched += 1
        remaining = rem if rem is not None else remaining
        for event in events:
            commence = event.get("commence_time")
            if not commence or datetime.fromisoformat(commence.replace("Z", "+00:00")) <= now:
                continue  # already started: in-play price, not part of the pre-match line
            prematch += 1
            for row in event_rows(event, sport["key"], stamp):
                key = (row["event_id"], row["bookmaker"])
                if previous.get(key) != (row["price_home"], row["price_away"]):
                    new_rows.append(row)
                    previous[key] = (row["price_home"], row["price_away"])

    _append(out_dir / f"{now.year:04d}-{now.month:02d}.csv", SNAPSHOT_COLS, new_rows)
    run = {
        "snapshot_utc": stamp, "regions": regions, "tournaments_fetched": fetched,
        "tournaments_skipped_budget": skipped, "events_prematch": prematch,
        "rows_written": len(new_rows), "credits_remaining": "" if remaining is None else remaining,
    }
    _append(out_dir / "runs.csv", RUN_COLS, [run])
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guarda snapshots de cuotas pre-partido ATP/WTA (The Odds API)")
    parser.add_argument("--regions", default=os.environ.get("ODDS_SNAPSHOT_REGIONS", DEFAULT_REGIONS),
                        help="regiones de The Odds API; cada una cuesta 1 credito por torneo (default: eu)")
    parser.add_argument("--min-remaining", type=int,
                        default=int(os.environ.get("ODDS_SNAPSHOT_MIN_REMAINING", DEFAULT_MIN_REMAINING)),
                        help="no gastar creditos por debajo de este saldo (reserva para el escaneo diario)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        sys.exit("ODDS_API_KEY no configurada (entorno o .env).")
    run = take_snapshot(api_key, args.out_dir, args.regions, args.min_remaining)
    print(f"Snapshot {run['snapshot_utc']}: {run['tournaments_fetched']} torneo(s), "
          f"{run['events_prematch']} partido(s) pre-partido, {run['rows_written']} fila(s) nuevas, "
          f"{run['tournaments_skipped_budget']} torneo(s) saltados por presupuesto, "
          f"creditos restantes {run['credits_remaining']}")


if __name__ == "__main__":
    main()

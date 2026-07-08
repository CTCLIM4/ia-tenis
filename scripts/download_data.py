"""
Download tennis datasets.

ATP  → stats.tennismylife.org API  (public)  → data/raw/tennis_atp_tml/{year}.csv
       Manifest: GET https://stats.tennismylife.org/api/data-files
       Uses `requests` + certifi's bundled CA bundle explicitly (NOT urllib's
       OS-trust-store default) — on Windows, urllib's default SSL context can
       fall back to the OS certificate store (schannel), which has shown
       flaky revocation-check behavior in prior diagnostics. requests, with
       certifi installed, verifies against a bundled root CA list instead,
       sidestepping that store entirely. Do not swap this back to urllib.
WTA  → tennis-data.co.uk           (public)  → data/raw/tennis_wta_tduk/{year}w.xls
       URL pattern: http://www.tennis-data.co.uk/{year}w/{year}w.xls
       Available from 2007.  Verify exact URLs at tennis-data.co.uk/wta.php if
       downloads fail (the site occasionally restructures paths between seasons).
"""
import datetime
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd
import requests

DATA_RAW = Path("data/raw")

# ── ATP ──────────────────────────────────────────────────────────────────────
ATP_DIR = DATA_RAW / "tennis_atp_tml"
ATP_MANIFEST_URL = "https://stats.tennismylife.org/api/data-files"

ATP_SCHEMA_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "indoor",
    "tourney_date", "match_num", "winner_id", "winner_seed", "winner_entry", "winner_name",
    "winner_hand", "winner_ht", "winner_ioc", "winner_age", "winner_rank", "winner_rank_points",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand", "loser_ht", "loser_ioc",
    "loser_age", "loser_rank", "loser_rank_points", "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved",
    "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced",
]

_ATP_YEAR_FILE_RE = re.compile(r"^\d{4}\.csv$")
_ATP_MERGE_KEY = ["tourney_id", "tourney_name", "round", "match_num"]
_ATP_LEGACY_PATHS = [
    ".git", ".github", "README.md", "logo.jpg", "ATP_Database.csv", "ongoing_tourneys.csv",
]


def _filter_atp_manifest(files: list[dict]) -> list[dict]:
    """Keep only main-tour year files and the live ongoing-tourneys feed.

    Drops challenger files, qualifying-draw files, and the ATP_Database.csv
    aggregate — none of these are read by src/data/loader.py.
    """
    return [
        f for f in files
        if _ATP_YEAR_FILE_RE.match(f["name"]) or f["name"] == "ongoing_tourneys.csv"
    ]


def _fetch_atp_manifest() -> list[dict]:
    """GET the file manifest from the API and return the filtered file list."""
    resp = requests.get(ATP_MANIFEST_URL, timeout=30)
    resp.raise_for_status()
    return _filter_atp_manifest(resp.json()["files"])


def _should_download(local_path: Path, remote_size: int) -> bool:
    """True if local_path is missing or its size differs from remote_size."""
    if not local_path.exists():
        return True
    return local_path.stat().st_size != remote_size

WTA_START_YEAR = 2007
WTA_DIR = DATA_RAW / "tennis_wta_tduk"
_WTA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


_XLS_MAGIC  = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_XLSX_MAGIC = b"PK"


def _detect_format(data: bytes):
    if data[:8] == _XLS_MAGIC:
        return "xls"
    if data[:2] == _XLSX_MAGIC:
        return "xlsx"
    head = data[:200].lstrip()
    if head and head[:1] not in (b"<",) and b"<html" not in head.lower():
        return "csv"
    return None


def _download_wta_year(year: int) -> bool:
    """Download one year of WTA data from tennis-data.co.uk. Returns True on success.

    tennis-data.co.uk uses inconsistent naming across years; try all known patterns.
    """
    # Skip if any format already present
    for ext in (".xls", ".xlsx", ".csv"):
        if (WTA_DIR / f"{year}w{ext}").exists():
            print(f"  {year}w already present, skipping.")
            return True

    base = "http://www.tennis-data.co.uk"
    patterns = [
        f"{base}/{year}w/{year}w.xls",
        f"{base}/{year}w/{year}.xls",
        f"{base}/{year}w/{year}w.xlsx",
        f"{base}/{year}w/{year}.xlsx",
        f"{base}/{year}w/{year}w.csv",
        f"{base}/{year}w/{year}.csv",
    ]
    for url in patterns:
        try:
            req = urllib.request.Request(url, headers=_WTA_HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            fmt = _detect_format(data)
            if fmt:
                dest = WTA_DIR / f"{year}w.{fmt}"
                dest.write_bytes(data)
                print(f"  Downloaded: {dest.name}")
                return True
        except Exception:
            pass
    print(f"  WARNING: could not download {year}: no pattern worked")
    return False


def download():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # ── ATP ──────────────────────────────────────────────────────────────────
    dest = DATA_RAW / ATP_REPO["dest"]
    url = ATP_REPO["url"]
    if dest.exists():
        print("tennis_atp_tml: already exists, pulling latest...")
        try:
            subprocess.run(["git", "-C", str(dest), "pull"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: pull failed — {e}")
    else:
        print("Cloning Tennismylife/TML-Database (ATP)...")
        try:
            subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)
            print(f"  Done: {dest}")
        except subprocess.CalledProcessError:
            print(f"\n  ERROR: could not clone {url}\n")

    # ── WTA ──────────────────────────────────────────────────────────────────
    import datetime
    current_year = datetime.date.today().year
    WTA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading WTA data ({WTA_START_YEAR}-{current_year}) from tennis-data.co.uk...")
    ok = sum(_download_wta_year(y) for y in range(WTA_START_YEAR, current_year + 1))
    print(f"  WTA: {ok} files ready in {WTA_DIR}")

    print("\nDownload step finished.")


if __name__ == "__main__":
    download()

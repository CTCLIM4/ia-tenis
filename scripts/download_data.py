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
import os
import re
import shutil
import stat
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


def _validate_atp_csv(data: bytes) -> bool:
    """Cheap check that `data` looks like a real ATP CSV: decodable, with a
    header line matching ATP_SCHEMA_COLUMNS exactly. Catches truncated
    downloads and HTML error pages served with a 200 status."""
    if not data:
        return False
    try:
        header_line = data.split(b"\n", 1)[0].decode("utf-8").strip()
    except UnicodeDecodeError:
        return False
    return header_line.split(",") == ATP_SCHEMA_COLUMNS


def _download_file(url: str, dest: Path) -> bool:
    """Download url and atomically write it to dest. Returns True on success.

    On any failure (network error or schema validation failure), dest is left
    completely untouched — no partial or invalid file is ever written there,
    and no .tmp file is left behind either way.
    """
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.content
    except Exception as e:
        print(f"  WARNING: could not download {url}: {e}")
        return False

    if not _validate_atp_csv(data):
        print(f"  WARNING: {dest.name} failed schema validation, skipping.")
        return False

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(dest)
    except Exception as e:
        print(f"  WARNING: could not write {dest.name}: {e}")
        tmp.unlink(missing_ok=True)
        return False
    return True


def _merge_ongoing_into_year(year_df: pd.DataFrame, ongoing_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the archived year file with the live ongoing-tourneys feed,
    deduplicating on a composite key that's robust to tourney_id being reused
    across unrelated tournaments (observed live: "2026-416" used by both
    Munich and Rome Masters). On a genuine duplicate match, the archived
    year_df row wins over the live ongoing_df row.
    """
    year_df = year_df.copy()
    ongoing_df = ongoing_df.copy()
    year_df["_source_priority"] = 0
    ongoing_df["_source_priority"] = 1

    combined = pd.concat([year_df, ongoing_df], ignore_index=True)
    combined = combined.sort_values("_source_priority", kind="stable")
    combined = combined.drop_duplicates(subset=_ATP_MERGE_KEY, keep="first")
    combined = combined.drop(columns=["_source_priority"])
    return combined[ATP_SCHEMA_COLUMNS].reset_index(drop=True)


def _force_remove_readonly(func, path, exc):
    """shutil.rmtree onexc callback: clear the read-only bit and retry.

    Git marks packed/loose objects under .git/objects read-only on Windows,
    which makes plain os.unlink/os.rmdir raise PermissionError. Clearing
    stat.S_IWRITE before retrying the failed operation is the standard
    workaround.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _cleanup_legacy_git_clone(dest: Path) -> None:
    """Remove artifacts from the old git-clone-based ATP source, if present.

    `.git` is the migration marker: if it's gone, this is a no-op, so it's
    safe to call on every run. Existing {year}.csv files are never touched
    here — the download loop (download_atp) decides whether each gets
    refreshed, based on remote size.

    Git marks packed/loose objects under .git/objects read-only on Windows,
    which would otherwise crash a plain shutil.rmtree/Path.unlink here. Each
    path removal is made resilient to that (and to any other filesystem
    hiccup, e.g. a locked file) so one stuck path can't abort cleanup of the
    rest, matching the "never let a filesystem hiccup abort the whole run"
    convention used by _download_file.
    """
    if not (dest / ".git").exists():
        return
    print("  Migrating away from git clone: removing legacy artifacts...")
    for name in _ATP_LEGACY_PATHS:
        path = dest / name
        try:
            if path.is_dir():
                shutil.rmtree(path, onexc=_force_remove_readonly)
            elif path.exists():
                path.unlink()
        except Exception as e:
            print(f"  WARNING: could not remove {path.name}: {e}")


def download_atp() -> None:
    """Download ATP main-tour year files + ongoing_tourneys.csv from the
    stats.tennismylife.org API, then merge the live ongoing feed into the
    current year's file so load_atp_matches() sees it with zero changes."""
    ATP_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_git_clone(ATP_DIR)

    try:
        manifest = _fetch_atp_manifest()
    except Exception as e:
        print(f"  ERROR: could not fetch ATP manifest: {e}")
        return

    current_year_name = f"{datetime.date.today().year}.csv"
    downloaded, skipped, failed = 0, 0, 0
    have_ongoing = have_current_year = False

    for entry in manifest:
        dest = ATP_DIR / entry["name"]
        if entry["name"] == "ongoing_tourneys.csv":
            have_ongoing = True
        elif entry["name"] == current_year_name:
            have_current_year = True

        if not _should_download(dest, entry["size"]):
            skipped += 1
            continue
        if _download_file(entry["url"], dest):
            downloaded += 1
        else:
            failed += 1

    print(f"  ATP: {downloaded} downloaded, {skipped} already current or unchanged, {failed} failed.")

    year_path = ATP_DIR / current_year_name
    ongoing_path = ATP_DIR / "ongoing_tourneys.csv"
    if have_ongoing and have_current_year and year_path.exists() and ongoing_path.exists():
        try:
            year_df = pd.read_csv(year_path, low_memory=False)
            ongoing_df = pd.read_csv(ongoing_path, low_memory=False)
            merged = _merge_ongoing_into_year(year_df, ongoing_df)
            tmp = year_path.with_suffix(year_path.suffix + ".tmp")
            merged.to_csv(tmp, index=False)
            tmp.replace(year_path)
            print(f"  Merged ongoing_tourneys.csv into {current_year_name} ({len(merged)} rows).")
        except Exception as e:
            print(f"  WARNING: could not merge ongoing_tourneys.csv into {current_year_name}: {e}")

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
    print("Downloading ATP data from stats.tennismylife.org...")
    download_atp()

    # ── WTA ──────────────────────────────────────────────────────────────────
    current_year = datetime.date.today().year
    WTA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading WTA data ({WTA_START_YEAR}-{current_year}) from tennis-data.co.uk...")
    ok = sum(_download_wta_year(y) for y in range(WTA_START_YEAR, current_year + 1))
    print(f"  WTA: {ok} files ready in {WTA_DIR}")

    print("\nDownload step finished.")


if __name__ == "__main__":
    download()

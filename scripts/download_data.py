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
import io
import os
import re
import shutil
import stat
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import requests

# Run as `python scripts/download_data.py` (run_prediction.py, README), so the
# repo root isn't on sys.path by default.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.wta_supplement import REQUIRED_COLUMNS as WTA_SUPPLEMENT_COLUMNS

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
    """shutil.rmtree onerror callback: clear the read-only bit and retry.

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
                shutil.rmtree(path, onerror=_force_remove_readonly)
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
        try:
            name = entry["name"]
            dest = ATP_DIR / name
            if name == "ongoing_tourneys.csv":
                have_ongoing = True
            elif name == current_year_name:
                have_current_year = True

            if not _should_download(dest, entry["size"]):
                skipped += 1
                continue
            if _download_file(entry["url"], dest):
                downloaded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  WARNING: could not process {entry.get('name', '?')}: {e}")
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
WTA_SUPPLEMENT_URL = "https://www.valuebetennis.com/datasets/valuebetennis-matchs-{year}.csv"
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


def _wta_remote_size(url: str) -> int | None:
    """HEAD `url` and return its Content-Length if the server reports one,
    else None (network error or missing header)."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers=_WTA_HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None
    except Exception:
        return None


_WTA_DATA_PAGE_URL = "https://www.tennis-data.co.uk/data.php"
_WTA_LINK_RE = re.compile(r'href="([^"]*/(\d{4})w/\d{4}\.(?:xlsx?|csv))"', re.IGNORECASE)


def _fetch_wta_links() -> dict[int, str]:
    """Scrape data.php for the WTA download URL of each year it currently lists.

    tennis-data.co.uk periodically moves its files under a new opaque path
    prefix (seen live 2026-09-18: the old {year}w/{year}w.xls-style guesses
    became .../hrjk-85HytOjkhth76j_ygh4jf7/{year}w/{year}.xlsx). Scraping the
    live page for the href it actually publishes survives that; guessing a
    hardcoded pattern list doesn't. Returns {} on any network error --
    callers must fall back to the guessed patterns in that case.
    """
    try:
        req = urllib.request.Request(_WTA_DATA_PAGE_URL, headers=_WTA_HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return {}

    links: dict[int, str] = {}
    base = "https://www.tennis-data.co.uk/"
    for href, year_str in _WTA_LINK_RE.findall(html):
        url = href if href.lower().startswith("http") else base + href.lstrip("/")
        links[int(year_str)] = url
    return links


def _download_wta_year(year: int, discovered_url: str | None = None) -> bool:
    """Download one year of WTA data from tennis-data.co.uk. Returns True on success.

    tennis-data.co.uk uses inconsistent naming across years; try all known patterns.
    discovered_url (from _fetch_wta_links, scraped off the live site) is tried
    first when given, since it reflects the site's current path -- the
    hardcoded guessed patterns below are only a fallback for when scraping
    itself fails.

    Mirrors the ATP path's remote-size comparison (_should_download): an
    already-present local file is only skipped once its size is confirmed to
    still match the remote's reported size, so the current, in-progress
    season's file keeps getting refreshed instead of being skipped forever
    once it first exists locally.
    """
    base = "http://www.tennis-data.co.uk"
    guessed_patterns = [
        f"{base}/{year}w/{year}w.xls",
        f"{base}/{year}w/{year}.xls",
        f"{base}/{year}w/{year}w.xlsx",
        f"{base}/{year}w/{year}.xlsx",
        f"{base}/{year}w/{year}w.csv",
        f"{base}/{year}w/{year}.csv",
    ]
    patterns = [discovered_url] + guessed_patterns if discovered_url else guessed_patterns

    for ext in (".xls", ".xlsx", ".csv"):
        local = WTA_DIR / f"{year}w{ext}"
        if not local.exists():
            continue
        for url in patterns:
            if not url.endswith(ext):
                continue
            remote_size = _wta_remote_size(url)
            if remote_size is not None and not _should_download(local, remote_size):
                print(f"  {year}w already current ({local.stat().st_size} bytes), skipping.")
                return True
        break  # a local file exists for this extension; fall through to re-download

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


def download_wta_supplement(year: int) -> bool:
    """Refresh the current-year Valuebetennis CSV without replacing a good copy
    on a download/schema failure. The loader only uses settled WTA main-draw
    matches beyond tennis-data.co.uk's most recent match.

    Valuebetennis, Résultats et cotes de tennis depuis 2021,
    https://www.valuebetennis.com/donnees.htm (CC BY 4.0).
    """
    url = WTA_SUPPLEMENT_URL.format(year=year)
    dest = WTA_DIR / f"valuebetennis_{year}.csv"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.content
        parsed = pd.read_csv(io.BytesIO(data), sep=";", encoding="utf-8-sig", low_memory=False)
        missing = WTA_SUPPLEMENT_COLUMNS - set(parsed.columns)
        if missing or not parsed["genre"].eq("wta").any():
            raise ValueError(f"invalid WTA supplement schema (missing: {sorted(missing)})")
        if dest.exists() and dest.read_bytes() == data:
            print(f"  WTA supplement {year} unchanged.")
            return True
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".csv.tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"  WTA supplement {year} downloaded ({len(data)} bytes).")
        return True
    except Exception as e:
        print(f"  WARNING: could not refresh WTA supplement {year}: {e}")
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
    wta_links = _fetch_wta_links()
    ok = sum(
        _download_wta_year(y, wta_links.get(y))
        for y in range(WTA_START_YEAR, current_year + 1)
    )
    print(f"  WTA: {ok} files ready in {WTA_DIR}")

    download_wta_supplement(current_year)

    print("\nDownload step finished.")


if __name__ == "__main__":
    download()

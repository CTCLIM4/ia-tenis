"""
Download tennis datasets.

ATP  → Tennismylife/TML-Database  (public)  → data/raw/tennis_atp_tml/{year}.csv
WTA  → tennis-data.co.uk           (public)  → data/raw/tennis_wta_tduk/{year}w.xls
       URL pattern: http://www.tennis-data.co.uk/{year}w/{year}w.xls
       Available from 2007.  Verify exact URLs at tennis-data.co.uk/wta.php if
       downloads fail (the site occasionally restructures paths between seasons).
"""
import subprocess
import urllib.request
from pathlib import Path

DATA_RAW = Path("data/raw")

ATP_REPO = {
    "dest": "tennis_atp_tml",
    "url": "https://github.com/Tennismylife/TML-Database.git",
}

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

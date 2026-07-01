"""
Download Jeff Sackmann tennis datasets.

NOTES:
  - JeffSackmann/tennis_atp and tennis_wta are currently private repos.
  - Option A: Authenticate with GitHub first (`gh auth login`), then run this script.
  - Option B: Place CSV files manually:
      data/raw/tennis_atp/atp_matches_YYYY.csv
      data/raw/tennis_wta/wta_matches_YYYY.csv
    Files are available from tennis-data.co.uk or other public mirrors.
"""
import subprocess
import sys
from pathlib import Path

DATA_RAW = Path("data/raw")

REPOS = {
    "tennis_atp": "https://github.com/JeffSackmann/tennis_atp.git",
    "tennis_wta": "https://github.com/JeffSackmann/tennis_wta.git",
}


def download():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        dest = DATA_RAW / name
        if dest.exists():
            print(f"{name}: already exists, pulling latest...")
            try:
                subprocess.run(["git", "-C", str(dest), "pull"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"  WARNING: pull failed — {e}")
        else:
            print(f"Cloning {name} (shallow)...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", url, str(dest)], check=True
                )
                print(f"  Done: {dest}")
            except subprocess.CalledProcessError:
                print(
                    f"\n  ERROR: could not clone {url}\n"
                    f"  The repo may be private. Authenticate with:\n"
                    f"    gh auth login\n"
                    f"  Then re-run this script.\n"
                    f"  Alternatively, place CSV files manually in {dest}/\n"
                )
    print("Download step finished.")


if __name__ == "__main__":
    download()

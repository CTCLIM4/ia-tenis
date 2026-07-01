"""
Download tennis datasets.

ATP  → Tennismylife/TML-Database  (public)  → data/raw/tennis_atp_tml/{year}.csv
WTA  → TODO: source not yet resolved
"""
import subprocess
from pathlib import Path

DATA_RAW = Path("data/raw")

ATP_REPO = {
    "dest": "tennis_atp_tml",
    "url": "https://github.com/Tennismylife/TML-Database.git",
}


def download():
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # ATP
    dest = DATA_RAW / ATP_REPO["dest"]
    url = ATP_REPO["url"]
    if dest.exists():
        print(f"tennis_atp_tml: already exists, pulling latest...")
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

    # WTA — TODO
    print("WTA: no data source configured yet (TODO).")

    print("Download step finished.")


if __name__ == "__main__":
    download()

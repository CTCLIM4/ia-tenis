import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.timezone_utils import lima_today

RAW_DATA_DIR = Path("data/raw")

_SURFACE_MAP = {
    "Clay": "clay",
    "Hard": "hard",
    "Grass": "grass",
    "Carpet": "carpet",
}

_SET_SCORE_PATTERN = re.compile(r"\d+-\d+(?:\(\d+\))?")
_BRACKETED_PATTERN = re.compile(r"\[.*?\]")


def _count_sets_played(score) -> int:
    """Count set-score tokens in a raw score string (e.g. "7-6(5) 6-4" -> 2).

    Retirement scores ("6-3 2-4 RET") count the partial set — real games
    were played. Walkovers ("W/O") and missing/non-string scores -> 0.
    Bracketed tokens (e.g. Davis Cup match-tiebreaks like "[10-7]", played
    instead of a 3rd set) are stripped before counting — they aren't a
    real extra set.
    """
    if not isinstance(score, str):
        return 0
    return len(_SET_SCORE_PATTERN.findall(_BRACKETED_PATTERN.sub("", score)))


def _clean(df: pd.DataFrame, reference_date: Optional[date] = None) -> pd.DataFrame:
    df = df.copy()
    df["match_date"] = pd.to_datetime(
        df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["surface"] = df["surface"].map(_SURFACE_MAP).fillna("unknown")
    df = df.dropna(subset=["winner_name", "loser_name", "match_date"])
    cutoff = pd.Timestamp((reference_date or lima_today()) + timedelta(days=1))
    df = df[df["match_date"] <= cutoff]
    for col in ("winner_rank", "loser_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")
    df["sets_played"] = df["score"].apply(_count_sets_played)
    return df.sort_values("match_date").reset_index(drop=True)


def load_atp_matches(
    start_year: int = 1990, end_year: int = 2024, raw_dir_override: Optional[Path] = None,
) -> pd.DataFrame:
    """Load ATP matches from Tennismylife/TML-Database (data/raw/tennis_atp_tml/{year}.csv).

    raw_dir_override: read from this directory instead of the default
    data/raw/tennis_atp_tml — used for snapshot-pinned loading
    (src/data/snapshots.py's resolve_snapshot_path), where the directory is
    a frozen copy under data/snapshots/{id}/raw/tennis_atp_tml.
    """
    tour_dir = raw_dir_override if raw_dir_override is not None else (RAW_DATA_DIR / "tennis_atp_tml")
    frames = []
    for year in range(start_year, end_year + 1):
        path = tour_dir / f"{year}.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            df["year"] = year
            df["tour"] = "atp"
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No ATP match files found in {tour_dir} for years {start_year}-{end_year}. "
            f"Clone Tennismylife/TML-Database into data/raw/tennis_atp_tml/ first."
        )
    return _clean(pd.concat(frames, ignore_index=True))


_TDUK_SURFACE_MAP = {
    "Hard": "hard",
    "hard": "hard",
    "Clay": "clay",
    "clay": "clay",
    "Grass": "grass",
    "grass": "grass",
    "Carpet": "carpet",
    "carpet": "carpet",
    "Hard (I)": "hard",
    "iHard": "hard",
    "Hardcourt": "hard",
    "Hardcourt (I)": "hard",
}

_TDUK_COL_ALIASES = {
    "Winner": "winner_name",
    "Loser": "loser_name",
    "WRank": "winner_rank",
    "LRank": "loser_rank",
    "Surface": "_surface_raw",
    "Date": "_date_raw",
}


def _clean_wta(df: pd.DataFrame, year: int, reference_date: Optional[date] = None) -> pd.DataFrame:
    """Normalize tennis-data.co.uk WTA columns to the internal schema."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _TDUK_COL_ALIASES.items() if k in df.columns})

    # Date: tennis-data.co.uk uses DD/MM/YYYY
    df["match_date"] = pd.to_datetime(df["_date_raw"], dayfirst=True, errors="coerce")
    df["surface"] = df["_surface_raw"].map(_TDUK_SURFACE_MAP).fillna("unknown")

    for col in ("winner_rank", "loser_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")

    df["year"] = year
    df["tour"] = "wta"

    w_sets = pd.to_numeric(df["Wsets"], errors="coerce") if "Wsets" in df.columns else pd.Series(0.0, index=df.index)
    l_sets = pd.to_numeric(df["Lsets"], errors="coerce") if "Lsets" in df.columns else pd.Series(0.0, index=df.index)
    df["sets_played"] = (w_sets.fillna(0) + l_sets.fillna(0)).astype(int)

    df = df.dropna(subset=["winner_name", "loser_name", "match_date"])
    df = df[df["winner_name"].str.strip() != ""]
    df = df[df["loser_name"].str.strip() != ""]
    cutoff = pd.Timestamp((reference_date or lima_today()) + timedelta(days=1))
    df = df[df["match_date"] <= cutoff]
    return df.sort_values("match_date").reset_index(drop=True)


def _read_tduk_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False, encoding="latin-1")
    if suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".xls":
        return pd.read_excel(path, engine="xlrd")
    raise ValueError(f"Unsupported file type: {path}")


def load_wta_matches(
    start_year: int = 2007, end_year: int = 2024, raw_dir_override: Optional[Path] = None,
) -> pd.DataFrame:
    """Load WTA matches from tennis-data.co.uk (data/raw/tennis_wta_tduk/{year}w.[xls|xlsx|csv]).

    Download files from tennis-data.co.uk/wta.php and place them in
    data/raw/tennis_wta_tduk/.  Run scripts/download_data.py to automate.

    raw_dir_override: same as load_atp_matches's, for
    data/raw/tennis_wta_tduk — used for snapshot-pinned loading.
    """
    tour_dir = raw_dir_override if raw_dir_override is not None else (RAW_DATA_DIR / "tennis_wta_tduk")
    frames = []
    for year in range(start_year, end_year + 1):
        for suffix in (f"{year}w.csv", f"{year}w.xlsx", f"{year}w.xls"):
            path = tour_dir / suffix
            if path.exists():
                try:
                    frames.append(_clean_wta(_read_tduk_file(path), year))
                except Exception as e:
                    print(f"  WARNING: could not read {path}: {e}")
                break

    if not frames:
        raise FileNotFoundError(
            f"No WTA match files found in {tour_dir} for years {start_year}-{end_year}. "
            "Run scripts/download_data.py or download manually from tennis-data.co.uk/wta.php."
        )
    return pd.concat(frames, ignore_index=True)

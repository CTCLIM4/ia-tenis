from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")

_SURFACE_MAP = {
    "Clay": "clay",
    "Hard": "hard",
    "Grass": "grass",
    "Carpet": "carpet",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_date"] = pd.to_datetime(
        df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["surface"] = df["surface"].map(_SURFACE_MAP).fillna("unknown")
    df = df.dropna(subset=["winner_name", "loser_name", "match_date"])
    for col in ("winner_rank", "loser_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")
    return df.sort_values("match_date").reset_index(drop=True)


def load_atp_matches(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    """Load ATP matches from Tennismylife/TML-Database (data/raw/tennis_atp_tml/{year}.csv)."""
    tour_dir = RAW_DATA_DIR / "tennis_atp_tml"
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


def load_wta_matches(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    # TODO: find a WTA data source and implement this loader.
    raise NotImplementedError("WTA data source not yet configured.")

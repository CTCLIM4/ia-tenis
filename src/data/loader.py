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


def _load_tour(tour: str, start_year: int, end_year: int) -> pd.DataFrame:
    prefix = "atp" if tour == "atp" else "wta"
    tour_dir = RAW_DATA_DIR / f"tennis_{tour}"
    frames = []
    for year in range(start_year, end_year + 1):
        path = tour_dir / f"{prefix}_matches_{year}.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            df["year"] = year
            df["tour"] = tour
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No {tour.upper()} match files found in {tour_dir} "
            f"for years {start_year}-{end_year}. Run scripts/download_data.py first."
        )
    return _clean(pd.concat(frames, ignore_index=True))


def load_atp_matches(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    return _load_tour("atp", start_year, end_year)


def load_wta_matches(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    return _load_tour("wta", start_year, end_year)

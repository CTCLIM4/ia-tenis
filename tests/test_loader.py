import pandas as pd
import pytest
from io import StringIO
from src.data.loader import _clean

SAMPLE_CSV = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
    "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
    "winner_ht,winner_ioc,winner_age,winner_rank,winner_rank_points,"
    "loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,"
    "loser_ioc,loser_age,loser_rank,loser_rank_points,score,best_of,round,minutes\n"
    "2023-1,AO,Hard,128,G,20230117,1,101,,,Djokovic,R,188,SRB,35.7,1,10000,"
    "102,,,Murray,R,190,GBR,35.5,5,500,6-3 6-4,5,R32,85\n"
    "2023-1,AO,Clay,128,G,20230116,2,103,,,Nadal,R,185,ESP,36.7,2,8000,"
    "104,,,Federer,R,185,SUI,41.7,3,7000,7-5 6-3,5,R32,92\n"
)


def test_clean_parses_dates():
    df = _clean(pd.read_csv(StringIO(SAMPLE_CSV)))
    assert str(df["match_date"].dtype).startswith("datetime64")
    assert df.iloc[0]["match_date"].year == 2023


def test_clean_normalizes_surface_to_lowercase():
    df = _clean(pd.read_csv(StringIO(SAMPLE_CSV)))
    assert set(df["surface"].unique()).issubset({"clay", "hard", "grass", "carpet", "unknown"})


def test_clean_sorts_by_date_ascending():
    df = _clean(pd.read_csv(StringIO(SAMPLE_CSV)))
    dates = df["match_date"].tolist()
    assert dates == sorted(dates)


def test_clean_drops_rows_with_missing_player_names():
    raw = pd.read_csv(StringIO(SAMPLE_CSV))
    raw.loc[0, "winner_name"] = None
    df = _clean(raw)
    assert len(df) == 1


def test_clean_coerces_rank_to_float_and_allows_nan():
    raw = pd.read_csv(StringIO(SAMPLE_CSV))
    raw["winner_rank"] = "N/A"
    df = _clean(raw)
    assert df["winner_rank"].isna().all()


def test_clean_adds_tour_column_when_provided():
    raw = pd.read_csv(StringIO(SAMPLE_CSV))
    raw["tour"] = "atp"
    df = _clean(raw)
    assert "tour" in df.columns

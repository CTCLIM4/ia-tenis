import pandas as pd
import pytest
from io import StringIO
from src.data.loader import _clean, _clean_wta, load_atp_matches, load_wta_matches

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


# ── tennis-data.co.uk WTA loader ─────────────────────────────────────────────

WTA_CSV = (
    "Date,Surface,Winner,Loser,WRank,LRank,B365W,B365L\n"
    "15/01/2023,Hard,Swiatek I.,Kvitova P.,1,6,1.15,5.50\n"
    "16/01/2023,Clay,Sabalenka A.,Rybakina E.,5,25,1.80,1.95\n"
    "17/01/2023,Grass,Gauff C.,Pegula J.,6,4,2.10,1.75\n"
)


def _wta_raw():
    return pd.read_csv(StringIO(WTA_CSV))


def test_clean_wta_renames_columns():
    df = _clean_wta(_wta_raw(), 2023)
    assert "winner_name" in df.columns
    assert "loser_name" in df.columns
    assert "winner_rank" in df.columns
    assert "loser_rank" in df.columns


def test_clean_wta_parses_dates():
    df = _clean_wta(_wta_raw(), 2023)
    assert str(df["match_date"].dtype).startswith("datetime64")
    assert df.iloc[0]["match_date"].year == 2023
    assert df.iloc[0]["match_date"].month == 1


def test_clean_wta_normalizes_surface():
    df = _clean_wta(_wta_raw(), 2023)
    assert set(df["surface"].unique()).issubset({"hard", "clay", "grass", "carpet", "unknown"})


def test_clean_wta_sets_tour_and_year():
    df = _clean_wta(_wta_raw(), 2023)
    assert (df["tour"] == "wta").all()
    assert (df["year"] == 2023).all()


def test_clean_wta_ranks_are_numeric():
    df = _clean_wta(_wta_raw(), 2023)
    assert pd.api.types.is_numeric_dtype(df["winner_rank"])
    assert df.iloc[0]["winner_rank"] == 1


def test_clean_wta_drops_rows_missing_players():
    raw = _wta_raw()
    raw.loc[0, "Winner"] = None
    df = _clean_wta(raw, 2023)
    assert len(df) == 2


def test_clean_wta_sorts_by_date():
    df = _clean_wta(_wta_raw(), 2023)
    dates = df["match_date"].tolist()
    assert dates == sorted(dates)


def test_clean_wta_indoor_hard_maps_to_hard():
    raw = _wta_raw()
    raw.loc[0, "Surface"] = "Hard (I)"
    df = _clean_wta(raw, 2023)
    assert df.iloc[df["match_date"].argmin()]["surface"] == "hard"


# ── raw_dir_override (snapshot-pinned loading) ──────────────────────────────

class TestLoadAtpMatchesRawDirOverride:
    def test_reads_from_override_directory_instead_of_default(self, tmp_path):
        override_dir = tmp_path / "custom_atp"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(SAMPLE_CSV)

        df = load_atp_matches(2023, 2023, raw_dir_override=override_dir)

        assert len(df) == 2
        assert set(df["winner_name"]) == {"Djokovic", "Nadal"}

    def test_raises_when_override_directory_has_no_matching_year_files(self, tmp_path):
        override_dir = tmp_path / "empty_atp"
        override_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_atp_matches(2023, 2023, raw_dir_override=override_dir)


class TestLoadWtaMatchesRawDirOverride:
    def test_reads_from_override_directory_instead_of_default(self, tmp_path):
        override_dir = tmp_path / "custom_wta"
        override_dir.mkdir()
        (override_dir / "2023w.csv").write_text(WTA_CSV)

        df = load_wta_matches(2023, 2023, raw_dir_override=override_dir)

        assert len(df) == 3

    def test_raises_when_override_directory_has_no_matching_year_files(self, tmp_path):
        override_dir = tmp_path / "empty_wta"
        override_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_wta_matches(2023, 2023, raw_dir_override=override_dir)

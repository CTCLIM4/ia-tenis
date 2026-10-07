from datetime import date

import pandas as pd
import pytest
from io import StringIO
from src.data import loader
from src.data.loader import (
    _clean, _clean_wta, _count_sets_played,
    load_atp_matches, load_davis_cup_matches, load_wta_matches,
)

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


def test_clean_drops_rows_dated_more_than_one_day_past_reference():
    raw = pd.read_csv(StringIO(SAMPLE_CSV))  # tourney_date rows: 20230117, 20230116
    df = _clean(raw, reference_date=date(2023, 1, 14))
    assert len(df) == 0


def test_clean_keeps_rows_within_one_day_of_reference():
    raw = pd.read_csv(StringIO(SAMPLE_CSV))
    df = _clean(raw, reference_date=date(2023, 1, 20))
    assert len(df) == 2


def test_clean_defaults_reference_date_to_lima_today(monkeypatch):
    monkeypatch.setattr(loader, "lima_today", lambda: date(2023, 1, 1))
    raw = pd.read_csv(StringIO(SAMPLE_CSV))
    df = _clean(raw)
    assert len(df) == 0


# ── tennis-data.co.uk WTA loader ─────────────────────────────────────────────

WTA_CSV = (
    "Date,Surface,Winner,Loser,WRank,LRank,Wsets,Lsets,B365W,B365L\n"
    "15/01/2023,Hard,Swiatek I.,Kvitova P.,1,6,2,0,1.15,5.50\n"
    "16/01/2023,Clay,Sabalenka A.,Rybakina E.,5,25,2,1,1.80,1.95\n"
    "17/01/2023,Grass,Gauff C.,Pegula J.,6,4,2,0,2.10,1.75\n"
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


def test_clean_wta_drops_rows_dated_more_than_one_day_past_reference():
    df = _clean_wta(_wta_raw(), 2023, reference_date=date(2023, 1, 13))
    assert len(df) == 0


def test_clean_wta_keeps_rows_within_one_day_of_reference():
    df = _clean_wta(_wta_raw(), 2023, reference_date=date(2023, 1, 20))
    assert len(df) == 3


def test_clean_wta_defaults_reference_date_to_lima_today(monkeypatch):
    monkeypatch.setattr(loader, "lima_today", lambda: date(2023, 1, 13))
    df = _clean_wta(_wta_raw(), 2023)
    assert len(df) == 0


# Regression test for the 2026-08-12 Iasi Open incident: a single-digit
# year typo in the source spreadsheet (2029-07-20 instead of 2026-07-20)
# propagated into last_match_date and tripped the invalid_future_date
# staleness guard. The filter must drop only the corrupted row.
WTA_CSV_WITH_FUTURE_TYPO = (
    "Date,Surface,Winner,Loser,WRank,LRank,Wsets,Lsets,B365W,B365L\n"
    "18/07/2026,Clay,Badosa P.,Zidansek T.,10,20,2,0,1.50,2.50\n"
    "20/07/2029,Clay,Sherif M.,Badosa P.,97,115,1,0,2.75,1.44\n"
)


def test_clean_wta_drops_future_typo_row_but_keeps_valid_row():
    raw = pd.read_csv(StringIO(WTA_CSV_WITH_FUTURE_TYPO))
    df = _clean_wta(raw, 2026, reference_date=date(2026, 8, 12))
    assert len(df) == 1
    assert df.iloc[0]["match_date"] == pd.Timestamp(2026, 7, 18)


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


class TestLoaderEndYearDefaultsToCurrentYear:
    """Regression: load_atp_matches/load_wta_matches/load_davis_cup_matches
    all had a stale hardcoded end_year default (2024/2026) -- a caller that
    omits end_year (or passes None) must see the current year's file, not
    silently stop short of it."""

    def test_atp_loader_uses_current_year_default(self, tmp_path):
        override_dir = tmp_path / "atp_source"
        override_dir.mkdir()
        current_year = date.today().year
        (override_dir / f"{current_year}.csv").write_text(SAMPLE_CSV)

        df = load_atp_matches(current_year, None, raw_dir_override=override_dir)

        assert len(df) == 2

    def test_wta_loader_uses_current_year_default(self, tmp_path):
        override_dir = tmp_path / "wta_source"
        override_dir.mkdir()
        current_year = date.today().year
        (override_dir / f"{current_year}w.csv").write_text(WTA_CSV)

        df = load_wta_matches(current_year, None, raw_dir_override=override_dir)

        assert len(df) == 3

    def test_davis_loader_uses_current_year_default(self, tmp_path):
        override_dir = tmp_path / "atp_source"
        override_dir.mkdir()
        current_year = date.today().year
        raw = DAVIS_SAMPLE_CSV.replace("2023", str(current_year))
        (override_dir / f"{current_year}.csv").write_text(raw)

        df = load_davis_cup_matches(current_year, None, raw_dir_override=override_dir)

        assert len(df) == 2


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


# ── Davis Cup (filtered from the same ATP source, tourney_level == "D") ────
#
# tennis-data.co.uk has no Davis Cup data (verified 2026-09-18: daviscup.php
# and every guessed archive path 404 there — the only "Davis Cup" text on the
# whole site is an <option> linking out to daviscup.org). Davis Cup ties are
# already embedded in the same stats.tennismylife.org yearly ATP feed used by
# load_atp_matches(), tagged tourney_level == "D", so load_davis_cup_matches()
# filters that instead of hitting a separate source or raw directory.

DAVIS_SAMPLE_CSV = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
    "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
    "winner_ht,winner_ioc,winner_age,winner_rank,winner_rank_points,"
    "loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,"
    "loser_ioc,loser_age,loser_rank,loser_rank_points,score,best_of,round,minutes\n"
    "2023-D-ESP-USA,Davis Cup,Hard,4,D,20230119,2,103,,,Nadal,R,185,ESP,36.7,2,8000,"
    "104,,,Isner,R,208,USA,42.5,50,600,7-5 6-3,3,RR,95\n"
    "2023-D-ESP-USA,Davis Cup,Hard,4,D,20230118,1,101,,,Alcaraz,R,183,ESP,19.7,1,10000,"
    "102,,,Fritz,R,196,USA,25.2,9,3000,6-3 6-4 6-2,3,RR,90\n"
    "2023-1,AO,Hard,128,G,20230117,1,105,,,Djokovic,R,188,SRB,35.7,1,10000,"
    "106,,,Murray,R,190,GBR,35.5,5,500,6-3 6-4,5,R32,85\n"
)


class TestLoadDavisCupMatches:
    def test_filters_to_davis_cup_rows_only(self, tmp_path):
        override_dir = tmp_path / "atp_source"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(DAVIS_SAMPLE_CSV)

        df = load_davis_cup_matches(2023, 2023, raw_dir_override=override_dir)

        assert len(df) == 2
        assert set(df["winner_name"]) == {"Alcaraz", "Nadal"}
        assert "Djokovic" not in set(df["winner_name"])

    def test_tour_column_is_davis(self, tmp_path):
        override_dir = tmp_path / "atp_source"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(DAVIS_SAMPLE_CSV)

        df = load_davis_cup_matches(2023, 2023, raw_dir_override=override_dir)

        assert (df["tour"] == "davis").all()

    def test_davis_cup_surface_normalization(self, tmp_path):
        override_dir = tmp_path / "atp_source"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(DAVIS_SAMPLE_CSV)

        df = load_davis_cup_matches(2023, 2023, raw_dir_override=override_dir)

        assert set(df["surface"].unique()).issubset({"clay", "hard", "grass", "carpet", "unknown"})

    def test_davis_cup_sorted_by_date(self, tmp_path):
        override_dir = tmp_path / "atp_source"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(DAVIS_SAMPLE_CSV)

        df = load_davis_cup_matches(2023, 2023, raw_dir_override=override_dir)

        dates = df["match_date"].tolist()
        assert dates == sorted(dates)
        assert df.iloc[0]["winner_name"] == "Alcaraz"  # 2023-01-18, earlier

    def test_davis_cup_drops_missing_names(self, tmp_path):
        raw = DAVIS_SAMPLE_CSV.replace("Nadal", "")
        override_dir = tmp_path / "atp_source"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(raw)

        df = load_davis_cup_matches(2023, 2023, raw_dir_override=override_dir)

        assert len(df) == 1
        assert df.iloc[0]["winner_name"] == "Alcaraz"

    def test_davis_cup_sets_played_count(self, tmp_path):
        override_dir = tmp_path / "atp_source"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(DAVIS_SAMPLE_CSV)

        df = load_davis_cup_matches(2023, 2023, raw_dir_override=override_dir)

        row = df[df["winner_name"] == "Alcaraz"].iloc[0]
        assert row["sets_played"] == 3  # "6-3 6-4 6-2"

    def test_davis_cup_missing_file_raises(self, tmp_path):
        override_dir = tmp_path / "no_files"
        override_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_davis_cup_matches(2023, 2023, raw_dir_override=override_dir)

    def test_davis_cup_no_matching_rows_raises(self, tmp_path):
        # ATP files exist for the range but contain zero tourney_level == "D"
        # rows -- distinct from the "no files at all" case above.
        override_dir = tmp_path / "atp_source_no_davis"
        override_dir.mkdir()
        (override_dir / "2023.csv").write_text(SAMPLE_CSV)  # only "G" level rows
        with pytest.raises(FileNotFoundError):
            load_davis_cup_matches(2023, 2023, raw_dir_override=override_dir)


# ── sets_played normalization ───────────────────────────────────────────────

class TestCountSetsPlayed:
    def test_two_straight_sets(self):
        assert _count_sets_played("6-4 6-2") == 2

    def test_tiebreak_set_counts_as_one(self):
        assert _count_sets_played("7-6(5) 6-4") == 2

    def test_three_sets(self):
        assert _count_sets_played("6-4 3-6 6-2") == 3

    def test_retirement_counts_the_partial_set(self):
        # confirmed decision: a RET set still involved real games played
        assert _count_sets_played("6-3 2-4 RET") == 2

    def test_walkover_is_zero_sets(self):
        assert _count_sets_played("W/O") == 0

    def test_davis_cup_match_tiebreak_bracket_not_counted_as_a_set(self):
        # [10-7] is a 10-point match-tiebreak played instead of a 3rd set, not
        # an actual set -- must not be counted.
        assert _count_sets_played("6-4 5-7 [10-7]") == 2

    def test_davis_cup_partial_match_tiebreak_with_retirement(self):
        assert _count_sets_played("4-6 6-3 [6-7] RET") == 2

    def test_missing_score_is_zero_sets(self):
        assert _count_sets_played(None) == 0
        assert _count_sets_played(float("nan")) == 0


def test_clean_computes_sets_played_for_atp():
    df = _clean(pd.read_csv(StringIO(SAMPLE_CSV)))
    # SAMPLE_CSV rows: "6-3 6-4" and "7-5 6-3" — both 2 sets
    assert list(df["sets_played"]) == [2, 2]


def test_clean_wta_computes_sets_played_from_wsets_lsets():
    df = _clean_wta(_wta_raw(), 2023)
    # WTA_CSV rows: Wsets/Lsets = (2,0), (2,1), (2,0) -> sets_played 2,3,2
    assert list(df["sets_played"]) == [2, 3, 2]


def test_clean_wta_sets_played_defaults_to_zero_when_columns_missing():
    raw = pd.read_csv(StringIO(WTA_CSV))  # original fixture predates Wsets/Lsets addition test
    raw = raw.drop(columns=["Wsets", "Lsets"], errors="ignore")
    df = _clean_wta(raw, 2023)
    assert list(df["sets_played"]) == [0, 0, 0]


def _one_tournament(rows):
    """rows: (round, match_num, winner, loser) all sharing one tourney_date,
    the way Tennismylife dates every match of a tournament."""
    return pd.DataFrame([{
        "tourney_id": "2024-580", "tourney_name": "AO", "surface": "Hard", "tourney_level": "G",
        "tourney_date": 20240115, "match_num": num, "round": rnd,
        "winner_name": w, "loser_name": l, "winner_rank": 1, "loser_rank": 2, "score": "6-3 6-4",
    } for rnd, num, w, l in rows])


def test_clean_orders_a_tournament_by_round_then_match_num():
    # Regression (2026-10-07): every match of a Tennismylife tournament shares
    # tourney_date, and an unstable sort on match_date alone listed later
    # rounds before earlier ones in 99% of ATP tournaments -- so features for
    # an early-round match were computed after that player's later matches.
    raw = _one_tournament([
        ("F", 7, "A", "B"), ("SF", 5, "A", "C"), ("BR", 6, "C", "D"), ("R16", 2, "A", "E"),
        ("RR", 3, "A", "F"), ("QF", 4, "A", "G"), ("R128", 1, "A", "H"), ("R128", 0, "B", "I"),
    ])
    out = _clean(raw, reference_date=date(2024, 2, 1))
    assert out["round"].tolist() == ["R128", "R128", "R16", "RR", "QF", "SF", "BR", "F"]
    assert out.loc[out["round"] == "R128", "match_num"].tolist() == [0, 1]


def test_clean_keeps_unknown_rounds_without_dropping_them():
    raw = _one_tournament([("F", 3, "A", "B"), ("Q1", 1, "C", "D"), (None, 2, "E", "F")])
    out = _clean(raw, reference_date=date(2024, 2, 1))
    assert len(out) == 3
    assert out["round"].tolist()[-1] == "F"


def test_early_round_features_never_see_later_rounds_of_same_tournament():
    from src.backtest.walkforward import build_match_features
    from src.features.engineering import FeatureBuilder
    from src.models.elo import EloSystem

    # Raw rows deliberately listed final-first, as Tennismylife's order ended up.
    raw = _one_tournament([("F", 3, "A", "B"), ("R64", 2, "A", "C"), ("R128", 1, "A", "D")])
    feats = build_match_features(_clean(raw, reference_date=date(2024, 2, 1)), EloSystem(), FeatureBuilder())
    first_round = feats[(~feats["is_mirror"]) & (feats["loser"] == "D")].iloc[0]
    # Neither player has any earlier match, so nothing distinguishes them yet.
    assert first_round["rest_diff"] == 0
    assert first_round["elo_diff"] == 0


def test_clean_tolerates_frames_without_round_or_tourney_id():
    raw = pd.read_csv(StringIO(SAMPLE_CSV)).drop(columns=["round", "tourney_id", "match_num"])
    out = _clean(raw)
    assert out["match_date"].tolist() == sorted(out["match_date"].tolist())

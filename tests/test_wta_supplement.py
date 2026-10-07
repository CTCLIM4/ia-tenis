"""Source-boundary checks for the optional current-year WTA supplement."""
from datetime import date

import pandas as pd

from scripts import download_data
from src.data.loader import load_wta_matches
from src.data.wta_supplement import _carried_rank, load_supplement, resolve_player_name


HEADER = (
    "match_id;date;tournoi;categorie;genre;surface;tour;joueur1;joueur1_id;"
    "joueur2;joueur2_id;vainqueur_id;score\n"
)


def _primary():
    return pd.DataFrame([
        {"match_date": pd.Timestamp("2026-09-25"), "year": 2026,
         "winner_name": "Ruse E.G.", "loser_name": "McNally C.",
         "winner_rank": 45, "loser_rank": 70},
        {"match_date": pd.Timestamp("2026-09-27"), "year": 2026,
         "winner_name": "McNally C.", "loser_name": "Ruse E.G.",
         "winner_rank": 68, "loser_rank": 42},
    ])


def _row(match_id=1, match_date="2026-10-04 14:10:00", tour=6,
         player1="Elena Gabriela Ruse", player2="Caty McNally", winner_id=10):
    return (f'{match_id};"{match_date}";"China Open - Beijing";Masters;wta;dur;{tour};'
            f'"{player1}";10;"{player2}";20;{winner_id};"6-4 6-2"\n')


def test_resolver_handles_compound_names_and_prefers_current_alias():
    stats = {
        "Ruse E.G.": (date(2026, 9, 27), 148),
        "Ruse E-G.": (date(2022, 7, 19), 1),
        "McNally C.": (date(2026, 9, 2), 66),
        "Mcnally C.": (date(2025, 1, 14), 71),
        "Wang Xin.": (date(2026, 9, 25), 236),
        "Wang X.": (date(2026, 10, 1), 7),
        "Wang Xiy.": (date(2026, 9, 1), 155),
    }
    assert resolve_player_name("Elena Gabriela Ruse", stats) == "Ruse E.G."
    assert resolve_player_name("Caty McNally", stats) == "McNally C."
    assert resolve_player_name("Xinyu Wang", stats) == "Wang Xin."
    assert resolve_player_name("Unseen Player", stats) is None


def test_old_rank_is_not_carried_into_new_matches():
    ranks = {"Lazaro A.": (500.0, date(2021, 5, 24))}
    assert pd.isna(_carried_rank("Lazaro A.", date(2026, 10, 1), ranks))
    future_ranks = {"Lazaro A.": (50.0, date(2026, 10, 8))}
    assert pd.isna(_carried_rank("Lazaro A.", date(2026, 10, 1), future_ranks))


def test_resolver_refuses_an_exact_identity_tie():
    stats = {
        "McNally C.": (date(2026, 9, 2), 10),
        "Mcnally C.": (date(2026, 9, 2), 10),
    }
    assert resolve_player_name("Caty McNally", stats) is None


def test_supplement_accepts_only_new_settled_main_draw_matches(tmp_path):
    path = tmp_path / "valuebetennis_2026.csv"
    path.write_text(
        HEADER
        + _row()
        + _row(match_id=1)  # repeated source ID
        + _row(match_id=2, tour=1)  # qualifying
        + _row(match_id=3, match_date="2026-09-27 12:00:00")  # already in primary
        + _row(match_id=4, match_date="2026-10-08 12:00:00"),  # future
        encoding="utf-8",
    )

    extra, eligible, unresolved = load_supplement(path, _primary(), 2026, date(2026, 10, 5))

    assert (eligible, unresolved, len(extra)) == (1, 0, 1)
    assert extra.iloc[0]["winner_name"] == "Ruse E.G."
    assert extra.iloc[0]["loser_name"] == "McNally C."
    assert extra.iloc[0]["winner_rank"] == 42  # latest observed, not stale winner rank 45
    assert extra.iloc[0]["loser_rank"] == 68
    assert extra.iloc[0]["surface"] == "hard"
    assert extra.iloc[0]["sets_played"] == 2


def test_supplement_rejects_low_identity_coverage(tmp_path):
    path = tmp_path / "valuebetennis_2026.csv"
    path.write_text(HEADER + _row() + _row(match_id=2, player1="Unknown Newcomer"), encoding="utf-8")

    extra, eligible, unresolved = load_supplement(path, _primary(), 2026, date(2026, 10, 5))

    assert extra.empty
    assert (eligible, unresolved) == (2, 1)


def test_loader_uses_supplement_from_raw_override(tmp_path):
    (tmp_path / "2026w.csv").write_text(
        "Date,Surface,Winner,Loser,WRank,LRank,Wsets,Lsets\n"
        "25/09/2026,Hard,Ruse E.G.,McNally C.,45,70,2,0\n"
        "27/09/2026,Hard,McNally C.,Ruse E.G.,68,42,2,0\n",
        encoding="utf-8",
    )
    (tmp_path / "valuebetennis_2026.csv").write_text(HEADER + _row(), encoding="utf-8")

    matches = load_wta_matches(2026, 2026, raw_dir_override=tmp_path)

    assert len(matches) == 3
    assert matches["match_date"].max() == pd.Timestamp("2026-10-04")
    assert matches.iloc[-1]["source_match_id"] == 1


def test_supplement_download_preserves_good_file_on_invalid_response(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "WTA_DIR", tmp_path)
    dest = tmp_path / "valuebetennis_2026.csv"
    dest.write_text(HEADER + _row(), encoding="utf-8")
    original = dest.read_bytes()

    class Response:
        content = b"<html>temporarily unavailable</html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download_data.requests, "get", lambda *a, **kw: Response())

    assert download_data.download_wta_supplement(2026) is False
    assert dest.read_bytes() == original


def test_supplement_download_replaces_same_size_update(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "WTA_DIR", tmp_path)
    dest = tmp_path / "valuebetennis_2026.csv"
    previous = (HEADER + _row(match_id=1)).encode()
    updated = (HEADER + _row(match_id=2)).encode()
    assert len(previous) == len(updated)
    dest.write_bytes(previous)

    class Response:
        content = updated

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download_data.requests, "get", lambda *a, **kw: Response())

    assert download_data.download_wta_supplement(2026) is True
    assert dest.read_bytes() == updated
    assert not (tmp_path / "valuebetennis_2026.csv.tmp").exists()

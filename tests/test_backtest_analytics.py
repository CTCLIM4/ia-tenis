"""Tests for src/backtest_analytics.py — Módulo 2 (backtest analytics dashboard)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_analytics import load_resolved_bets

_BETS_HEADER = "tour,match_date,status,result,kelly_a,kelly_b,odds_a,odds_b,ev_a,ev_b,profit"

_BETS_ROWS = [
    # Row A: real, resolved, side A bet, win
    "atp,2026-01-01,ok,A_win,0.05,0.0,2.0,1.9,0.1,-0.05,0.05",
    # Row B: real, resolved, side B bet, win
    "wta,2026-01-02,ok,B_win,0.0,0.03,1.8,2.1,-0.02,0.08,0.03",
    # Row C: excluded — missing Elo
    "atp,2026-01-04,invalid_missing_elo,A_win,0.05,0.0,2.0,1.9,0.1,-0.05,0.05",
    # Row D: excluded — still pending
    "atp,2026-01-05,ok,pending,0.05,0.0,2.0,1.9,0.1,-0.05,",
    # Row E: excluded — test fixture row
    "atp,2026-01-06,test_never_played,excluded,0.0,0.0,1.5,2.5,0.0,0.0,",
]


def _write_bets_csv(tmp_path, rows=_BETS_ROWS):
    path = tmp_path / "value_bets_log.csv"
    path.write_text(_BETS_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


class TestLoadResolvedBets:
    def test_filters_to_status_ok_and_resolved_result(self, tmp_path):
        df = load_resolved_bets(_write_bets_csv(tmp_path))
        assert len(df) == 2
        assert set(df["result"]) == {"A_win", "B_win"}

    def test_derives_bet_side_a(self, tmp_path):
        df = load_resolved_bets(_write_bets_csv(tmp_path))
        row = df[df["result"] == "A_win"].iloc[0]
        assert row["bet_side"] == "a"
        assert row["stake"] == pytest.approx(0.05)
        assert row["odds_taken"] == pytest.approx(2.0)
        assert row["ev_theoretical"] == pytest.approx(0.1)
        assert row["win"] == True  # noqa: E712

    def test_derives_bet_side_b(self, tmp_path):
        df = load_resolved_bets(_write_bets_csv(tmp_path))
        row = df[df["result"] == "B_win"].iloc[0]
        assert row["bet_side"] == "b"
        assert row["stake"] == pytest.approx(0.03)
        assert row["odds_taken"] == pytest.approx(2.1)
        assert row["ev_theoretical"] == pytest.approx(0.08)
        assert row["win"] == True  # noqa: E712

    def test_tour_filter(self, tmp_path):
        df = load_resolved_bets(_write_bets_csv(tmp_path), tour="atp")
        assert len(df) == 1
        assert df.iloc[0]["tour"] == "atp"

    def test_sorted_by_match_date(self, tmp_path):
        rows = list(reversed(_BETS_ROWS[:2]))  # B before A in the file
        df = load_resolved_bets(_write_bets_csv(tmp_path, rows))
        assert list(df["result"]) == ["A_win", "B_win"]

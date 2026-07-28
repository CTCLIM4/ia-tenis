"""Tests for src/backtest_analytics.py — Módulo 2 (backtest analytics dashboard)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_analytics import (
    compute_audit_coverage,
    compute_calibration_metrics,
    compute_financial_metrics,
    compute_risk_metrics,
    load_audit_log,
    load_resolved_bets,
    print_report,
    run_report,
)

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

# Row F: excluded — resolved, status ok, but no edge on either side (both Kelly
# fractions 0), meaning no real stake was ever placed. Reachable via the
# interactive CLI's "Guardar en log? (s/n)" prompt, which defaults to yes with
# no has_value gate (see src/value_analysis.py around line 1114).
_ZERO_STAKE_ROW = "atp,2026-01-07,ok,A_win,0.0,0.0,2.0,1.9,0.0,0.0,0.0"


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

    def test_excludes_zero_stake_rows(self, tmp_path):
        rows = _BETS_ROWS + [_ZERO_STAKE_ROW]
        df = load_resolved_bets(_write_bets_csv(tmp_path, rows))
        assert len(df) == 2
        assert set(df["match_date"].dt.strftime("%Y-%m-%d")) == {"2026-01-01", "2026-01-02"}


def _synthetic_resolved_df():
    """4 resolved bets with hand-computed expected metrics — see design spec
    Task 2/3/4 comments below for the arithmetic behind each assertion."""
    return pd.DataFrame({
        "match_date": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
        ]),
        "profit":         [0.05, 0.03, -0.04, -0.02],
        "stake":          [0.05, 0.03, 0.04, 0.02],
        "ev_theoretical": [0.10, 0.08, 0.05, 0.02],
        "win":            [True, True, False, False],
    })


class TestComputeFinancialMetrics:
    def test_metrics(self):
        df = _synthetic_resolved_df()
        m = compute_financial_metrics(df, bankroll=1000.0)
        # total_profit=0.02, total_stake=0.14 -> roi = 0.02/0.14*100
        assert m["roi_pct"] == pytest.approx(14.285714, rel=1e-4)
        assert m["win_rate_pct"] == pytest.approx(50.0)
        assert m["wins"] == 2
        assert m["losses"] == 2
        assert m["net_profit_usd"] == pytest.approx(20.0)
        assert m["avg_stake_usd"] == pytest.approx(35.0)
        assert m["avg_stake_pct"] == pytest.approx(3.5)
        assert m["num_bets"] == 4

    def test_zero_stake_does_not_divide_by_zero(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2026-01-01"]),
            "profit": [0.0], "stake": [0.0],
            "ev_theoretical": [0.0], "win": [False],
        })
        m = compute_financial_metrics(df, bankroll=1000.0)
        assert m["roi_pct"] == 0.0


class TestComputeRiskMetrics:
    def test_metrics(self):
        df = _synthetic_resolved_df()
        r = compute_risk_metrics(df, bankroll=1000.0)
        # equity = 1000 + cumsum([.05,.03,-.04,-.02])*1000 = [1050,1080,1040,1020]
        # running_max = [1050,1080,1080,1080]; drawdown = [0,0,-40,-60]
        assert r["max_drawdown_usd"] == pytest.approx(-60.0)
        # -60 / 1080 * 100
        assert r["max_drawdown_pct"] == pytest.approx(-5.555556, rel=1e-4)
        # sample variance (ddof=1) of [0.05,0.03,-0.04,-0.02]
        assert r["variance"] == pytest.approx(0.0017667, rel=1e-3)
        # win sequence [T,T,F,F]
        assert r["max_win_streak"] == 2
        assert r["max_loss_streak"] == 2

    def test_streak_breaks_correctly(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime([f"2026-01-0{i}" for i in range(1, 6)]),
            "profit": [0.01, 0.01, 0.01, -0.01, 0.01],
            "stake": [0.01] * 5,
            "ev_theoretical": [0.05] * 5,
            "win": [True, True, True, False, True],
        })
        r = compute_risk_metrics(df, bankroll=1000.0)
        assert r["max_win_streak"] == 3
        assert r["max_loss_streak"] == 1

    def test_single_row_variance_is_zero(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2026-01-01"]),
            "profit": [0.02], "stake": [0.02],
            "ev_theoretical": [0.02], "win": [True],
        })
        r = compute_risk_metrics(df, bankroll=1000.0)
        assert r["variance"] == 0.0

    def test_empty_df_returns_zeros(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime([]),
            "profit": pd.Series(dtype=float),
            "stake": pd.Series(dtype=float),
            "ev_theoretical": pd.Series(dtype=float),
            "win": pd.Series(dtype=bool),
        })
        r = compute_risk_metrics(df, bankroll=1000.0)
        assert r == {
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "variance": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }

    def test_zero_drawdown_when_all_wins(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "profit": [0.02, 0.03, 0.01],
            "stake": [0.02, 0.03, 0.01],
            "ev_theoretical": [0.02, 0.03, 0.01],
            "win": [True, True, True],
        })
        r = compute_risk_metrics(df, bankroll=1000.0)
        assert r["max_drawdown_usd"] == pytest.approx(0.0)
        assert r["max_drawdown_pct"] == pytest.approx(0.0)

    def test_drawdown_tie_break_picks_larger_magnitude_pct(self):
        # peak $1000 -> dip $900 (-$100, -10%); later peak $2000 -> dip $1900
        # (-$100, -5%). Both troughs tie on dollar drawdown; idxmin() picks
        # the first occurrence, whose running_max (peak) is <= the later
        # tied trough's peak (cummax is monotonically non-decreasing), so
        # the earlier/lower-peak trough is always the equal-or-more-severe
        # % reading.
        df = pd.DataFrame({
            "match_date": pd.to_datetime([
                "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
            ]),
            "profit": [0.0, -0.1, 1.0, -0.1],
            "stake": [0.0, 0.1, 1.0, 0.1],
            "ev_theoretical": [0.0, 0.1, 1.0, 0.1],
            "win": [False, False, True, False],
        })
        r = compute_risk_metrics(df, bankroll=1000.0)
        assert r["max_drawdown_usd"] == pytest.approx(-100.0)
        assert r["max_drawdown_pct"] == pytest.approx(-10.0)


class TestComputeCalibrationMetrics:
    def test_metrics(self):
        df = _synthetic_resolved_df()
        c = compute_calibration_metrics(df)
        # ev_theoretical mean = (.10+.08+.05+.02)/4 = 0.0625
        assert c["ev_theoretical_avg"] == pytest.approx(0.0625)
        # real_return per row = profit/stake = [1.0, 1.0, -1.0, -1.0] -> mean 0.0
        assert c["real_return_avg"] == pytest.approx(0.0)
        assert c["difference"] == pytest.approx(-0.0625)


_AUDIT_HEADER = "tour,decision"
_AUDIT_ROWS = [
    "atp,logged", "wta,logged",
    "atp,passed_low_edge",
    "wta,blocked_suspicious", "atp,blocked_suspicious", "atp,blocked_suspicious",
]


def _write_audit_csv(tmp_path):
    path = tmp_path / "prediction_audit_log.csv"
    path.write_text(_AUDIT_HEADER + "\n" + "\n".join(_AUDIT_ROWS) + "\n", encoding="utf-8")
    return str(path)


class TestLoadAuditLog:
    def test_loads_all_rows(self, tmp_path):
        df = load_audit_log(_write_audit_csv(tmp_path))
        assert len(df) == 6


class TestComputeAuditCoverage:
    def test_counts_by_decision(self, tmp_path):
        audit_df = load_audit_log(_write_audit_csv(tmp_path))
        cov = compute_audit_coverage(audit_df)
        assert cov["total"] == 6
        assert cov["by_decision"] == {
            "logged": 2,
            "passed_low_edge": 1,
            "passed_user_declined": 0,
            "blocked_suspicious": 3,
            "invalid_missing_elo": 0,
        }
        assert cov["logged"] == 2
        assert cov["passed"] == 4


class TestPrintReport:
    def test_report_contains_all_sections(self, capsys):
        df = _synthetic_resolved_df()
        financial = compute_financial_metrics(df, bankroll=1000.0)
        risk = compute_risk_metrics(df, bankroll=1000.0)
        calibration = compute_calibration_metrics(df)
        coverage = {
            "total": 6,
            "by_decision": {
                "logged": 2, "passed_low_edge": 1, "passed_user_declined": 0,
                "blocked_suspicious": 3, "invalid_missing_elo": 0,
            },
            "logged": 2, "passed": 4,
        }

        print_report(df, financial, risk, calibration, coverage, bankroll=1000.0)
        out = capsys.readouterr().out

        assert "REPORTE DE RENDIMIENTO Y BANCA" in out
        assert "$1,000.00" in out
        assert "2026-01-01 a 2026-01-04" in out
        assert "METRICAS FINANCIERAS" in out
        assert "METRICAS DE RIESGO" in out
        assert "CALIBRACION" in out
        assert "COBERTURA DEL AUDIT LOG" in out
        assert "Muestra pequena" in out  # N=4 < 30

    def test_report_without_coverage(self, capsys):
        df = _synthetic_resolved_df()
        financial = compute_financial_metrics(df, bankroll=1000.0)
        risk = compute_risk_metrics(df, bankroll=1000.0)
        calibration = compute_calibration_metrics(df)

        print_report(df, financial, risk, calibration, None, bankroll=1000.0)
        out = capsys.readouterr().out

        assert "COBERTURA DEL AUDIT LOG: no disponible" in out


class TestRunReport:
    def test_missing_bets_file(self, tmp_path, capsys):
        run_report(bets_path=str(tmp_path / "nope.csv"), audit_path=str(tmp_path / "nope2.csv"))
        out = capsys.readouterr().out
        assert "No se encontro" in out

    def test_zero_resolved_bets(self, tmp_path, capsys):
        rows = ["atp,2026-01-05,ok,pending,0.05,0.0,2.0,1.9,0.1,-0.05,"]
        bets_path = _write_bets_csv(tmp_path, rows)
        run_report(bets_path=bets_path, audit_path=str(tmp_path / "nope.csv"))
        out = capsys.readouterr().out
        assert "No hay apuestas resueltas todavia" in out

    def test_normal_run_without_audit_file(self, tmp_path, capsys):
        bets_path = _write_bets_csv(tmp_path)  # 2 resolved rows (A_win, B_win)
        run_report(bets_path=bets_path, audit_path=str(tmp_path / "missing_audit.csv"))
        out = capsys.readouterr().out
        assert "REPORTE DE RENDIMIENTO Y BANCA" in out
        assert "Aviso: no se encontro" in out
        assert "COBERTURA DEL AUDIT LOG: no disponible" in out

    def test_normal_run_with_audit_file(self, tmp_path, capsys):
        bets_path = _write_bets_csv(tmp_path)
        audit_path = _write_audit_csv(tmp_path)
        run_report(bets_path=bets_path, audit_path=audit_path, bankroll=500.0, tour="atp")
        out = capsys.readouterr().out
        assert "REPORTE DE RENDIMIENTO Y BANCA" in out
        assert "$500.00" in out
        assert "COBERTURA DEL AUDIT LOG (prediction_audit_log.csv)" in out

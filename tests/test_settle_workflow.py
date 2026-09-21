"""Tests for scripts/settle_workflow.py — semi-automated nightly settlement."""
from __future__ import annotations

import csv
from datetime import date

import scripts.settle_workflow as settle


HEADER = [
    "timestamp", "tour", "tournament", "surface", "match_date", "player_a", "player_b",
    "odds_a", "odds_b", "ev_a", "ev_b", "kelly_a", "kelly_b", "status", "result", "profit",
    "bookmaker_a", "bookmaker_b",
]


def _row(match_date="2026-08-19", player_a="A Player", player_b="B Player",
         odds_a="2.0", odds_b="1.8", kelly_a="0.02", kelly_b="0.0",
         status="ok", result="pending", profit="",
         bookmaker_a="", bookmaker_b=""):
    return {
        "timestamp": "2026-08-19T10:00:00", "tour": "atp", "tournament": "ATP Test Open",
        "surface": "hard", "match_date": match_date, "player_a": player_a, "player_b": player_b,
        "odds_a": odds_a, "odds_b": odds_b, "ev_a": "0.1", "ev_b": "0.1",
        "kelly_a": kelly_a, "kelly_b": kelly_b,
        "status": status, "result": result, "profit": profit,
        "bookmaker_a": bookmaker_a, "bookmaker_b": bookmaker_b,
    }


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


class TestBetSide:
    def test_side_a_when_kelly_a_positive(self):
        assert settle._bet_side(_row(kelly_a="0.02", kelly_b="0.0")) == "a"

    def test_side_b_when_kelly_b_positive(self):
        assert settle._bet_side(_row(kelly_a="0.0", kelly_b="0.014")) == "b"

    def test_side_b_when_both_positive_but_b_is_larger(self):
        # Real case: 2026-09-19 Stearns/Jovic row had both sides showing
        # edge (different bookmakers, betfair vs onexbet) — kelly_a=0.0025,
        # kelly_b=0.0171. The dominant recommended position (6.8x bigger)
        # was on B, not the "a wins if kelly_a>0" default.
        assert settle._bet_side(_row(kelly_a="0.0025", kelly_b="0.0171")) == "b"

    def test_side_a_when_both_positive_but_a_is_larger(self):
        assert settle._bet_side(_row(kelly_a="0.03", kelly_b="0.01")) == "a"


class TestComputeProfit:
    def test_win_on_bet_side_a(self):
        row = _row(odds_a="2.0", kelly_a="0.05", kelly_b="0.0")
        result, profit = settle._compute_profit(row, winner_side="a")
        assert result == "A_win"
        assert profit == 0.05  # 0.05 * (2.0 - 1)

    def test_loss_on_bet_side_a(self):
        row = _row(odds_a="2.0", kelly_a="0.05", kelly_b="0.0")
        result, profit = settle._compute_profit(row, winner_side="b")
        assert result == "B_win"
        assert profit == -0.05

    def test_win_on_bet_side_b(self):
        row = _row(odds_b="3.29", kelly_a="0.0", kelly_b="0.0146")
        result, profit = settle._compute_profit(row, winner_side="b")
        assert result == "B_win"
        assert profit == round(0.0146 * (3.29 - 1), 6)

    def test_loss_on_bet_side_b(self):
        row = _row(odds_b="3.29", kelly_a="0.0", kelly_b="0.0146")
        result, profit = settle._compute_profit(row, winner_side="a")
        assert result == "A_win"
        assert profit == -0.0146


class TestRun:
    def test_settles_only_pending_rows_for_target_date(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [
            _row(match_date="2026-08-19", player_a="Today Pending"),
            _row(match_date="2026-08-18", player_a="Yesterday Pending"),
            _row(match_date="2026-08-19", player_a="Already Settled", result="A_win", profit="0.01"),
        ])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settled = settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        assert len(settled) == 1
        assert settled[0]["player_a"] == "Today Pending"

    def test_skip_leaves_row_pending(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [_row(match_date="2026-08-19")])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: None)
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settled = settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        assert settled == []
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["result"] == "pending"

    def test_writes_result_and_profit_back_to_csv(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [_row(match_date="2026-08-19", odds_a="2.0", kelly_a="0.05")])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["result"] == "A_win"
        assert float(rows[0]["profit"]) == 0.05

    def test_writes_report_file_named_by_target_date(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [_row(match_date="2026-08-19", odds_a="2.0", kelly_a="0.05")])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        report_path = tmp_path / "2026-08-19-jornada-cincinnati.md"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "A Player" in content

    def test_dry_run_does_not_write_csv_or_report_or_push(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [_row(match_date="2026-08-19", odds_a="2.0", kelly_a="0.05")])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        pushed = {}
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: pushed.setdefault("called", True))

        settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path), dry_run=True)

        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["result"] == "pending"
        assert not (tmp_path / "2026-08-19-jornada-cincinnati.md").exists()
        assert pushed == {}

    def test_pushes_csv_and_report_when_something_settled(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [_row(match_date="2026-08-19", odds_a="2.0", kelly_a="0.05")])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        pushed = {}
        monkeypatch.setattr(
            settle, "commit_and_push",
            lambda paths, message: pushed.update(paths=paths, message=message) or True,
        )

        settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        assert str(path) in pushed["paths"]
        assert any("2026-08-19-jornada-cincinnati.md" in p for p in pushed["paths"])

    def test_report_includes_bets_settled_in_an_earlier_run_same_date(self, tmp_path, monkeypatch):
        """Regression: settling the same date in two batches (e.g. some
        matches confirmed same-day, a rain-suspended one confirmed the next
        day) must not make the second run's report silently overwrite the
        first run's — the report should always reflect every bet resolved
        for that match_date, not just the ones settled in this particular
        run."""
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [
            _row(match_date="2026-08-30", player_a="Already Settled Earlier",
                 result="A_win", profit="0.02"),
            _row(match_date="2026-08-30", player_a="Settled This Run",
                 odds_a="2.0", kelly_a="0.05"),
        ])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settle.run(date(2026, 8, 30), log_path=str(path), report_dir=str(tmp_path))

        report_path = tmp_path / "2026-08-30-jornada-cincinnati.md"
        content = report_path.read_text(encoding="utf-8")
        assert "Already Settled Earlier" in content
        assert "Settled This Run" in content

    def test_no_pending_rows_returns_empty_without_writing(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [_row(match_date="2026-08-18")])
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settled = settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        assert settled == []
        assert not (tmp_path / "2026-08-19-jornada-cincinnati.md").exists()


class TestBuildReportBookmakerColumn:
    def test_shows_bookmaker_for_settled_side(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [
            _row(match_date="2026-08-19", odds_a="2.0", kelly_a="0.05", bookmaker_a="bet365"),
        ])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        report_path = tmp_path / "2026-08-19-jornada-cincinnati.md"
        content = report_path.read_text(encoding="utf-8")
        assert "bet365" in content

    def test_blank_when_bookmaker_field_absent(self, tmp_path, monkeypatch):
        # Simulates a pre-migration CSV on disk (written before this task
        # added bookmaker_a/bookmaker_b to HEADER) — the row dicts read back
        # via csv.DictReader genuinely have no bookmaker_a/bookmaker_b keys
        # at all, so _build_report's row.get(...) fallback must not raise.
        OLD_HEADER = [h for h in HEADER if h not in ("bookmaker_a", "bookmaker_b")]
        path = tmp_path / "value_bets_log.csv"
        old_row = {k: v for k, v in _row(match_date="2026-08-19", odds_a="2.0", kelly_a="0.05").items()
                   if k in OLD_HEADER}
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OLD_HEADER)
            writer.writeheader()
            writer.writerow(old_row)
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        report_path = tmp_path / "2026-08-19-jornada-cincinnati.md"
        assert report_path.exists()

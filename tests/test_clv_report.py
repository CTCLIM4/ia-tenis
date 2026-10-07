"""Tests for scripts/clv_report.py on synthetic snapshots + audit rows
(tmp_path only; never reads data/odds_snapshots/ or the real audit log)."""
from __future__ import annotations

import pandas as pd
import pytest

import scripts.clv_report as clv

COLS = ["snapshot_utc", "sport_key", "event_id", "commence_time", "home", "away",
        "bookmaker", "bookmaker_last_update", "price_home", "price_away"]


def _row(t, eid, commence, home, away, bk, ph, pa, sport="tennis_atp_shanghai_masters"):
    return [t, sport, eid, commence, home, away, bk, t, ph, pa]


def _write(tmp_path, rows, runs):
    d = tmp_path / "snaps"
    d.mkdir()
    pd.DataFrame(rows, columns=COLS).to_csv(d / "2026-10.csv", index=False)
    pd.DataFrame({"snapshot_utc": runs}).to_csv(d / "runs.csv", index=False)
    return d


def _audit(**over):
    base = {"timestamp": "2026-10-08T08:00:00", "tour": "atp", "match_date": "2026-10-08",
            "player_a": "Ilia Simakin", "player_b": "Camilo Ugo Carabelli",
            "p_a_cal": 0.60, "p_b_cal": 0.40, "odds_a": 2.0, "odds_b": 1.9}
    base.update(over)
    return pd.DataFrame([base])


@pytest.fixture
def snapshot_dir(tmp_path):
    # Lima 08:00 scan = 13:00 UTC. Market moves the home player (Carabelli)
    # from 0.5 to 0.4 after the scan, i.e. toward Simakin -- the model's side.
    rows = [
        _row("2026-10-08T12:00:00Z", "e1", "2026-10-08T20:00:00Z", "Camilo Ugo Carabelli", "Ilia Simakin", "pin", 2.0, 2.0),
        _row("2026-10-08T12:00:00Z", "e1", "2026-10-08T20:00:00Z", "Camilo Ugo Carabelli", "Ilia Simakin", "b2", 1.9, 1.9),
        _row("2026-10-08T18:00:00Z", "e1", "2026-10-08T20:00:00Z", "Camilo Ugo Carabelli", "Ilia Simakin", "pin", 2.5, 2.5 / 1.5),
        _row("2026-10-08T18:00:00Z", "e1", "2026-10-08T20:00:00Z", "Camilo Ugo Carabelli", "Ilia Simakin", "b2", 2.5, 2.5 / 1.5),
    ]
    return _write(tmp_path, rows, ["2026-10-08T12:00:00Z", "2026-10-08T18:00:00Z"])


def test_fair_at_uses_each_bookmakers_latest_price(snapshot_dir):
    snaps, _ = clv.load_snapshots(snapshot_dir)
    ev = snaps[snaps["event_id"] == "e1"]
    assert clv.fair_at(ev, pd.Timestamp("2026-10-08T13:00:00Z")) == pytest.approx(0.5)
    assert clv.fair_at(ev, pd.Timestamp("2026-10-08T19:00:00Z")) == pytest.approx(0.4)
    assert clv.fair_at(ev, pd.Timestamp("2026-10-08T11:00:00Z")) is None


def test_joins_reversed_orientation_and_orients_lines(snapshot_dir):
    snaps, runs = clv.load_snapshots(snapshot_dir)
    events = clv.event_lines(snaps, runs, 8)
    joined = clv.join_audit(_audit(), events)
    assert len(joined) == 1 and not joined.iloc[0]["a_is_home"]

    s = clv.sides(joined, snaps, runs)
    a = s.iloc[0]  # Simakin, away
    assert a["q_scan"] == pytest.approx(0.5) and a["q_close"] == pytest.approx(0.6)
    slope, lo, hi = clv.line_move_slope(s, n_boot=50)
    assert slope > 0


def test_wrong_tour_or_date_does_not_join(snapshot_dir):
    snaps, runs = clv.load_snapshots(snapshot_dir)
    events = clv.event_lines(snaps, runs, 8)
    assert clv.join_audit(_audit(tour="wta"), events).empty
    assert clv.join_audit(_audit(match_date="2026-10-12"), events).empty


def test_closing_line_requires_a_recent_run(snapshot_dir):
    snaps, runs = clv.load_snapshots(snapshot_dir)
    events = clv.event_lines(snaps, runs, 1)  # last run is 2 h before start
    assert events.iloc[0]["q_close_home"] is None
    assert clv.sides(clv.join_audit(_audit(), events), snaps, runs).empty


def test_rule_picks_clv(snapshot_dir):
    snaps, runs = clv.load_snapshots(snapshot_dir)
    s = clv.sides(clv.join_audit(_audit(), clv.event_lines(snaps, runs, 8)), snaps, runs)
    picks = clv.rule_picks(s)
    # Simakin: p 0.60 vs 1/2.0 -> edge 10% (inside the rule), closes at 0.6
    assert len(picks) == 1
    assert picks.iloc[0]["clv"] == pytest.approx(2.0 * 0.6 - 1)


def test_scan_before_first_run_has_no_scan_line(snapshot_dir):
    snaps, runs = clv.load_snapshots(snapshot_dir)
    joined = clv.join_audit(_audit(timestamp="2026-10-08T06:00:00"), clv.event_lines(snaps, runs, 8))
    assert clv.sides(joined, snaps, runs).empty


def test_main_without_snapshots_exits_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["clv_report", "--snapshot-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        clv.main()
    assert "Sin snapshots" in str(exc.value)

"""Tests for scripts/snapshot_odds.py. Every test mocks fetch_json and
writes to tmp_path; none of them call The Odds API or touch
data/odds_snapshots/."""
from __future__ import annotations

import csv
from datetime import datetime, timezone

import pytest

import scripts.snapshot_odds as snap

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
INDEX = [
    {"key": "tennis_atp_shanghai_masters", "title": "ATP Shanghai Masters"},
    {"key": "tennis_wta_china_open", "title": "WTA China Open"},
    {"key": "soccer_epl", "title": "EPL"},
]


def _event(eid, commence, prices_by_bk):
    return {
        "id": eid, "commence_time": commence, "home_team": "A Player", "away_team": "B Player",
        "bookmakers": [
            {"key": bk, "markets": [{"key": "h2h", "last_update": "2026-10-07T11:00:00Z", "outcomes": [
                {"name": "A Player", "price": ph}, {"name": "B Player", "price": pa}]}]}
            for bk, (ph, pa) in prices_by_bk.items()
        ],
    }


def _fake_api(odds_by_sport, remaining=400, calls=None):
    state = {"remaining": remaining}

    def fetch(url):
        if calls is not None:
            calls.append(url)
        if "/odds/" not in url:
            return INDEX, state["remaining"]
        sport = url.split("/sports/")[1].split("/")[0]
        state["remaining"] -= 1
        return odds_by_sport.get(sport, []), state["remaining"]
    return fetch


def _read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_writes_prematch_rows_and_run_log(monkeypatch, tmp_path):
    odds = {"tennis_atp_shanghai_masters": [
        _event("e1", "2026-10-07T14:00:00Z", {"pinnacle": (1.8, 2.1), "betfair": (1.85, 2.05)}),
        _event("live", "2026-10-07T11:00:00Z", {"pinnacle": (1.2, 4.5)}),  # already started
    ]}
    monkeypatch.setattr(snap, "fetch_json", _fake_api(odds))

    run = snap.take_snapshot("k", tmp_path, "eu", 150, now=NOW)

    rows = _read(tmp_path / "2026-10.csv")
    assert {(r["event_id"], r["bookmaker"]) for r in rows} == {("e1", "pinnacle"), ("e1", "betfair")}
    assert run["events_prematch"] == 1 and run["rows_written"] == 2
    assert run["tournaments_fetched"] == 2  # soccer is never fetched
    runs = _read(tmp_path / "runs.csv")
    assert len(runs) == 1 and runs[0]["credits_remaining"] == "398"


def test_only_changed_prices_are_appended(monkeypatch, tmp_path):
    first = {"tennis_atp_shanghai_masters": [
        _event("e1", "2026-10-07T20:00:00Z", {"pinnacle": (1.8, 2.1), "betfair": (1.85, 2.05)})]}
    monkeypatch.setattr(snap, "fetch_json", _fake_api(first))
    snap.take_snapshot("k", tmp_path, "eu", 150, now=NOW)

    second = {"tennis_atp_shanghai_masters": [
        _event("e1", "2026-10-07T20:00:00Z", {"pinnacle": (1.75, 2.15), "betfair": (1.85, 2.05)})]}
    monkeypatch.setattr(snap, "fetch_json", _fake_api(second))
    run = snap.take_snapshot("k", tmp_path, "eu", 150, now=NOW.replace(hour=18))

    rows = _read(tmp_path / "2026-10.csv")
    assert len(rows) == 3
    assert rows[-1]["bookmaker"] == "pinnacle" and float(rows[-1]["price_home"]) == 1.75
    assert run["rows_written"] == 1
    assert len(_read(tmp_path / "runs.csv")) == 2  # unchanged runs are still logged


def test_previous_month_prices_count_as_already_stored(monkeypatch, tmp_path):
    odds = {"tennis_atp_shanghai_masters": [_event("e1", "2026-11-01T10:00:00Z", {"pinnacle": (1.8, 2.1)})]}
    monkeypatch.setattr(snap, "fetch_json", _fake_api(odds))
    snap.take_snapshot("k", tmp_path, "eu", 150, now=datetime(2026, 10, 31, 20, tzinfo=timezone.utc))
    run = snap.take_snapshot("k", tmp_path, "eu", 150, now=datetime(2026, 11, 1, 2, tzinfo=timezone.utc))
    assert run["rows_written"] == 0
    assert not (tmp_path / "2026-11.csv").exists()


def test_stops_spending_below_min_remaining(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(snap, "fetch_json", _fake_api({}, remaining=151, calls=calls))

    run = snap.take_snapshot("k", tmp_path, "eu", 150, now=NOW)

    odds_calls = [c for c in calls if "/odds/" in c]
    assert len(odds_calls) == 1  # 151 -> 150, then 150 - 1 < 150 stops
    assert run["tournaments_fetched"] == 1 and run["tournaments_skipped_budget"] == 1


def test_each_region_costs_one_credit(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(snap, "fetch_json", _fake_api({}, remaining=152, calls=calls))
    run = snap.take_snapshot("k", tmp_path, "eu,uk,us", 150, now=NOW)
    assert run["tournaments_fetched"] == 0 and run["tournaments_skipped_budget"] == 2


def test_tournament_failure_skips_only_that_tournament(monkeypatch, tmp_path):
    good = _fake_api({"tennis_wta_china_open": [_event("w1", "2026-10-08T05:00:00Z", {"pinnacle": (1.5, 2.6)})]})

    def fetch(url):
        if "shanghai" in url:
            raise snap.urllib.error.URLError("reset")
        return good(url)
    monkeypatch.setattr(snap, "fetch_json", fetch)

    run = snap.take_snapshot("k", tmp_path, "eu", 150, now=NOW)
    assert run["tournaments_fetched"] == 1 and run["rows_written"] == 1


def test_index_failure_raises_after_retries(monkeypatch, tmp_path):
    attempts = []

    def fetch(url):
        attempts.append(url)
        raise snap.urllib.error.URLError("down")
    monkeypatch.setattr(snap, "fetch_json", fetch)
    monkeypatch.setattr(snap.time, "sleep", lambda s: None)

    with pytest.raises(snap.urllib.error.URLError):
        snap.take_snapshot("k", tmp_path, "eu", 150, now=NOW)
    assert len(attempts) == 3
    assert not (tmp_path / "runs.csv").exists()


def test_skips_bookmaker_missing_a_side():
    event = _event("e1", "2026-10-07T20:00:00Z", {"pinnacle": (1.8, 2.1)})
    event["bookmakers"][0]["markets"][0]["outcomes"].pop()
    assert snap.event_rows(event, "tennis_atp_x", "t") == []


class TestMaybeCommit:
    def _repo(self, monkeypatch, tmp_path):
        import subprocess
        import src.git_utils as git_utils

        def git(*args):
            return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True)
        git("init", "-q")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (tmp_path / "other.txt").write_text("base")
        git("add", "other.txt")
        git("commit", "-q", "-m", "base")
        monkeypatch.setattr(git_utils, "_ROOT", tmp_path)
        monkeypatch.setattr(snap, "_ROOT", tmp_path)
        return git

    def test_commits_only_snapshots_leaving_user_staging_alone(self, monkeypatch, tmp_path):
        git = self._repo(monkeypatch, tmp_path)
        out = tmp_path / "data" / "odds_snapshots"
        out.mkdir(parents=True)
        (out / "runs.csv").write_text("x\n")
        (tmp_path / "other.txt").write_text("user work in progress")
        git("add", "other.txt")

        assert snap.maybe_commit(out, 24, now=NOW) is True

        files = git("show", "--name-only", "--format=", "HEAD").stdout.split()
        assert files == ["data/odds_snapshots/runs.csv"]
        assert "M  other.txt" in git("status", "--porcelain").stdout  # still staged, not committed

    def test_throttled_and_noop_without_changes(self, monkeypatch, tmp_path):
        self._repo(monkeypatch, tmp_path)
        out = tmp_path / "data" / "odds_snapshots"
        out.mkdir(parents=True)
        (out / "runs.csv").write_text("x\n")
        assert snap.maybe_commit(out, 24) is True
        (out / "runs.csv").write_text("x\ny\n")
        assert snap.maybe_commit(out, 24) is False  # last commit is seconds old
        assert snap.maybe_commit(out, 0) is True
        assert snap.maybe_commit(out, 0) is False  # nothing changed

"""Tests for src/value_analysis.py — value functions, shrinkage, disambiguation, log."""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.decay import EloHistoryTracker
from src.config import MAX_SUSPICIOUS_EDGE
from src.value_analysis import (
    FEATURE_COLS,
    KELLY_CAP,
    _LOG_FIELDS,
    _build_age_lookup,
    _build_elo_fb,
    _check_staleness,
    _current_age,
    _is_elo_known,
    _resolve_player_name,
    _should_halt_on_suspicious_edge,
    _should_log_prediction,
    apply_shrinkage,
    build_prediction_features,
    calculate_value,
    check_existing_log_entry,
    log_query,
    predict_match,
    update_log_entry,
    LogMatchStatus,
)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _fake_elo(ratings: dict, counts: dict | None = None):
    """Minimal EloSystem stand-in: only .general_ratings and .match_counts needed."""
    return SimpleNamespace(
        general_ratings=ratings,
        match_counts=counts or {k: 1 for k in ratings},
    )


def _make_pred(elo_found_a: bool = True, elo_found_b: bool = True) -> dict:
    """Minimal prediction dict for log_query tests."""
    return {
        "player_a":      "Player A",
        "player_b":      "Player B",
        "p_a_raw":       0.60,
        "p_a_cal":       0.58,
        "p_b_raw":       0.40,
        "p_b_cal":       0.42,
        "rank_a":        1,
        "rank_b":        2,
        "rank_a_source": "manual",
        "rank_b_source": "manual",
        "features":      {c: 0.0 for c in FEATURE_COLS},
        "elo_a":         1750.0 if elo_found_a else 1500.0,
        "elo_b":         1650.0 if elo_found_b else 1500.0,
        "elo_found_a":   elo_found_a,
        "elo_found_b":   elo_found_b,
    }


def _val(p: float = 0.58, odds: float = 1.90) -> dict:
    return calculate_value(p, odds)


def _read_log(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── 1. calculate_value ─────────────────────────────────────────────────────────

class TestCalculateValue:
    def test_positive_edge_fields(self):
        v = calculate_value(0.60, 2.00)
        assert v["implied_prob"]    == pytest.approx(0.50)
        assert v["edge"]            == pytest.approx(0.10)
        assert v["ev"]              == pytest.approx(0.20)
        assert v["has_value"]       is True

    def test_kelly_capped_at_kelly_cap(self):
        # raw kelly = (0.80 - 1/3) / 2 ≈ 0.233 → must be capped
        v = calculate_value(0.80, 3.00)
        assert v["kelly_fraction"] == pytest.approx(KELLY_CAP)
        assert v["has_value"] is True

    def test_kelly_uncapped_when_below_cap(self):
        # raw kelly = 0.02 / 1.0 = 0.02 < 0.05
        v = calculate_value(0.52, 2.00)
        assert v["kelly_fraction"] == pytest.approx(0.02)
        assert v["kelly_fraction"] <  KELLY_CAP

    def test_break_even_no_value(self):
        v = calculate_value(0.50, 2.00)
        assert v["edge"]            == pytest.approx(0.0)
        assert v["ev"]              == pytest.approx(0.0)
        assert v["kelly_fraction"]  == 0.0
        assert v["has_value"]       is False

    def test_negative_edge_no_value(self):
        v = calculate_value(0.45, 2.00)
        assert v["edge"]           < 0
        assert v["kelly_fraction"] == 0.0
        assert v["has_value"]      is False

    def test_ev_matches_formula(self):
        # EV = p*(odds-1) - (1-p)
        p, o = 0.55, 2.10
        assert calculate_value(p, o)["ev"] == pytest.approx(p * (o - 1) - (1 - p))

    def test_implied_prob_is_reciprocal_of_odds(self):
        assert calculate_value(0.60, 1.80)["implied_prob"] == pytest.approx(1 / 1.80)


# ── 2. apply_shrinkage ─────────────────────────────────────────────────────────

class TestApplyShrinkage:
    def test_p095_shrinks_to_092(self):
        # 0.90 + (0.95-0.90)*0.40 = 0.90 + 0.02 = 0.92
        assert apply_shrinkage(0.95) == pytest.approx(0.92)

    def test_p099_shrinks_to_0936(self):
        # 0.90 + (0.99-0.90)*0.40 = 0.90 + 0.036 = 0.936
        assert apply_shrinkage(0.99) == pytest.approx(0.936)

    def test_p005_shrinks_to_008(self):
        # symmetric: 0.10 - (0.10-0.05)*0.40 = 0.10 - 0.02 = 0.08
        assert apply_shrinkage(0.05) == pytest.approx(0.08)

    def test_p001_shrinks_to_0064(self):
        # 0.10 - (0.10-0.01)*0.40 = 0.10 - 0.036 = 0.064
        assert apply_shrinkage(0.01) == pytest.approx(0.064)

    @pytest.mark.parametrize("p", [0.10, 0.20, 0.30, 0.50, 0.70, 0.89, 0.90])
    def test_inside_bounds_unchanged(self, p):
        assert apply_shrinkage(p) == pytest.approx(p)

    def test_boundary_090_not_shrunk(self):
        # p > 0.90 triggers shrinkage; p == 0.90 must not
        assert apply_shrinkage(0.90) == pytest.approx(0.90)

    def test_boundary_010_not_shrunk(self):
        assert apply_shrinkage(0.10) == pytest.approx(0.10)


class TestCheckStaleness:
    """_check_staleness delegates to src.data.staleness.evaluate_staleness for
    the OK/WARNING/CRITICAL decision (see tests/test_staleness.py for full
    rule coverage) — these tests only check the CLI's print-adaptation of
    that report, using an explicit reference_date so behavior doesn't depend
    on which calendar day the suite happens to run on."""

    def test_no_warning_within_regular_week_tolerance(self, capsys):
        ref = date(2026, 7, 20)
        recent_date = ref - timedelta(days=2)
        _check_staleness("wta", recent_date, reference_date=ref)
        assert capsys.readouterr().out == ""

    def test_warns_when_beyond_regular_week_critical_threshold(self, capsys):
        ref = date(2026, 7, 20)
        old_date = ref - timedelta(days=45)
        _check_staleness("atp", old_date, reference_date=ref)
        out = capsys.readouterr().out
        assert "CRITICO" in out
        assert "ATP" in out
        assert str(old_date) in out

    def test_warning_level_prints_advertencia_tag(self, capsys):
        ref = date(2026, 7, 20)
        old_date = ref - timedelta(days=5)  # WARNING band: >3, <=7
        _check_staleness("atp", old_date, reference_date=ref)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" in out

    def test_no_warning_within_offseason_tolerance(self, capsys):
        ref = date(2026, 12, 20)
        last_match = ref - timedelta(days=40)  # Nov -> end of season, within 45d grace
        _check_staleness("atp", last_match, reference_date=ref)
        assert capsys.readouterr().out == ""

    def test_live_tournament_mode_warns_sooner_than_regular_week(self, capsys):
        ref = date(2026, 7, 20)
        last_match = ref - timedelta(days=2)  # OK under regular-week, WARNING under live mode
        _check_staleness("wta", last_match, live_tournament_mode=True, reference_date=ref)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" in out

    def test_defaults_reference_date_to_lima_today(self, monkeypatch, capsys):
        fixed_today = date(2026, 7, 30)
        monkeypatch.setattr("src.value_analysis.lima_today", lambda: fixed_today)
        _check_staleness("atp", fixed_today, reference_date=None)
        assert capsys.readouterr().out == ""


# ── 3. _resolve_player_name / disambiguation ───────────────────────────────────

class TestResolvePlayerName:
    """Pin the expected resolved key for known ambiguous WTA cases.

    These are regression tests: a future change to the disambiguation heuristic
    must not silently re-route Karolina Pliskova to the 3-match legacy key
    'Pliskova Kar.' or route Anastasia Rodionova to 'Rodionova A.' (75 matches)
    instead of the canonical 'Rodionova An.' (104 matches).
    """

    @pytest.fixture()
    def pliskova_elo(self):
        """Five Pliskova variants as seen in the live WTA Elo index."""
        return _fake_elo(
            ratings={
                "Pliskova Ka.":  1553.0,   # Karolina — canonical, 602 matches
                "Pliskova Kar.": 1521.1,   # legacy 3-match artifact from 2008
                "Pliskova K.":   1487.3,   # 1-match ambiguous entry
                "Pliskova Kr.":  1443.3,   # Kristyna — canonical, 227 matches
                "Pliskova Kri.": 1481.1,   # 1-match artifact
            },
            counts={
                "Pliskova Ka.":  602,
                "Pliskova Kar.": 3,
                "Pliskova K.":   1,
                "Pliskova Kr.":  227,
                "Pliskova Kri.": 1,
            },
        )

    @pytest.fixture()
    def rodionova_elo(self):
        return _fake_elo(
            ratings={
                "Rodionova A.":  1600.0,   # Anastasia early career, 75 matches
                "Rodionova An.": 1580.0,   # Anastasia canonical, 104 matches
                "Rodionova Ar.": 1560.0,   # Arina, 47 matches
            },
            counts={
                "Rodionova A.":  75,
                "Rodionova An.": 104,
                "Rodionova Ar.": 47,
            },
        )

    @pytest.fixture()
    def simple_elo(self):
        return _fake_elo(
            ratings={"Swiatek I.": 1850.0, "Sabalenka A.": 1750.0},
            counts={"Swiatek I.": 244, "Sabalenka A.": 353},
        )

    # Pliskova ----------------------------------------------------------------

    def test_karolina_resolves_to_ka(self, pliskova_elo):
        assert _resolve_player_name("Karolina Pliskova", pliskova_elo) == "Pliskova Ka."

    def test_kristyna_resolves_to_kr(self, pliskova_elo):
        assert _resolve_player_name("Kristyna Pliskova", pliskova_elo) == "Pliskova Kr."

    def test_exact_key_returns_unchanged(self, pliskova_elo):
        # Direct key bypass: no WTA translation needed
        assert _resolve_player_name("Pliskova Ka.", pliskova_elo) == "Pliskova Ka."

    def test_kar_artifact_never_beats_ka(self, pliskova_elo):
        # Core regression: 'Pliskova Kar.' (3 matches) must lose to 'Pliskova Ka.' (602).
        # If this fails, the heuristic changed in a way that breaks Karolina's lookup.
        result = _resolve_player_name("Karolina Pliskova", pliskova_elo)
        assert result != "Pliskova Kar.", (
            "Pliskova Kar. (3 matches, 2008 artifact) must not beat Pliskova Ka. (602)"
        )

    # Rodionova ---------------------------------------------------------------

    def test_anastasia_resolves_to_an(self, rodionova_elo):
        # An. (104 matches) beats A. (75 matches) for the name 'Anastasia'.
        assert _resolve_player_name("Anastasia Rodionova", rodionova_elo) == "Rodionova An."

    def test_anastasia_not_routed_to_a_dot(self, rodionova_elo):
        # Regression: 'Rodionova A.' (75) must not win over 'Rodionova An.' (104).
        result = _resolve_player_name("Anastasia Rodionova", rodionova_elo)
        assert result != "Rodionova A.", (
            "Rodionova A. (75 matches) must not beat Rodionova An. (104) for 'Anastasia'"
        )

    # Standard single-key WTA names -------------------------------------------

    def test_iga_swiatek_resolves_to_wta_key(self, simple_elo):
        assert _resolve_player_name("Iga Swiatek", simple_elo) == "Swiatek I."

    def test_aryna_sabalenka_resolves_to_wta_key(self, simple_elo):
        assert _resolve_player_name("Aryna Sabalenka", simple_elo) == "Sabalenka A."

    def test_unknown_player_returns_original_name(self, simple_elo):
        assert _resolve_player_name("Ghost Player", simple_elo) == "Ghost Player"

    # _is_elo_known -----------------------------------------------------------

    def test_is_known_true_for_wta_full_name(self, simple_elo):
        assert _is_elo_known("Iga Swiatek", simple_elo) is True

    def test_is_known_false_for_unknown_player(self, simple_elo):
        assert _is_elo_known("Ghost Player", simple_elo) is False


# ── 4 & 5. log_query status + bug regression ──────────────────────────────────

@pytest.fixture()
def log_path(tmp_path, monkeypatch):
    """Redirect LOG_PATH to a temp file so tests don't touch the real log."""
    path = tmp_path / "value_bets_log.csv"
    monkeypatch.setattr("src.value_analysis.LOG_PATH", path)
    return path


class TestLogQueryStatus:

    def test_valid_prediction_saves_status_ok(self, log_path):
        pred = _make_pred(elo_found_a=True, elo_found_b=True)
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        rows = _read_log(log_path)
        assert len(rows) == 1
        assert rows[0]["status"] == "ok"

    def test_missing_elo_a_saves_invalid(self, log_path):
        pred = _make_pred(elo_found_a=False, elo_found_b=True)
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        assert _read_log(log_path)[0]["status"] == "invalid_missing_elo"

    def test_missing_elo_b_saves_invalid(self, log_path):
        pred = _make_pred(elo_found_a=True, elo_found_b=False)
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        assert _read_log(log_path)[0]["status"] == "invalid_missing_elo"

    def test_missing_both_elo_saves_invalid(self, log_path):
        pred = _make_pred(elo_found_a=False, elo_found_b=False)
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        assert _read_log(log_path)[0]["status"] == "invalid_missing_elo"

    def test_bug_regression_default_elo_1500_never_saves_ok(self, log_path):
        """Regression for the original bug.

        Before the fix, a player not in the Elo index silently used the default
        rating (1500.0) and the row was saved as status='ok', contaminating the log
        with predictions based on missing data.  After the fix, elo_found_b=False
        must always produce status='invalid_missing_elo'.
        """
        pred = _make_pred(elo_found_a=True, elo_found_b=False)
        pred["elo_b"] = 1500.0  # exactly the default — this was the original symptom
        log_query("atp", "Wimbledon", "grass", date(2026, 7, 1),
                  "Jannik Sinner", "Valentin Royer", pred,
                  _val(), _val(), 1.20, 5.00)
        row = _read_log(log_path)[0]
        assert row["status"] != "ok", (
            "A prediction with elo=1500 (default) must never be logged as status='ok'"
        )
        assert row["status"] == "invalid_missing_elo"

    def test_multiple_rows_each_get_correct_status(self, log_path):
        pred_ok  = _make_pred(True, True)
        pred_bad = _make_pred(False, True)
        v = _val()
        log_query("atp", "T", "hard", date(2026, 7, 1), "A", "B", pred_ok,  v, v, 1.9, 2.1)
        log_query("atp", "T", "hard", date(2026, 7, 1), "A", "B", pred_bad, v, v, 1.9, 2.1)
        rows = _read_log(log_path)
        assert rows[0]["status"] == "ok"
        assert rows[1]["status"] == "invalid_missing_elo"

    def test_status_field_present_in_every_row(self, log_path):
        for found in [(True, True), (True, False), (False, True)]:
            pred = _make_pred(*found)
            log_query("atp", "T", "hard", date(2026, 7, 1),
                      "A", "B", pred, _val(), _val(), 1.9, 2.1)
        rows = _read_log(log_path)
        assert len(rows) == 3
        assert all("status" in r for r in rows)

    def test_bookmaker_a_and_b_recorded_when_provided(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10,
                  bookmaker_a="bet365", bookmaker_b="pinnacle")
        row = _read_log(log_path)[0]
        assert row["bookmaker_a"] == "bet365"
        assert row["bookmaker_b"] == "pinnacle"

    def test_bookmaker_defaults_to_empty_string(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        row = _read_log(log_path)[0]
        assert row["bookmaker_a"] == ""
        assert row["bookmaker_b"] == ""


class TestCheckExistingLogEntry:
    """Prompted by 2026-09-19: daily_scanner.py had no protection against
    re-logging the same still-pending match across separate scan runs --
    each re-log would double-count that position's stake/P&L once the
    match settles. check_existing_log_entry lets a caller distinguish a
    genuinely new match from a re-scan of one already logged."""

    def test_value_bets_log_allows_new_match(self, log_path):
        status, existing = check_existing_log_entry(
            "atp", "Test", "Player A", "Player B", date(2026, 7, 1), 1.90,
        )
        assert status == LogMatchStatus.NEW
        assert existing is None

    def test_value_bets_log_duplicate_detection(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)

        status, existing = check_existing_log_entry(
            "atp", "Test", "Player A", "Player B", date(2026, 7, 1), 1.90,
        )
        assert status == LogMatchStatus.DUPLICATE
        assert existing is not None

    def test_duplicate_detection_within_2pct_tolerance(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 3.80, 2.10)

        # 3.87 is a 1.8% move from 3.80 -- inside tolerance, still a dup.
        status, _ = check_existing_log_entry(
            "atp", "Test", "Player A", "Player B", date(2026, 7, 1), 3.87,
        )
        assert status == LogMatchStatus.DUPLICATE

    def test_value_bets_log_allows_different_odds(self, log_path):
        """Real regression case: Stearns vs Jovic odds moved 3.80 -> 3.90
        (2.6%) between two separate scans -- a legitimate update, not a
        duplicate to silently skip."""
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 3.80, 2.10)

        status, existing = check_existing_log_entry(
            "atp", "Test", "Player A", "Player B", date(2026, 7, 1), 3.90,
        )
        assert status == LogMatchStatus.UPDATE
        assert existing is not None

    def test_odds_b_move_also_triggers_update(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)

        # odds_a unchanged, but odds_b moved > 2% -- still a real update.
        status, _ = check_existing_log_entry(
            "atp", "Test", "Player A", "Player B", date(2026, 7, 1), 1.90,
            odds_b=2.30,
        )
        assert status == LogMatchStatus.UPDATE

    def test_value_bets_log_ignores_resolved_match(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        rows = _read_log(log_path)
        rows[0]["result"] = "A_win"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        status, existing = check_existing_log_entry(
            "atp", "Test", "Player A", "Player B", date(2026, 7, 1), 1.90,
        )
        assert status == LogMatchStatus.RESOLVED
        assert existing is not None

    def test_different_match_date_is_a_new_match(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)

        status, _ = check_existing_log_entry(
            "atp", "Test", "Player A", "Player B", date(2026, 7, 2), 1.90,
        )
        assert status == LogMatchStatus.NEW

    def test_different_tournament_is_a_new_match(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test Open", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)

        status, _ = check_existing_log_entry(
            "atp", "Other Open", "Player A", "Player B", date(2026, 7, 1), 1.90,
        )
        assert status == LogMatchStatus.NEW


class TestUpdateLogEntry:
    def test_updates_existing_row_odds_in_place(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 3.80, 2.10)

        updated = update_log_entry(
            "atp", "Test", "hard", date(2026, 7, 1),
            "Player A", "Player B", pred, _val(), _val(), 3.90, 2.10,
        )

        assert updated is True
        rows = _read_log(log_path)
        assert len(rows) == 1
        assert rows[0]["odds_a"] == "3.9"

    def test_does_not_duplicate_the_row(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 3.80, 2.10)

        update_log_entry(
            "atp", "Test", "hard", date(2026, 7, 1),
            "Player A", "Player B", pred, _val(), _val(), 3.90, 2.10,
        )

        assert len(_read_log(log_path)) == 1

    def test_returns_false_when_no_matching_row(self, log_path):
        pred = _make_pred()
        updated = update_log_entry(
            "atp", "Test", "hard", date(2026, 7, 1),
            "Player A", "Player B", pred, _val(), _val(), 3.90, 2.10,
        )
        assert updated is False

    def test_leaves_other_rows_untouched(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        log_query("wta", "Other", "hard", date(2026, 7, 2),
                  "Player C", "Player D", pred, _val(), _val(), 2.50, 1.60)

        update_log_entry(
            "atp", "Test", "hard", date(2026, 7, 1),
            "Player A", "Player B", pred, _val(), _val(), 2.10, 1.80,
        )

        rows = _read_log(log_path)
        assert len(rows) == 2
        other = next(r for r in rows if r["player_a"] == "Player C")
        assert other["odds_a"] == "2.5"


class TestLogQueryOddsSource:
    def test_odds_source_defaults_to_manual(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        row = _read_log(log_path)[0]
        assert row["odds_a_source"] == "manual"
        assert row["odds_b_source"] == "manual"

    def test_odds_source_records_auto_when_provided(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10,
                  odds_a_source="auto", odds_b_source="auto")
        row = _read_log(log_path)[0]
        assert row["odds_a_source"] == "auto"
        assert row["odds_b_source"] == "auto"

    def test_migrates_old_header_and_preserves_old_row(self, log_path):
        """Regression: a pre-existing log written under the old header (no
        odds_a_source/odds_b_source) must be migrated in place, not silently
        misaligned, the next time log_query() appends a row."""
        old_fields = [f for f in _LOG_FIELDS
                      if f not in ("odds_a_source", "odds_b_source",
                                   "bookmaker_a", "bookmaker_b")]
        old_row = {
            "timestamp": "2026-07-01T16:27:16",
            "tour": "wta",
            "tournament": "Wimbledon TEST",
            "surface": "grass",
            "match_date": "2026-07-01",
            "player_a": "Aryna Sabalenka",
            "player_b": "Iga Swiatek",
            "rank_a": "1",
            "rank_a_source": "manual",
            "rank_b": "2",
            "rank_b_source": "manual",
            "p_a_raw": "0.3214",
            "p_a_cal": "0.3214",
            "p_b_raw": "0.6786",
            "p_b_cal": "0.6786",
            "odds_a": "2.1",
            "odds_b": "1.8",
            "implied_a": "0.4762",
            "implied_b": "0.5556",
            "edge_a": "-0.1548",
            "ev_a": "-0.3251",
            "kelly_a": "0.0",
            "edge_b": "0.1231",
            "ev_b": "0.2215",
            "kelly_b": "0.05",
            "shrinkage_applied": "False",
            "status": "ok",
            "result": "pending",
            "profit": "",
            "elo_diff": "-105.1327",
            "elo_prob": "0.3532",
            "rank_diff": "1.0",
            "form_diff": "-0.4",
            "surface_form_diff": "-0.1",
            "h2h_rate": "0.3333",
            "rest_diff": "1.0",
        }
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=old_fields)
            writer.writeheader()
            writer.writerow(old_row)

        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 2),
                  "Player C", "Player D", pred, _val(), _val(), 1.5, 2.5,
                  odds_a_source="auto", odds_b_source="auto")

        rows = _read_log(log_path)
        assert len(rows) == 2

        migrated = rows[0]
        assert migrated["player_a"] == "Aryna Sabalenka"
        assert migrated["odds_a"] == "2.1"
        assert migrated["odds_b"] == "1.8"
        assert migrated["odds_a_source"] == "manual"
        assert migrated["odds_b_source"] == "manual"
        assert migrated["bookmaker_a"] == "pinnacle"
        assert migrated["bookmaker_b"] == "pinnacle"

        new_row = rows[1]
        assert new_row["odds_a"] == "1.5"
        assert new_row["odds_b"] == "2.5"
        assert new_row["odds_a_source"] == "auto"
        assert new_row["odds_b_source"] == "auto"


class TestTryAutoOdds:
    """Regression: try_auto_odds must pass allowed_bookmakers (Optional[set[str]])
    to get_match_odds, not a bare string — see Task 14 fix."""

    def test_passes_set_or_none_not_a_bare_string(self, monkeypatch):
        import src.value_analysis as va
        captured = {}

        def fake_get_match_odds(tour, player_a, player_b, api_key, allowed_bookmakers, cache_minutes):
            captured["allowed_bookmakers"] = allowed_bookmakers
            return None

        monkeypatch.setattr(va, "get_match_odds", fake_get_match_odds)
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")
        monkeypatch.delenv("ODDS_API_BOOKMAKER", raising=False)

        va.try_auto_odds("atp", "Player A", "Player B")

        assert captured["allowed_bookmakers"] is None or isinstance(captured["allowed_bookmakers"], set)

    def test_explicit_bookmaker_env_collapses_to_single_item_set(self, monkeypatch):
        import src.value_analysis as va
        captured = {}

        def fake_get_match_odds(tour, player_a, player_b, api_key, allowed_bookmakers, cache_minutes):
            captured["allowed_bookmakers"] = allowed_bookmakers
            return None

        monkeypatch.setattr(va, "get_match_odds", fake_get_match_odds)
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")
        monkeypatch.setenv("ODDS_API_BOOKMAKER", "pinnacle")

        va.try_auto_odds("atp", "Player A", "Player B")

        assert captured["allowed_bookmakers"] == {"pinnacle"}


class TestCacheRoundTripsLastMatchDate:
    def test_last_match_date_round_trips(self, tmp_path, monkeypatch):
        import src.value_analysis as va
        monkeypatch.setattr(va, "_CACHE_DIR", tmp_path)

        fake_elo = _fake_elo({"A": 1500.0})
        fake_fb = SimpleNamespace()
        fake_clf = SimpleNamespace()
        rank_lookup = {"A": 10}
        last_match_date = date(2025, 12, 22)
        fake_tracker = EloHistoryTracker()
        for v in [1500, 1510, 1520, 1530, 1540]:
            fake_tracker.record("A", "clay", v)
        age_lookup = {"A": (28.5, date(2025, 12, 1))}

        va._save_cache(
            "atp", fake_elo, fake_fb, fake_clf, rank_lookup, last_match_date,
            fake_tracker, age_lookup,
        )
        cached = va._load_cache("atp")

        assert cached is not None
        elo, fb, clf, rank_lkp, cached_date, tracker, age_lkp = cached
        assert cached_date == last_match_date
        assert rank_lkp == rank_lookup
        # Pickle round-trips always produce a new object — compare behavior,
        # not identity: the recorded snapshot must survive the round trip.
        assert tracker.rolling_elo("A", "clay") == fake_tracker.rolling_elo("A", "clay")
        assert age_lkp == age_lookup

    def test_missing_last_match_date_treated_as_cache_miss(self, tmp_path, monkeypatch):
        import pickle
        from datetime import datetime
        import src.value_analysis as va
        monkeypatch.setattr(va, "_CACHE_DIR", tmp_path)

        # Simulate a cache file written before this feature existed: no
        # "last_match_date" key in the payload.
        old_payload = {
            "elo":         _fake_elo({"A": 1500.0}),
            "fb":          SimpleNamespace(),
            "clf":         SimpleNamespace(),
            "rank_lookup": {"A": 10},
            "timestamp":   datetime.now(),
            "tour":        "atp",
        }
        path = tmp_path / "atp.pkl"
        with open(path, "wb") as f:
            pickle.dump(old_payload, f)

        assert va._load_cache("atp") is None


class TestGetLastMatchDate:
    def test_returns_last_match_date_from_cache(self, tmp_path, monkeypatch):
        import src.value_analysis as va
        monkeypatch.setattr(va, "_CACHE_DIR", tmp_path)

        last_match_date = date(2025, 12, 22)
        va._save_cache(
            "atp", _fake_elo({"A": 1500.0}), SimpleNamespace(), SimpleNamespace(),
            {"A": 10}, last_match_date, EloHistoryTracker(), {},
        )

        assert va.get_last_match_date("atp") == last_match_date

    def test_returns_none_when_no_cache_exists(self, tmp_path, monkeypatch):
        import src.value_analysis as va
        monkeypatch.setattr(va, "_CACHE_DIR", tmp_path)
        assert va.get_last_match_date("wta") is None


class TestBuildAgeLookup:
    def test_returns_most_recent_age_and_date_per_player(self):
        df = pd.DataFrame([
            {"winner_name": "A", "winner_age": 28.0, "loser_name": "B", "loser_age": 24.0,
             "match_date": pd.Timestamp("2023-01-01")},
            {"winner_name": "B", "winner_age": 24.5, "loser_name": "A", "loser_age": 28.5,
             "match_date": pd.Timestamp("2023-06-01")},
        ])
        lookup = _build_age_lookup(df)
        assert lookup["A"] == (28.5, date(2023, 6, 1))
        assert lookup["B"] == (24.5, date(2023, 6, 1))

    def test_empty_when_no_age_columns_present(self):
        # WTA-shaped data: no winner_age/loser_age columns at all.
        df = pd.DataFrame([
            {"winner_name": "A", "loser_name": "B", "match_date": pd.Timestamp("2023-01-01")},
        ])
        assert _build_age_lookup(df) == {}


class _StopEarly(Exception):
    """Raised by a fake loader right after capturing its call args, to skip
    the rest of _build_elo_fb (Elo/FeatureBuilder replay) -- these tests
    only care about which loader gets dispatched to for a given tour."""


class TestBuildEloFbDavisDispatch:
    def test_davis_tour_calls_load_davis_cup_matches_not_atp_or_wta(self, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("wrong loader called for tour='davis'")

        monkeypatch.setattr("src.data.loader.load_atp_matches", _boom)
        monkeypatch.setattr("src.data.loader.load_wta_matches", _boom)

        captured = {}

        def _fake_davis_loader(start_year, end_year, raw_dir_override=None):
            captured["start_year"] = start_year
            raise _StopEarly

        monkeypatch.setattr("src.data.loader.load_davis_cup_matches", _fake_davis_loader)

        with pytest.raises(_StopEarly):
            _build_elo_fb("davis")

        assert captured["start_year"] == 1981


class TestCurrentAge:
    def test_extrapolates_forward_from_observed_date(self):
        lookup = {"A": (30.0, date(2026, 1, 1))}
        # 365 days later -> ~+1 year
        result = _current_age("A", lookup, date(2027, 1, 1))
        assert abs(result - 31.0) < 0.01

    def test_returns_none_for_unknown_player(self):
        assert _current_age("Ghost", {}, date(2026, 1, 1)) is None


class TestBuildPredictionFeaturesDecay:
    def test_includes_neutral_decay_keys_by_default(self):
        elo = _fake_elo({"A": 1500.0, "B": 1500.0})
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.5, "h2h_matches": 0, "h2h_trend": 0.0, "rest_days": 14.0,
                "momentum_3": 0.5, "momentum_5": 0.5, "surface_win_rate_trend": 0.0,
            },
            match_dates=lambda p: [],
            workload_history=lambda p: [],
            last_surface_and_date=lambda p: None,
        )
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5

        feats = build_prediction_features(elo, fb, "A", "B", "clay", date(2026, 7, 21))
        assert feats["rolling_elo_diff"] == 0.0
        assert feats["age_multiplier_diff"] == 0.0
        assert feats["rust_factor_diff"] == 0.0
        assert feats["adjusted_elo_diff"] == 0.0

    def test_penalizes_older_rustier_player_in_adjusted_elo_diff(self):
        elo = _fake_elo({"Veteran": 1700.0, "Young": 1500.0})
        elo.get_effective_rating = lambda p, s: 1700.0 if p == "Veteran" else 1500.0
        elo.expected_score = lambda a, b: 1.0 / (1.0 + 10 ** (-(a - b) / 400))
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.5, "h2h_matches": 0, "h2h_trend": 0.0, "rest_days": 14.0,
                "momentum_3": 0.5, "momentum_5": 0.5, "surface_win_rate_trend": 0.0,
            },
            match_dates=lambda p: [] if p == "Young" else [date(2018, 1, 1)],
            workload_history=lambda p: [],
            last_surface_and_date=lambda p: None,
        )

        feats = build_prediction_features(
            elo, fb, "Veteran", "Young", "clay", date(2026, 7, 21),
            age_a=41, age_b=24,
        )
        # Raw elo_diff favors the veteran, but adjusted_elo_diff should not
        assert feats["elo_diff"] == 200.0
        assert feats["adjusted_elo_diff"] < feats["elo_diff"]


class TestPredictMatchH2hWiring:
    def test_h2h_rate_reaches_lr_input_vector(self):
        """Proves h2h_rate is positionally wired into the LR input, the same
        way TestPredictMatchAgeLookup proves it for adjusted_elo_diff."""
        elo = _fake_elo({"A": 1500.0, "B": 1500.0})
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.9, "h2h_matches": 5, "h2h_trend": 0.0, "rest_days": 14.0,
                "momentum_3": 0.5, "momentum_5": 0.5, "surface_win_rate_trend": 0.0,
            },
            match_dates=lambda p: [],
            workload_history=lambda p: [],
            last_surface_and_date=lambda p: None,
        )

        class _StubClf:
            def predict_proba(self, X):
                import numpy as _np
                idx = FEATURE_COLS.index("h2h_rate")
                h2h = X[0][idx]
                return _np.array([[1 - h2h, h2h]])

        pred = predict_match(
            elo, fb, _StubClf(),
            "A", "B", "clay", date(2026, 7, 21),
            rank_lookup=None, age_lookup=None,
        )
        assert pred["p_a_raw"] == pytest.approx(0.9)


class TestPredictMatchMatchCounts:
    """predict_match must expose each player's elo.match_counts so callers
    can flag low-sample predictions (see src/config.MIN_MATCHES_THRESHOLD)
    without re-resolving player names themselves."""

    def _stub_fb(self):
        return SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.5, "h2h_matches": 0, "h2h_trend": 0.0, "rest_days": 14.0,
                "momentum_3": 0.5, "momentum_5": 0.5, "surface_win_rate_trend": 0.0,
            },
            match_dates=lambda p: [],
            workload_history=lambda p: [],
            last_surface_and_date=lambda p: None,
        )

    def _neutral_clf(self):
        class _StubClf:
            def predict_proba(self, X):
                return np.array([[0.5, 0.5]])
        return _StubClf()

    def test_returns_match_counts_for_both_players(self):
        elo = _fake_elo({"A": 1500.0, "B": 1500.0}, counts={"A": 302, "B": 14})
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5

        pred = predict_match(
            elo, self._stub_fb(), self._neutral_clf(),
            "A", "B", "hard", date(2026, 8, 11),
            rank_lookup=None, age_lookup=None,
        )
        assert pred["matches_a"] == 302
        assert pred["matches_b"] == 14

    def test_unknown_player_reports_zero_matches(self):
        elo = _fake_elo({"A": 1500.0}, counts={"A": 5})
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5

        pred = predict_match(
            elo, self._stub_fb(), self._neutral_clf(),
            "A", "Ghost Player", "hard", date(2026, 8, 11),
            rank_lookup=None, age_lookup=None,
        )
        assert pred["matches_a"] == 5
        assert pred["matches_b"] == 0


class TestPredictMatchFatigueWiring:
    def test_fatigue_multiplier_diff_reaches_lr_input_vector(self):
        elo = _fake_elo({"A": 1500.0, "B": 1500.0})
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.5, "h2h_matches": 0, "h2h_trend": 0.0, "rest_days": 14.0,
                "momentum_3": 0.5, "momentum_5": 0.5, "surface_win_rate_trend": 0.0,
            },
            match_dates=lambda p: [],
            workload_history=lambda p: (
                [(date(2026, 7, 18), 3), (date(2026, 7, 19), 2), (date(2026, 7, 20), 3)]
                if p == "A" else []
            ),
            last_surface_and_date=lambda p: None,
        )

        class _StubClf:
            def predict_proba(self, X):
                import numpy as _np
                idx = FEATURE_COLS.index("fatigue_multiplier_diff")
                diff = X[0][idx]
                # A is heavily loaded (fatigue_multiplier_diff < 0), B is fresh
                # -> lower P(A wins) than a neutral coin flip would give.
                p = 0.5 + diff  # arbitrary monotonic mapping, just needs to move away from 0.5
                return _np.array([[1 - p, p]])

        pred = predict_match(
            elo, fb, _StubClf(),
            "A", "B", "clay", date(2026, 7, 21),
            rank_lookup=None, age_lookup=None,
        )
        assert pred["p_a_raw"] < 0.5  # A's heavier recent workload should pull this below neutral


class TestPredictMatchSurfaceTransitionWiring:
    def test_surface_transition_multiplier_diff_reaches_lr_input_vector(self):
        elo = _fake_elo({"A": 1500.0, "B": 1500.0})
        elo.get_effective_rating = lambda p, s: 1500.0
        elo.expected_score = lambda a, b: 0.5
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.5, "h2h_matches": 0, "h2h_trend": 0.0, "rest_days": 14.0,
                "momentum_3": 0.5, "momentum_5": 0.5, "surface_win_rate_trend": 0.0,
            },
            match_dates=lambda p: [],
            workload_history=lambda p: [],
            last_surface_and_date=lambda p: (
                (date(2026, 7, 18), "clay") if p == "A" else None
            ),
        )

        class _StubClf:
            def predict_proba(self, X):
                import numpy as _np
                idx = FEATURE_COLS.index("surface_transition_multiplier_diff")
                diff = X[0][idx]
                p = 0.5 + diff
                return _np.array([[1 - p, p]])

        pred = predict_match(
            elo, fb, _StubClf(),
            "A", "B", "grass", date(2026, 7, 21),
            rank_lookup=None, age_lookup=None,
        )
        assert pred["p_a_raw"] < 0.5  # A just switched clay->grass 3 days ago


class TestPredictMatchAgeLookup:
    def test_auto_fills_age_from_lookup_and_lowers_veterans_win_prob(self):
        """An aging, rusty veteran with a big historical Elo edge should get
        a materially lower win probability than raw Elo alone implies, once
        age_lookup supplies his age and the model can apply the decay
        features — the Wawrinka-vs-Burruchaga scenario this feature set
        exists for."""
        elo = _fake_elo({"Veteran": 1700.0, "Young": 1500.0})
        elo.get_effective_rating = lambda p, s: 1700.0 if p == "Veteran" else 1500.0
        elo.expected_score = lambda a, b: 1.0 / (1.0 + 10 ** (-(a - b) / 400))
        fb = SimpleNamespace(
            get_features=lambda p, o, s, d: {
                "recent_win_rate": 0.5, "recent_win_rate_surface": 0.5,
                "h2h_win_rate": 0.5, "h2h_matches": 0, "h2h_trend": 0.0, "rest_days": 14.0,
                "momentum_3": 0.5, "momentum_5": 0.5, "surface_win_rate_trend": 0.0,
            },
            match_dates=lambda p: [] if p == "Young" else [date(2018, 1, 1)],
            workload_history=lambda p: [],
            last_surface_and_date=lambda p: None,
        )

        class _StubClf:
            """Returns P(Veteran wins) driven mostly by adjusted_elo_diff,
            so the test observes the decay features actually reaching the
            LR input vector rather than asserting on internal feature dicts."""
            def predict_proba(self, X):
                import numpy as _np
                idx = FEATURE_COLS.index("adjusted_elo_diff")
                adjusted_diff = X[0][idx]
                p = 1.0 / (1.0 + 10 ** (-adjusted_diff / 400))
                return _np.array([[1 - p, p]])

        age_lookup = {"Veteran": (41.0, date(2026, 7, 21))}

        pred = predict_match(
            elo, fb, _StubClf(),
            "Veteran", "Young", "clay", date(2026, 7, 21),
            rank_lookup=None, age_lookup=age_lookup,
        )
        # Raw Elo alone (200pp gap) would give the veteran ~76%; the decayed
        # adjusted_elo_diff must pull that down substantially.
        assert pred["p_a_raw"] < 0.60


class TestShouldLogPrediction:
    """Regression coverage for the CLI flow bug: a missing-Elo prediction
    must still be logged (as status='invalid_missing_elo' via log_query)
    when the user opts in — the old elif branch made that path
    unreachable, silently breaking the documented invalid-status feature."""

    def test_suspicious_edge_never_logs_even_on_yes(self):
        assert _should_log_prediction(suspicious=True, save_response="s") is False

    def test_suspicious_edge_never_logs_regardless_of_response_case(self):
        assert _should_log_prediction(suspicious=True, save_response="") is False

    def test_logs_on_affirmative_answer(self):
        assert _should_log_prediction(suspicious=False, save_response="s") is True

    def test_skips_on_negative_answer(self):
        assert _should_log_prediction(suspicious=False, save_response="n") is False

    def test_empty_answer_counts_as_affirmative(self):
        assert _should_log_prediction(suspicious=False, save_response="") is True

    def test_response_is_case_insensitive(self):
        assert _should_log_prediction(suspicious=False, save_response="S") is True
        assert _should_log_prediction(suspicious=False, save_response="N") is False


class TestShouldHaltOnSuspiciousEdge:
    def test_true_when_edge_above_threshold_and_flag_on(self):
        val_a = {"edge": 0.15}
        val_b = {"edge": -0.05}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is True

    def test_false_when_flag_off_even_if_edge_high(self):
        val_a = {"edge": 0.20}
        val_b = {"edge": -0.05}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=False) is False

    def test_false_when_edge_below_threshold(self):
        val_a = {"edge": 0.05}
        val_b = {"edge": 0.02}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is False

    def test_false_exactly_at_threshold(self):
        val_a = {"edge": MAX_SUSPICIOUS_EDGE}
        val_b = {"edge": 0.0}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is False

    def test_checks_both_sides_takes_max(self):
        val_a = {"edge": -0.5}
        val_b = {"edge": 0.11}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is True


class TestTrainLRPipeline:
    """_train_lr must scale features the same way walk_forward_backtest does
    (StandardScaler fit on the training data, applied before LogisticRegression)
    — see project_pending_deferred_items.md item 3: the backtest was scaled
    first, leaving production predictions on an unscaled model, an asymmetry
    this closes."""

    @staticmethod
    def _write_synthetic_features_csv(tmp_path: Path, tour: str = "atp", n: int = 200) -> Path:
        rng = np.random.default_rng(7)
        elo_diff = rng.normal(0, 200, n)
        df = pd.DataFrame({
            "elo_diff":            elo_diff,
            "elo_prob":            1 / (1 + 10 ** (-elo_diff / 400)),
            "rank_diff":           rng.normal(0, 50, n),
            "form_diff":           rng.uniform(-0.5, 0.5, n),
            "surface_form_diff":   rng.uniform(-0.5, 0.5, n),
            "h2h_rate":            rng.uniform(0.3, 0.7, n),
            "rest_diff":           rng.normal(0, 5, n),
            "rolling_elo_diff":    rng.normal(0, 30, n),
            "age_multiplier_diff": rng.uniform(-0.3, 0.3, n),
            "rust_factor_diff":    rng.uniform(-0.5, 0.5, n),
            "fatigue_multiplier_diff": rng.uniform(-0.15, 0.15, n),
            "surface_transition_multiplier_diff": rng.uniform(-0.10, 0.10, n),
            "adjusted_elo_diff":   elo_diff * rng.uniform(0.6, 1.0, n),
            "h2h_trend":           rng.uniform(-0.3, 0.3, n),
            "momentum_3_diff":     rng.uniform(-0.5, 0.5, n),
            "momentum_5_diff":     rng.uniform(-0.5, 0.5, n),
            "surface_win_rate_trend_diff": rng.uniform(-0.5, 0.5, n),
            "outcome":             1,
        })
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        path = processed_dir / f"{tour}_features.csv"
        df.to_csv(path, index=False)
        return path

    def test_returns_pipeline_with_scaler_and_logistic_regression(self, tmp_path, monkeypatch):
        import src.value_analysis as va
        monkeypatch.setattr(va, "_DATA_DIR", tmp_path)
        self._write_synthetic_features_csv(tmp_path)

        clf = va._train_lr("atp")

        assert isinstance(clf, Pipeline)
        assert list(clf.named_steps.keys()) == ["standardscaler", "logisticregression"]
        assert isinstance(clf.named_steps["standardscaler"], StandardScaler)
        assert isinstance(clf.named_steps["logisticregression"], LogisticRegression)

    def test_predict_proba_returns_valid_probabilities(self, tmp_path, monkeypatch):
        import src.value_analysis as va
        monkeypatch.setattr(va, "_DATA_DIR", tmp_path)
        self._write_synthetic_features_csv(tmp_path)

        clf = va._train_lr("atp")
        X = np.array([[100.0, 0.7, 20.0, 0.1, 0.05, 0.6, 1.0, 10.0, 0.0, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0]])
        probs = clf.predict_proba(X)

        assert probs.shape == (1, 2)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-8)
        assert 0.0 <= probs[0, 1] <= 1.0

    def test_predictions_are_deterministic_across_calls(self, tmp_path, monkeypatch):
        import src.value_analysis as va
        monkeypatch.setattr(va, "_DATA_DIR", tmp_path)
        self._write_synthetic_features_csv(tmp_path)

        clf = va._train_lr("atp")
        X = np.array([[50.0, 0.6, 10.0, 0.2, 0.1, 0.55, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0]])
        p1 = clf.predict_proba(X)
        p2 = clf.predict_proba(X)
        np.testing.assert_array_equal(p1, p2)

    def test_scaling_actually_applied_matches_manual_pipeline(self, tmp_path, monkeypatch):
        """Proves the pipeline genuinely scales (not a no-op wrapper):
        predictions must match a manually-built StandardScaler +
        LogisticRegression fit on the same data with the same random_state —
        this is the assertion that actually fails against the old unscaled
        _train_lr, unlike the determinism/shape checks above which would
        pass either way."""
        import src.value_analysis as va
        monkeypatch.setattr(va, "_DATA_DIR", tmp_path)
        path = self._write_synthetic_features_csv(tmp_path)

        clf = va._train_lr("atp")

        df = pd.read_csv(path)
        mirror = df.copy()
        for col in va._MIRROR_FLIP_COLS:
            mirror[col] = -mirror[col]
        mirror["elo_prob"] = 1 - mirror["elo_prob"]
        mirror["h2h_rate"] = 1 - mirror["h2h_rate"]
        mirror["outcome"] = 0
        full = pd.concat([df, mirror], ignore_index=True)

        manual_scaler = StandardScaler()
        X_scaled = manual_scaler.fit_transform(full[va.FEATURE_COLS].fillna(0).values)
        manual_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        manual_lr.fit(X_scaled, full["outcome"].values)

        X_query = np.array([[30.0, 0.55, 5.0, 0.05, 0.02, 0.5, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0]])
        pipeline_probs = clf.predict_proba(X_query)
        manual_probs = manual_lr.predict_proba(manual_scaler.transform(X_query))

        np.testing.assert_allclose(pipeline_probs, manual_probs, atol=1e-10)

    def test_features_path_override_takes_precedence_over_default_location(self, tmp_path, monkeypatch):
        """Proves the (currently unused by any caller) features_path param
        actually overrides the default data/processed/{tour}_features.csv
        lookup — the seam a later task in the snapshot plan wires up."""
        import src.value_analysis as va
        monkeypatch.setattr(va, "_DATA_DIR", tmp_path / "does_not_exist")
        custom_path = self._write_synthetic_features_csv(tmp_path, tour="atp")

        clf = va._train_lr("atp", features_path=custom_path)

        assert isinstance(clf, Pipeline)


class TestLoadModelSnapshot:
    """load_model(snapshot=...) pins loading to data/snapshots/{id}/ instead
    of live data, uses a separate never-expiring cache file, and skips the
    staleness check (a deliberately old pinned dataset isn't 'stale')."""

    _ATP_ROW = (
        "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
        "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
        "winner_ht,winner_ioc,winner_age,winner_rank,winner_rank_points,"
        "loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,"
        "loser_ioc,loser_age,loser_rank,loser_rank_points,score,best_of,round,minutes\n"
        "2020-1,AO,Hard,128,G,20200115,1,101,,,PlayerA,R,188,USA,25.0,1,10000,"
        "102,,,PlayerB,R,190,USA,26.0,2,9000,6-3 6-4,3,R32,85\n"
    )

    @staticmethod
    def _write_features_csv(path):
        rng = np.random.default_rng(11)
        n = 300
        elo_diff = rng.normal(0, 150, n)
        pd.DataFrame({
            "year": rng.integers(2000, 2015, n),
            "elo_diff": elo_diff,
            "elo_prob": 1 / (1 + 10 ** (-elo_diff / 400)),
            "rank_diff": rng.normal(0, 50, n),
            "form_diff": rng.uniform(-0.5, 0.5, n),
            "surface_form_diff": rng.uniform(-0.5, 0.5, n),
            "h2h_rate": rng.uniform(0.3, 0.7, n),
            "rest_diff": rng.normal(0, 5, n),
            "rolling_elo_diff": rng.normal(0, 30, n),
            "age_multiplier_diff": rng.uniform(-0.3, 0.3, n),
            "rust_factor_diff": rng.uniform(-0.5, 0.5, n),
            "fatigue_multiplier_diff": rng.uniform(-0.15, 0.15, n),
            "surface_transition_multiplier_diff": rng.uniform(-0.10, 0.10, n),
            "adjusted_elo_diff": elo_diff * rng.uniform(0.6, 1.0, n),
            "h2h_trend": rng.uniform(-0.3, 0.3, n),
            "momentum_3_diff": rng.uniform(-0.5, 0.5, n),
            "momentum_5_diff": rng.uniform(-0.5, 0.5, n),
            "surface_win_rate_trend_diff": rng.uniform(-0.5, 0.5, n),
            "outcome": 1,
            "is_mirror": False,
        }).to_csv(path, index=False)

    @pytest.fixture()
    def fake_snapshot(self, tmp_path, monkeypatch):
        import src.data.snapshots as snap_mod
        import src.value_analysis as va
        monkeypatch.setattr(snap_mod, "SNAPSHOT_ROOT", tmp_path / "snapshots")
        monkeypatch.setattr(va, "_CACHE_DIR", tmp_path / "model_cache")

        snapshot_dir = tmp_path / "snapshots" / "2020-01-01"
        raw_dir = snapshot_dir / "raw" / "tennis_atp_tml"
        processed_dir = snapshot_dir / "processed"
        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)

        (raw_dir / "2020.csv").write_text(self._ATP_ROW)
        self._write_features_csv(processed_dir / "atp_features.csv")
        (snapshot_dir / "metadata.json").write_text("{}")
        return "2020-01-01"

    def test_builds_from_snapshot_and_caches_separately(self, fake_snapshot):
        import src.value_analysis as va
        elo, fb, clf, rank_lookup, elo_tracker, age_lookup = va.load_model("atp", snapshot=fake_snapshot)

        assert "PlayerA" in elo.general_ratings
        assert va._cache_path("atp", fake_snapshot).exists()
        assert not va._cache_path("atp").exists()  # live cache untouched

    def test_snapshot_cache_never_expires(self, fake_snapshot):
        import pickle
        from datetime import datetime, timedelta
        import src.value_analysis as va

        va.load_model("atp", snapshot=fake_snapshot)  # first build, writes cache
        path = va._cache_path("atp", fake_snapshot)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        payload["timestamp"] = datetime.now() - timedelta(days=999)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        result = va._load_cache("atp", fake_snapshot)
        assert result is not None  # not treated as expired

    def test_skips_staleness_check_under_snapshot(self, fake_snapshot, capsys):
        import src.value_analysis as va
        va.load_model("atp", snapshot=fake_snapshot)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" not in out
        assert "CRITICO" not in out

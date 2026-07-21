"""Tests for src/value_analysis.py — value functions, shrinkage, disambiguation, log."""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import pandas as pd

from src.features.decay import EloHistoryTracker
from src.value_analysis import (
    FEATURE_COLS,
    KELLY_CAP,
    STALENESS_WARNING_DAYS,
    SUSPICIOUS_EDGE_THRESHOLD,
    _LOG_FIELDS,
    _build_age_lookup,
    _check_staleness,
    _current_age,
    _is_elo_known,
    _resolve_player_name,
    _should_halt_on_suspicious_edge,
    apply_shrinkage,
    build_prediction_features,
    calculate_value,
    log_query,
    predict_match,
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
    def test_warns_when_data_older_than_threshold(self, capsys):
        old_date = date.today() - timedelta(days=STALENESS_WARNING_DAYS + 15)
        _check_staleness("atp", old_date)
        out = capsys.readouterr().out
        assert "ADVERTENCIA" in out
        assert "ATP" in out
        assert str(old_date) in out

    def test_no_warning_when_data_recent(self, capsys):
        recent_date = date.today() - timedelta(days=5)
        _check_staleness("wta", recent_date)
        assert capsys.readouterr().out == ""

    def test_no_warning_exactly_at_threshold(self, capsys):
        boundary_date = date.today() - timedelta(days=STALENESS_WARNING_DAYS)
        _check_staleness("atp", boundary_date)
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
        old_fields = [f for f in _LOG_FIELDS if f not in ("odds_a_source", "odds_b_source")]
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

        new_row = rows[1]
        assert new_row["odds_a"] == "1.5"
        assert new_row["odds_b"] == "2.5"
        assert new_row["odds_a_source"] == "auto"
        assert new_row["odds_b_source"] == "auto"


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
                "h2h_win_rate": 0.5, "h2h_matches": 0, "rest_days": 14.0,
            },
            match_dates=lambda p: [],
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
                "h2h_win_rate": 0.5, "h2h_matches": 0, "rest_days": 14.0,
            },
            match_dates=lambda p: [] if p == "Young" else [date(2018, 1, 1)],
        )

        feats = build_prediction_features(
            elo, fb, "Veteran", "Young", "clay", date(2026, 7, 21),
            age_a=41, age_b=24,
        )
        # Raw elo_diff favors the veteran, but adjusted_elo_diff should not
        assert feats["elo_diff"] == 200.0
        assert feats["adjusted_elo_diff"] < feats["elo_diff"]


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
                "h2h_win_rate": 0.5, "h2h_matches": 0, "rest_days": 14.0,
            },
            match_dates=lambda p: [] if p == "Young" else [date(2018, 1, 1)],
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
        val_a = {"edge": SUSPICIOUS_EDGE_THRESHOLD}
        val_b = {"edge": 0.0}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is False

    def test_checks_both_sides_takes_max(self):
        val_a = {"edge": -0.5}
        val_b = {"edge": 0.11}
        assert _should_halt_on_suspicious_edge(val_a, val_b, halt_on_suspicious=True) is True

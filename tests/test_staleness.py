"""Tests for src/data/staleness.py — context-aware dataset freshness checks.

Every test pins an explicit reference_date so behavior doesn't depend on
which calendar day the suite happens to run on (off-season vs. regular-week
rules differ by month, so date.today() would make these tests flaky).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.staleness import (
    LIVE_CRITICAL_DAYS,
    LIVE_WARNING_DAYS,
    OFFSEASON_TOLERANCE_DAYS,
    REGULAR_CRITICAL_DAYS,
    REGULAR_WARNING_DAYS,
    StalenessLevel,
    check_dataset_staleness,
    evaluate_staleness,
)


def _df(last_match: date) -> pd.DataFrame:
    """Minimal match dataframe whose newest row is `last_match`."""
    return pd.DataFrame({"match_date": pd.to_datetime([last_match])})


# ── Semana regular de torneos ───────────────────────────────────────────────

class TestRegularWeek:
    REF = date(2026, 7, 20)  # mid-July: clearly outside the off-season window

    def test_ok_within_warning_threshold(self):
        df = _df(self.REF - timedelta(days=REGULAR_WARNING_DAYS))
        report = check_dataset_staleness(df, reference_date=self.REF)
        assert report.level == StalenessLevel.OK

    def test_warning_just_above_warning_threshold(self):
        df = _df(self.REF - timedelta(days=REGULAR_WARNING_DAYS + 1))
        report = check_dataset_staleness(df, reference_date=self.REF)
        assert report.level == StalenessLevel.WARNING

    def test_warning_at_critical_boundary(self):
        df = _df(self.REF - timedelta(days=REGULAR_CRITICAL_DAYS))
        report = check_dataset_staleness(df, reference_date=self.REF)
        assert report.level == StalenessLevel.WARNING

    def test_critical_beyond_critical_threshold(self):
        df = _df(self.REF - timedelta(days=REGULAR_CRITICAL_DAYS + 1))
        report = check_dataset_staleness(df, reference_date=self.REF)
        assert report.level == StalenessLevel.CRITICAL

    def test_rule_label_is_regular_week(self):
        df = _df(self.REF - timedelta(days=1))
        report = check_dataset_staleness(df, reference_date=self.REF)
        assert report.rule == "regular_week"


# ── Off-season (diciembre a inicios de enero) ───────────────────────────────

class TestOffSeason:
    REF = date(2026, 12, 20)  # December -> off-season calendar window

    def test_ok_within_tolerance(self):
        # Last match in November -> counts as "end of regular season".
        last_match = self.REF - timedelta(days=OFFSEASON_TOLERANCE_DAYS)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=self.REF)
        assert report.level == StalenessLevel.OK
        assert report.rule == "off_season"

    def test_critical_beyond_tolerance(self):
        last_match = self.REF - timedelta(days=OFFSEASON_TOLERANCE_DAYS + 1)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=self.REF)
        assert report.level == StalenessLevel.CRITICAL
        assert report.rule == "off_season"

    def test_early_january_counts_as_offseason(self):
        ref = date(2027, 1, 10)
        last_match = date(2026, 12, 1)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.rule == "off_season"
        assert report.level == StalenessLevel.OK

    def test_mid_january_no_longer_offseason(self):
        # Past "inicios de enero" -> falls back to regular-week rules.
        ref = date(2027, 1, 20)
        last_match = date(2027, 1, 5)  # 15 days stale
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.rule == "regular_week"
        assert report.level == StalenessLevel.CRITICAL  # 15 > REGULAR_CRITICAL_DAYS

    def test_offseason_grace_requires_last_match_near_season_end(self):
        # December reference, but the last match is from mid-season (August)
        # — a genuinely abnormal gap, not a normal off-season lull. Must NOT
        # receive the 45-day grace.
        ref = date(2026, 12, 20)
        last_match = date(2026, 8, 1)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.rule == "regular_week"
        assert report.level == StalenessLevel.CRITICAL


# ── Modo torneo en vivo ──────────────────────────────────────────────────────

class TestLiveTournamentMode:
    REF = date(2026, 7, 20)

    def test_ok_same_day(self):
        df = _df(self.REF)
        report = check_dataset_staleness(df, reference_date=self.REF, live_tournament_mode=True)
        assert report.level == StalenessLevel.OK

    def test_ok_at_warning_boundary(self):
        df = _df(self.REF - timedelta(days=LIVE_WARNING_DAYS))
        report = check_dataset_staleness(df, reference_date=self.REF, live_tournament_mode=True)
        assert report.level == StalenessLevel.OK

    def test_warning_just_above_warning_boundary(self):
        df = _df(self.REF - timedelta(days=LIVE_WARNING_DAYS + 1))
        report = check_dataset_staleness(df, reference_date=self.REF, live_tournament_mode=True)
        assert report.level == StalenessLevel.WARNING

    def test_warning_at_critical_boundary(self):
        df = _df(self.REF - timedelta(days=LIVE_CRITICAL_DAYS))
        report = check_dataset_staleness(df, reference_date=self.REF, live_tournament_mode=True)
        assert report.level == StalenessLevel.WARNING

    def test_critical_beyond_critical_boundary(self):
        df = _df(self.REF - timedelta(days=LIVE_CRITICAL_DAYS + 1))
        report = check_dataset_staleness(df, reference_date=self.REF, live_tournament_mode=True)
        assert report.level == StalenessLevel.CRITICAL

    def test_rule_label_is_live_tournament(self):
        df = _df(self.REF - timedelta(days=1))
        report = check_dataset_staleness(df, reference_date=self.REF, live_tournament_mode=True)
        assert report.rule == "live_tournament"

    def test_live_mode_overrides_offseason_leniency(self):
        # Even inside the off-season calendar window, live_tournament_mode
        # must apply the strict rule, not the lenient 45-day grace.
        ref = date(2026, 12, 20)
        last_match = ref - timedelta(days=5)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref, live_tournament_mode=True)
        assert report.rule == "live_tournament"
        assert report.level == StalenessLevel.CRITICAL


# ── check_dataset_staleness: DataFrame handling, defaults, errors ──────────

class TestCheckDatasetStalenessDataFrameHandling:
    def test_days_stale_matches_actual_gap(self):
        ref = date(2026, 7, 20)
        last_match = date(2026, 7, 10)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.days_stale == 10

    def test_uses_max_match_date_when_multiple_rows(self):
        ref = date(2026, 7, 20)
        df = pd.DataFrame({
            "match_date": pd.to_datetime([date(2026, 7, 1), date(2026, 7, 18), date(2026, 7, 5)]),
        })
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.days_stale == 2  # from 2026-07-18, not the other rows

    def test_defaults_reference_date_to_today(self):
        today = date.today()
        df = _df(today)
        report = check_dataset_staleness(df)
        assert report.days_stale == 0

    def test_raises_on_empty_dataframe(self):
        df = pd.DataFrame({"match_date": pd.to_datetime([])})
        with pytest.raises(ValueError):
            check_dataset_staleness(df)


# ── Future-date guard (regression) ──────────────────────────────────────────

class TestFutureDateGuard:
    """A last_match_date after reference_date is never 'fresh' — it's invalid
    data, not a signal of freshness. Regression for a real bug found
    2026-07-24: the WTA raw file (data/raw/tennis_wta_tduk/2026w.xlsx) had
    one row's date typo'd as 2029-07-20 instead of 2026-07-20 (Iasi Open
    final, Sherif def. Badosa), which made
    (reference_date - last_match_date).days come out negative — and the
    plain regular_week classification (days_stale > threshold) let a
    negative number sail through as OK, masking the data corruption instead
    of flagging it."""

    def test_future_last_match_date_is_critical(self):
        ref = date(2026, 7, 24)
        last_match = date(2029, 7, 20)  # the actual corrupt value found
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.level == StalenessLevel.CRITICAL

    def test_future_last_match_date_rule_label(self):
        ref = date(2026, 7, 24)
        last_match = date(2029, 7, 20)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.rule == "invalid_future_date"

    def test_future_date_guard_overrides_live_tournament_mode(self):
        ref = date(2026, 7, 24)
        last_match = ref + timedelta(days=1)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref, live_tournament_mode=True)
        assert report.rule == "invalid_future_date"
        assert report.level == StalenessLevel.CRITICAL

    def test_future_date_guard_overrides_offseason_leniency(self):
        ref = date(2026, 12, 20)
        last_match = ref + timedelta(days=1)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.rule == "invalid_future_date"
        assert report.level == StalenessLevel.CRITICAL

    def test_same_day_match_is_not_flagged_as_future(self):
        # last_match_date == reference_date (days_stale=0) is valid, not future.
        ref = date(2026, 7, 24)
        df = _df(ref)
        report = check_dataset_staleness(df, reference_date=ref)
        assert report.rule != "invalid_future_date"
        assert report.level == StalenessLevel.OK

    def test_message_flags_invalid_data_not_freshness(self):
        ref = date(2026, 7, 24)
        last_match = date(2029, 7, 20)
        df = _df(last_match)
        report = check_dataset_staleness(df, reference_date=ref)
        lowered = report.message.lower()
        assert "invalid" in lowered or "futur" in lowered


# ── evaluate_staleness: low-level pure function ─────────────────────────────

class TestEvaluateStaleness:
    def test_matches_check_dataset_staleness_for_same_inputs(self):
        ref = date(2026, 7, 20)
        last_match = date(2026, 7, 5)
        via_df = check_dataset_staleness(_df(last_match), reference_date=ref)
        direct = evaluate_staleness(last_match, ref, live_tournament_mode=False)
        assert via_df == direct

    def test_ok_message_does_not_look_alarming(self):
        ref = date(2026, 7, 20)
        report = evaluate_staleness(ref, ref, live_tournament_mode=False)
        assert report.message
        assert "CRITIC" not in report.message.upper()

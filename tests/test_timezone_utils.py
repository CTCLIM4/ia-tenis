"""Tests for src/data/timezone_utils.py — explicit America/Lima conversion
for timestamps sourced from The Odds API (UTC)."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.data.timezone_utils import LIMA_TZ, lima_today, to_lima


class TestToLima:
    def test_converts_aware_utc_datetime_crossing_day_boundary(self):
        dt = datetime(2026, 7, 27, 2, 0, tzinfo=timezone.utc)
        result = to_lima(dt)
        assert result == datetime(2026, 7, 26, 21, 0, tzinfo=LIMA_TZ)

    def test_offset_is_minus_five_in_july(self):
        dt = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        assert to_lima(dt).utcoffset().total_seconds() / 3600 == -5

    def test_offset_is_minus_five_in_january(self):
        # Lima does not observe DST — offset must stay -05:00 year-round.
        dt = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert to_lima(dt).utcoffset().total_seconds() / 3600 == -5

    def test_raises_on_naive_datetime(self):
        naive = datetime(2026, 7, 27, 2, 0)
        with pytest.raises(ValueError):
            to_lima(naive)


class TestLimaToday:
    def test_returns_a_date_instance(self):
        assert isinstance(lima_today(), date)

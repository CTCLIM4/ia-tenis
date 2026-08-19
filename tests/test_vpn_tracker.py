"""Tests for src/utils/vpn_tracker.py."""
from __future__ import annotations

import json
from datetime import datetime

import src.utils.vpn_tracker as vpn_tracker


class TestQueryAdapterBytes:
    def test_returns_bytes_tuple_on_success(self, monkeypatch):
        class _Proc:
            stdout = '{"ReceivedBytes":1000,"SentBytes":500}'

        monkeypatch.setattr(vpn_tracker.subprocess, "run", lambda *a, **kw: _Proc())
        assert vpn_tracker._query_adapter_bytes() == (1000, 500)

    def test_returns_none_when_powershell_missing(self, monkeypatch):
        def _boom(*a, **kw):
            raise FileNotFoundError("powershell not found")
        monkeypatch.setattr(vpn_tracker.subprocess, "run", _boom)
        assert vpn_tracker._query_adapter_bytes() is None

    def test_returns_none_when_adapter_not_found(self, monkeypatch):
        class _Proc:
            stdout = ""
        monkeypatch.setattr(vpn_tracker.subprocess, "run", lambda *a, **kw: _Proc())
        assert vpn_tracker._query_adapter_bytes() is None

    def test_returns_none_on_malformed_json(self, monkeypatch):
        class _Proc:
            stdout = "not json"
        monkeypatch.setattr(vpn_tracker.subprocess, "run", lambda *a, **kw: _Proc())
        assert vpn_tracker._query_adapter_bytes() is None


class TestTrackVpnUsage:
    def _mock_stats(self, monkeypatch, rx, tx):
        monkeypatch.setattr(vpn_tracker, "_query_adapter_bytes", lambda *a, **kw: (rx, tx))

    def test_adapter_unavailable_returns_zeros_but_does_not_crash(self, monkeypatch, tmp_path):
        monkeypatch.setattr(vpn_tracker, "_query_adapter_bytes", lambda *a, **kw: None)
        result = vpn_tracker.track_vpn_usage(usage_path=tmp_path / "usage.json")
        assert result["available"] is False
        assert result["session_mb"] == 0.0

    def test_first_reading_of_the_month_uses_full_total_as_session(self, monkeypatch, tmp_path):
        self._mock_stats(monkeypatch, rx=5 * vpn_tracker.MB, tx=5 * vpn_tracker.MB)
        result = vpn_tracker.track_vpn_usage(
            usage_path=tmp_path / "usage.json", now=datetime(2026, 8, 19),
        )
        assert result["available"] is True
        assert result["session_mb"] == 10.0
        assert result["monthly_cumulative_mb"] == 10.0
        assert result["remaining_mb"] == vpn_tracker.DEFAULT_MONTHLY_LIMIT_MB - 10.0

    def test_second_reading_same_month_accumulates_delta(self, monkeypatch, tmp_path):
        path = tmp_path / "usage.json"
        self._mock_stats(monkeypatch, rx=5 * vpn_tracker.MB, tx=5 * vpn_tracker.MB)
        vpn_tracker.track_vpn_usage(usage_path=path, now=datetime(2026, 8, 19))

        self._mock_stats(monkeypatch, rx=8 * vpn_tracker.MB, tx=5 * vpn_tracker.MB)
        result = vpn_tracker.track_vpn_usage(usage_path=path, now=datetime(2026, 8, 19, 12))

        assert result["session_mb"] == 3.0
        assert result["monthly_cumulative_mb"] == 13.0

    def test_counter_reset_treated_as_new_session_not_negative(self, monkeypatch, tmp_path):
        path = tmp_path / "usage.json"
        self._mock_stats(monkeypatch, rx=100 * vpn_tracker.MB, tx=0)
        vpn_tracker.track_vpn_usage(usage_path=path, now=datetime(2026, 8, 19))

        # adapter reconnected: counters dropped back near zero
        self._mock_stats(monkeypatch, rx=2 * vpn_tracker.MB, tx=0)
        result = vpn_tracker.track_vpn_usage(usage_path=path, now=datetime(2026, 8, 19, 13))

        assert result["session_mb"] == 2.0
        assert result["monthly_cumulative_mb"] == 102.0

    def test_new_month_resets_cumulative(self, monkeypatch, tmp_path):
        path = tmp_path / "usage.json"
        self._mock_stats(monkeypatch, rx=1900 * vpn_tracker.MB, tx=0)
        vpn_tracker.track_vpn_usage(usage_path=path, now=datetime(2026, 7, 31))

        self._mock_stats(monkeypatch, rx=1 * vpn_tracker.MB, tx=0)
        result = vpn_tracker.track_vpn_usage(usage_path=path, now=datetime(2026, 8, 1))

        assert result["monthly_cumulative_mb"] == 1.0
        assert result["month"] == "2026-08"

    def test_percent_used_and_remaining_respect_custom_limit(self, monkeypatch, tmp_path):
        self._mock_stats(monkeypatch, rx=100 * vpn_tracker.MB, tx=0)
        result = vpn_tracker.track_vpn_usage(
            monthly_limit_mb=200, usage_path=tmp_path / "usage.json", now=datetime(2026, 8, 19),
        )
        assert result["percent_used"] == 50.0
        assert result["remaining_mb"] == 100.0

    def test_remaining_mb_floors_at_zero_when_over_limit(self, monkeypatch, tmp_path):
        self._mock_stats(monkeypatch, rx=3000 * vpn_tracker.MB, tx=0)
        result = vpn_tracker.track_vpn_usage(
            monthly_limit_mb=2000, usage_path=tmp_path / "usage.json", now=datetime(2026, 8, 19),
        )
        assert result["remaining_mb"] == 0.0
        assert result["percent_used"] > 100.0

    def test_persists_state_to_disk(self, monkeypatch, tmp_path):
        path = tmp_path / "usage.json"
        self._mock_stats(monkeypatch, rx=5 * vpn_tracker.MB, tx=0)
        vpn_tracker.track_vpn_usage(usage_path=path, now=datetime(2026, 8, 19))
        assert path.exists()
        state = json.loads(path.read_text(encoding="utf-8"))
        assert state["month"] == "2026-08"
        assert state["cumulative_mb"] == 5.0

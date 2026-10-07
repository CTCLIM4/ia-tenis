"""Tests for src/utils/notifier.py."""
from __future__ import annotations

import src.utils.notifier as notifier

SAMPLE_PICKS = [
    {"match": "Tirante vs Mensik", "pick": "Tirante", "odds": 2.75, "edge": 0.0322, "kelly": 0.0092},
    {"match": "Zverev vs Paul", "pick": "Paul", "odds": 3.29, "edge": 0.067, "kelly": 0.0146},
]

SAMPLE_VPN_OK = {
    "available": True, "session_mb": 12.5, "monthly_cumulative_mb": 250.0,
    "remaining_mb": 1750.0, "percent_used": 12.5,
}

SAMPLE_VPN_UNAVAILABLE = {"available": False, "session_mb": 0.0, "monthly_cumulative_mb": 0.0,
                          "remaining_mb": 2000.0, "percent_used": 0.0}

SAMPLE_PICKS_WITH_BOOKMAKER = [
    {"match": "Tirante vs Mensik", "pick": "Tirante", "odds": 2.75, "edge": 0.0322,
     "kelly": 0.0092, "bookmaker": "bet365"},
]

ALL_ENV_VARS = {
    "SMTP_SERVER": "smtp.example.com", "SMTP_PORT": "587",
    "EMAIL_SENDER": "bot@example.com", "EMAIL_PASSWORD": "secret",
    "EMAIL_RECEIVER": "me@example.com",
}


class TestMissingEnvVars:
    def test_all_present_returns_empty(self, monkeypatch):
        for k, v in ALL_ENV_VARS.items():
            monkeypatch.setenv(k, v)
        assert notifier._missing_env_vars() == []

    def test_reports_each_missing_var(self, monkeypatch):
        for k in notifier.REQUIRED_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("SMTP_SERVER", "smtp.example.com")
        missing = notifier._missing_env_vars()
        assert "SMTP_SERVER" not in missing
        assert "EMAIL_PASSWORD" in missing
        assert "EMAIL_RECEIVER" in missing


class TestBuildEmailHtml:
    def test_includes_each_pick_row(self):
        html = notifier.build_email_html(SAMPLE_PICKS, SAMPLE_VPN_OK)
        assert "Tirante vs Mensik" in html
        assert "Tirante" in html
        assert "2.75" in html
        assert "Zverev vs Paul" in html

    def test_empty_picks_shows_placeholder(self):
        html = notifier.build_email_html([], SAMPLE_VPN_OK)
        assert "No se seleccionaron picks" in html

    def test_vpn_panel_shows_remaining_and_percent(self):
        html = notifier.build_email_html(SAMPLE_PICKS, SAMPLE_VPN_OK)
        assert "1750.0" in html
        assert "12.5" in html

    def test_vpn_unavailable_shows_fallback_message(self):
        html = notifier.build_email_html(SAMPLE_PICKS, SAMPLE_VPN_UNAVAILABLE)
        assert "no disponible" in html


class TestSendPicksEmail:
    def test_skips_and_returns_false_when_env_missing(self, monkeypatch):
        for k in notifier.REQUIRED_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        assert notifier.send_picks_email(SAMPLE_PICKS, SAMPLE_VPN_OK) is False

    def test_sends_via_smtp_when_configured(self, monkeypatch):
        for k, v in ALL_ENV_VARS.items():
            monkeypatch.setenv(k, v)

        sent = {}

        class _FakeSMTP:
            def __init__(self, server, port, timeout=None):
                sent["server"] = server
                sent["port"] = port

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def starttls(self):
                sent["starttls"] = True

            def login(self, user, password):
                sent["login"] = (user, password)

            def send_message(self, msg):
                sent["message"] = msg

        monkeypatch.setattr(notifier.smtplib, "SMTP", _FakeSMTP)
        result = notifier.send_picks_email(SAMPLE_PICKS, SAMPLE_VPN_OK)

        assert result is True
        assert sent["server"] == "smtp.example.com"
        assert sent["port"] == 587
        assert sent["login"] == ("bot@example.com", "secret")
        assert sent["message"]["To"] == "me@example.com"

    def test_returns_false_and_does_not_raise_on_smtp_failure(self, monkeypatch):
        for k, v in ALL_ENV_VARS.items():
            monkeypatch.setenv(k, v)

        def _boom(*a, **kw):
            raise OSError("connection refused")

        monkeypatch.setattr(notifier.smtplib, "SMTP", _boom)
        assert notifier.send_picks_email(SAMPLE_PICKS, SAMPLE_VPN_OK) is False


class TestPicksTableBookmakerColumn:
    def test_shows_bookmaker_when_present(self):
        html = notifier.build_email_html(SAMPLE_PICKS_WITH_BOOKMAKER, SAMPLE_VPN_OK)
        assert "bet365" in html

    def test_blank_when_bookmaker_absent(self):
        # SAMPLE_PICKS (module-level) has no "bookmaker" key — must not raise.
        html = notifier.build_email_html(SAMPLE_PICKS, SAMPLE_VPN_OK)
        assert "Tirante vs Mensik" in html


class TestTrackingPanels:
    SIGNAL = {"match": "Sinner vs Rune", "pick": "Rune", "odds": 3.1, "edge": 0.045,
              "kelly": 0.0, "bookmaker": "pinnacle"}

    def test_stake_header_says_quarter_kelly(self):
        # daily_workflow stakes 1/4 Kelly (KELLY_DIVISOR); the header said 1/2.
        assert "1/4 Kelly" in notifier.build_email_html(SAMPLE_PICKS, SAMPLE_VPN_OK)

    def test_lists_signals_or_says_none(self):
        html = notifier.build_email_html([], SAMPLE_VPN_OK, signals=[self.SIGNAL])
        assert "Sinner vs Rune" in html and "3.10" in html and "+4.5%" in html
        assert "Ninguna hoy" in notifier.build_email_html([], SAMPLE_VPN_OK, signals=[])

    def test_tracking_without_data(self):
        html = notifier.build_email_html([], SAMPLE_VPN_OK, tracking={"runs": 0, "slope": None})
        assert "Sin snapshots" in html and "aun no hay partidos" in html
        assert "No disponible hoy" in notifier.build_email_html([], SAMPLE_VPN_OK, tracking=None)

    def test_tracking_with_clv(self):
        import pandas as pd
        tracking = {"runs": 9, "runs_24h": 4, "last_run": pd.Timestamp("2026-10-20T18:00:00Z"),
                    "events": 120, "credits_remaining": 310, "matches": 85,
                    "slope": 0.12, "slope_ci": (-0.05, 0.29), "picks": 14, "clv": -0.021,
                    "clv_ci": (-0.06, 0.02)}
        html = notifier.build_email_html([], SAMPLE_VPN_OK, tracking=tracking)
        assert "2026-10-20 13:00" in html  # Lima time
        assert "310" in html and "+0.12" in html and "N=14" in html and "-2.1%" in html

    def test_subject_mentions_signals(self, monkeypatch):
        for k, v in ALL_ENV_VARS.items():
            monkeypatch.setenv(k, v)
        sent = {}

        class _FakeSMTP:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def starttls(self): pass
            def login(self, *a): pass
            def send_message(self, msg): sent["msg"] = msg

        monkeypatch.setattr(notifier.smtplib, "SMTP", _FakeSMTP)
        notifier.send_picks_email([], SAMPLE_VPN_OK, signals=[self.SIGNAL])
        assert sent["msg"]["Subject"] == "ia-tenis: 0 pick(s) del dia, 1 senal(es) en seguimiento"

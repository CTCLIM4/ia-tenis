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

"""Tests for src/utils/telegram.py. _send / urlopen are always mocked:
nothing here ever reaches the Telegram API."""
from __future__ import annotations

import io
import json
import urllib.error
from datetime import date

import pytest

import src.utils.telegram as tg

DAY = date(2026, 10, 8)
SIGNAL = {"match": "Navone vs Carreno Busta", "pick": "Navone", "odds": 1.79, "edge": 0.084,
          "bookmaker": "gtbets", "p_model": 0.64, "tournament": "ATP Shanghai Masters",
          "match_date": "2026-10-08"}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-10042")


class TestBuildMessages:
    def test_every_message_carries_the_disclaimer(self):
        msgs = tg.build_messages([SIGNAL], DAY)
        assert len(msgs) == 1
        m = msgs[0]
        assert "Navone" in m and "1.79" in m and "64%" in m and "+8.4%" in m and "2026-10-08" in m
        assert "No son recomendaciones de apuesta" in m

    def test_nothing_to_post(self):
        assert tg.build_messages([], DAY) == []

    def test_escapes_html(self):
        m = tg.build_messages([{**SIGNAL, "pick": "A<b>", "match": "A<b> vs C&D"}], DAY)[0]
        assert "A&lt;b&gt;" in m and "C&amp;D" in m

    def test_splits_long_lists_under_telegram_limit(self):
        msgs = tg.build_messages([SIGNAL] * 60, DAY)
        assert len(msgs) > 1
        assert all(len(m) <= tg.MAX_MESSAGE_CHARS and tg.DISCLAIMER in m for m in msgs)
        assert sum(m.count("• ") for m in msgs) == 60


class TestSendSignals:
    def test_skips_without_config(self, monkeypatch, capsys):
        for v in tg.REQUIRED_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setattr(tg, "_send", lambda *a: pytest.fail("must not send"))
        assert tg.send_signals([SIGNAL], DAY) is False
        assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().out

    def test_sends_nothing_without_signals(self, configured, monkeypatch):
        monkeypatch.setattr(tg, "_send", lambda *a: pytest.fail("must not send"))
        assert tg.send_signals([], DAY) is False

    def test_posts_to_configured_chat(self, configured, monkeypatch):
        sent = []
        monkeypatch.setattr(tg, "_send", lambda token, chat, text: sent.append((token, chat, text)))
        assert tg.send_signals([SIGNAL], DAY) is True
        assert sent[0][:2] == ("123:SECRET", "-10042")

    def test_request_shape(self, configured, monkeypatch):
        captured = {}

        class _Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout):
            captured["url"], captured["data"] = req.full_url, req.data.decode()
            return _Resp(json.dumps({"ok": True}).encode())
        monkeypatch.setattr(tg.urllib.request, "urlopen", fake_urlopen)

        assert tg.send_signals([SIGNAL], DAY) is True
        assert captured["url"] == "https://api.telegram.org/bot123:SECRET/sendMessage"
        assert "chat_id=-10042" in captured["data"] and "parse_mode=HTML" in captured["data"]

    def test_http_error_never_leaks_token(self, configured, monkeypatch, capsys):
        def boom(*a, **kw):
            raise urllib.error.HTTPError(
                "https://api.telegram.org/bot123:SECRET/sendMessage", 400, "Bad Request", {},
                io.BytesIO(b'{"ok":false,"description":"chat not found"}'))
        monkeypatch.setattr(tg.urllib.request, "urlopen", boom)

        assert tg.send_signals([SIGNAL], DAY) is False
        out = capsys.readouterr().out
        assert "chat not found" in out and "SECRET" not in out

    def test_api_not_ok_is_a_failure(self, configured, monkeypatch):
        def not_ok(*a):
            raise RuntimeError("Forbidden: bot is not a member")
        monkeypatch.setattr(tg, "_send", not_ok)
        assert tg.send_signals([SIGNAL], DAY) is False

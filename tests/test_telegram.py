"""Tests for src/utils/telegram.py. _send / urlopen are always mocked:
nothing here ever reaches the Telegram API."""
from __future__ import annotations

import io
import json
import urllib.error
from datetime import date

import pytest

import src.utils.telegram as tg

DAY = date(2026, 10, 7)
PRED = {"tournament": "ATP Shanghai Masters", "match_date": "2026-10-08",
        "commence_time": "2026-10-08T05:10:00Z",
        "player_a": "Mariano Navone", "player_b": "Pablo Carreno Busta", "p_a": 0.64}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-10042")


class TestBuildMessages:
    def test_shows_both_probabilities_and_no_betting_fields(self):
        (m,) = tg.build_messages([PRED], DAY)
        assert "PREDICCIONES DEL MODELO" in m and "Miércoles 7 de octubre" in m and "1 partido(s)" in m
        assert "jue 00:10" in m  # Lima time
        assert "<b>Mariano Navone 64%</b>" in m and "Pablo Carreno Busta 36%" in m
        for betting in ("@", "Edge", "Cuota", "➜"):
            assert betting not in m
        assert tg.FOOTER_NOTE in m

    def test_bolds_the_favourite_only(self):
        (m,) = tg.build_messages([{**PRED, "p_a": 0.3}], DAY)
        assert "Mariano Navone 30%" in m and "<b>Mariano Navone" not in m
        assert "<b>Pablo Carreno Busta 70%</b>" in m
        (even,) = tg.build_messages([{**PRED, "p_a": 0.5}], DAY)
        assert "<b>Mariano" not in even and "<b>Pablo" not in even

    def test_probabilities_add_up_to_100(self):
        (m,) = tg.build_messages([{**PRED, "p_a": 0.645}], DAY)
        assert "64%" in m and "36%" in m or "65%" in m and "35%" in m

    def test_grouped_by_tournament_and_sorted_by_time(self):
        preds = [
            PRED,
            {**PRED, "tournament": "WTA China Open", "player_a": "Ekaterina Alexandrova",
             "player_b": "Mirra Andreeva", "commence_time": "2026-10-08T11:00:00Z"},
            {**PRED, "player_a": "Jaume Munar", "player_b": "Jenson Brooksby", "commence_time": "2026-10-08T04:00:00Z"},
        ]
        (m,) = tg.build_messages(preds, DAY)
        assert m.index("ATP Shanghai") < m.index("Munar") < m.index("Navone") < m.index("WTA China Open")
        assert m.count("🏆") == 2

    def test_falls_back_to_match_date_without_kickoff(self):
        (m,) = tg.build_messages([{**PRED, "commence_time": ""}], DAY)
        assert "2026-10-08" in m

    def test_nothing_to_post(self):
        assert tg.build_messages([], DAY) == []

    def test_escapes_html(self):
        (m,) = tg.build_messages([{**PRED, "player_a": "A<b>", "player_b": "C&D"}], DAY)
        assert "A&lt;b&gt;" in m and "C&amp;D" in m

    def test_splits_long_lists_under_telegram_limit(self):
        msgs = tg.build_messages([PRED] * 80, DAY)
        assert len(msgs) > 1
        assert all(len(m) <= tg.MAX_MESSAGE_CHARS and tg.FOOTER_NOTE in m for m in msgs)
        assert sum(m.count("🕐") for m in msgs) == 80
        assert all(m.count("🏆") == 1 for m in msgs)  # tournament heading repeated per message


class TestSendPredictions:
    def test_skips_without_config(self, monkeypatch, capsys):
        for v in tg.REQUIRED_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setattr(tg, "_send", lambda *a: pytest.fail("must not send"))
        assert tg.send_predictions([PRED], DAY) is False
        assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().out

    def test_sends_nothing_without_predictions(self, configured, monkeypatch):
        monkeypatch.setattr(tg, "_send", lambda *a: pytest.fail("must not send"))
        assert tg.send_predictions([], DAY) is False

    def test_posts_to_configured_chat(self, configured, monkeypatch):
        sent = []
        monkeypatch.setattr(tg, "_send", lambda token, chat, text: sent.append((token, chat, text)))
        assert tg.send_predictions([PRED], DAY) is True
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

        assert tg.send_predictions([PRED], DAY) is True
        assert captured["url"] == "https://api.telegram.org/bot123:SECRET/sendMessage"
        assert "chat_id=-10042" in captured["data"] and "parse_mode=HTML" in captured["data"]

    def test_http_error_never_leaks_token(self, configured, monkeypatch, capsys):
        def boom(*a, **kw):
            raise urllib.error.HTTPError(
                "https://api.telegram.org/bot123:SECRET/sendMessage", 400, "Bad Request", {},
                io.BytesIO(b'{"ok":false,"description":"chat not found"}'))
        monkeypatch.setattr(tg.urllib.request, "urlopen", boom)

        assert tg.send_predictions([PRED], DAY) is False
        out = capsys.readouterr().out
        assert "chat not found" in out and "SECRET" not in out

    def test_api_not_ok_is_a_failure(self, configured, monkeypatch):
        def not_ok(*a):
            raise RuntimeError("Forbidden: bot is not a member")
        monkeypatch.setattr(tg, "_send", not_ok)
        assert tg.send_predictions([PRED], DAY) is False

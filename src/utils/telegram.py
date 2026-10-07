"""Post the daily model signals to a Telegram group/channel.

Configured through TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (repo .env).
Only signals without warnings are posted -- the same gate as
scripts/daily_workflow._qualifying_sides: Elo found for both players, 25+
matches each, MIN_EDGE <= edge <= MAX_SUSPICIOUS_EDGE.

Other people read the chat, so every message says plainly that these are
not bets: the market-edge backtest (docs/metrics/2026-10-07-model-vs-market-
edge.md) found that exactly these picks lose money against the closing
market. Never raises -- a notification problem must not break the daily run.
"""
from __future__ import annotations

import html
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import src.config  # noqa: F401  (loads .env as a side effect)

REQUIRED_ENV_VARS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_CHARS = 4096

DISCLAIMER = (
    "⚠️ <b>No son recomendaciones de apuesta.</b> El modelo no ha demostrado ventaja frente al "
    "mercado: en el backtest 2010–2026 este mismo tipo de señal perdió dinero "
    "(ROI ≈ −11% a cuota media). Se publican solo para seguimiento."
)


def _missing_env_vars() -> list[str]:
    return [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]


def _signal_line(s: dict) -> str:
    where = " · ".join(html.escape(str(x)) for x in (s.get("tournament"), s.get("match_date")) if x)
    line = (f"• <b>{html.escape(s['pick'])}</b> @ {s['odds']:.2f}"
            f"{' (' + html.escape(s['bookmaker']) + ')' if s.get('bookmaker') else ''}\n"
            f"   {html.escape(s['match'])}")
    if where:
        line += f" — {where}"
    p = s.get("p_model")
    line += f"\n   modelo {p * 100:.0f}% · edge {s['edge'] * 100:+.1f}%" if p is not None else \
        f"\n   edge {s['edge'] * 100:+.1f}%"
    return line


def build_messages(signals: list[dict], day: date) -> list[str]:
    """Telegram-sized HTML messages; empty list when there's nothing to post."""
    if not signals:
        return []
    header = f"🎾 <b>Señales del modelo — {day.isoformat()}</b> ({len(signals)})\n"
    messages, current = [], header
    for line in map(_signal_line, signals):
        # Leave room for the disclaimer, which goes at the end of every message.
        if len(current) + len(line) + len(DISCLAIMER) + 4 > MAX_MESSAGE_CHARS:
            messages.append(current + "\n" + DISCLAIMER)
            current = header
        current += "\n" + line
    messages.append(current + "\n\n" + DISCLAIMER)
    return messages


def _send(token: str, chat_id: str, text: str) -> None:
    data = urllib.parse.urlencode({
        "chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true",
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(API_URL.format(token=token), data=data), timeout=15) as resp:
        body = json.loads(resp.read())
    if not body.get("ok"):
        raise RuntimeError(body.get("description", "respuesta no ok"))


def send_signals(signals: list[dict], day: date) -> bool:
    """Post the signals. Returns True if every message was sent, False if
    skipped (no config / nothing to send) or any send failed."""
    missing = _missing_env_vars()
    if missing:
        print(f"  Aviso: Telegram omitido (faltan variables de entorno: {', '.join(missing)}).")
        return False
    messages = build_messages(signals, day)
    if not messages:
        print("  Telegram: sin señales hoy, no se envía nada.")
        return False
    token, chat_id = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    try:
        for text in messages:
            _send(token, chat_id, text)
    except urllib.error.HTTPError as e:
        # Never print the URL: it contains the bot token.
        detail = ""
        try:
            detail = json.loads(e.read()).get("description", "")
        except Exception:
            pass
        print(f"  Aviso: fallo al enviar a Telegram (HTTP {e.code} {detail}).")
        return False
    except Exception as e:
        print(f"  Aviso: fallo al enviar a Telegram ({type(e).__name__}: {e}).")
        return False
    print(f"  Telegram: {len(signals)} señal(es) enviadas en {len(messages)} mensaje(s).")
    return True

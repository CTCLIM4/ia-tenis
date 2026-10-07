"""Post the model's daily match predictions to a Telegram channel.

Configured through TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (repo .env).
Each match shows who plays and the model's win probability for each player
-- no picks, odds or edges (the channel asked for predictions, not betting
suggestions). Only matches the model can rate properly are posted: Elo
found for both players and 25+ matches each. A short footer says these are
model estimates, not betting advice: the market-edge backtest
(docs/metrics/2026-10-07-model-vs-market-edge.md) found the market more
accurate than the model. Never raises -- a notification problem must not
break the daily run.
"""
from __future__ import annotations

import html
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime

import src.config  # noqa: F401  (loads .env as a side effect)
from src.data.timezone_utils import LIMA_TZ

REQUIRED_ENV_VARS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
API_URL = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MESSAGE_CHARS = 4096

FOOTER_NOTE = "ℹ️ <i>Probabilidades estimadas por el modelo. No son recomendaciones de apuesta.</i>"
SEPARATOR = "➖➖➖➖➖➖➖➖"

_DAYS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
_FULL_DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
_MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
           "septiembre", "octubre", "noviembre", "diciembre"]


def _missing_env_vars() -> list[str]:
    return [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]


def _kickoff(p: dict) -> tuple[str, str]:
    """(sort key, label) in Lima time; falls back to the match date."""
    raw = p.get("commence_time")
    if raw:
        t = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(LIMA_TZ)
        return t.isoformat(), f"{_DAYS[t.weekday()]} {t:%H:%M}"
    return str(p.get("match_date", "")), str(p.get("match_date", ""))


def _prediction_block(p: dict) -> str:
    _, when = _kickoff(p)
    a, b = html.escape(p["player_a"]), html.escape(p["player_b"])
    pa = round(p["p_a"] * 100)
    pb = 100 - pa
    # Bold the model's favourite; a 50/50 call stays plain.
    sa = f"<b>{a} {pa}%</b>" if pa > pb else f"{a} {pa}%"
    sb = f"<b>{b} {pb}%</b>" if pb > pa else f"{b} {pb}%"
    return f"🕐 <b>{when}</b> · {a} vs {b}\n     📈 {sa} · {sb}"


def _header(day: date, n: int) -> str:
    return "\n".join([
        "🎾 <b>PREDICCIONES DEL MODELO</b>",
        f"📅 {_FULL_DAYS[day.weekday()]} {day.day} de {_MONTHS[day.month - 1]} · hora de Lima",
        f"📊 {n} partido(s)",
    ]) + "\n"


def _tournament_heading(tournament: str) -> str:
    return f"\n🏆 <b>{html.escape(tournament)}</b>\n" if tournament else "\n"


def build_messages(predictions: list[dict], day: date) -> list[str]:
    """Telegram-sized HTML messages, grouped by tournament and ordered by
    start time, each ending with the footer note; empty list when there's
    nothing to post."""
    if not predictions:
        return []
    ordered = sorted(predictions, key=lambda p: (p.get("tournament", ""), _kickoff(p)[0]))
    footer = f"\n{SEPARATOR}\n{FOOTER_NOTE}"
    header = _header(day, len(predictions))
    messages, current, tournament = [], header, None
    for p in ordered:
        block = _prediction_block(p) + "\n"
        heading = ""
        if p.get("tournament", "") != tournament:
            tournament = p.get("tournament", "")
            heading = _tournament_heading(tournament)
        if len(current) + len(heading) + len(block) + len(footer) + 1 > MAX_MESSAGE_CHARS:
            messages.append(current + footer)
            current, heading = header, _tournament_heading(tournament)
        current += heading + "\n" + block
    messages.append(current + footer)
    return messages


def _send(token: str, chat_id: str, text: str) -> None:
    data = urllib.parse.urlencode({
        "chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true",
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(API_URL.format(token=token), data=data), timeout=15) as resp:
        body = json.loads(resp.read())
    if not body.get("ok"):
        raise RuntimeError(body.get("description", "respuesta no ok"))


def send_predictions(predictions: list[dict], day: date) -> bool:
    """Post the predictions. Returns True if every message was sent, False
    if skipped (no config / nothing to send) or any send failed."""
    missing = _missing_env_vars()
    if missing:
        print(f"  Aviso: Telegram omitido (faltan variables de entorno: {', '.join(missing)}).")
        return False
    messages = build_messages(predictions, day)
    if not messages:
        print("  Telegram: sin partidos para publicar hoy.")
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
    print(f"  Telegram: {len(predictions)} prediccion(es) enviadas en {len(messages)} mensaje(s).")
    return True

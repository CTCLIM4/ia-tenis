"""Email notification for the daily picks run.

Sends an HTML summary of the selected value bets, the model's signals in
tours excluded from auto-logging, the odds-snapshot/CLV tracking status
(scripts/clv_report.summary), plus the VPN usage status
(see src/utils/vpn_tracker.py) to EMAIL_RECEIVER, via a plain SMTP+STARTTLS
connection configured through environment variables (see .env.example).
"""
from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

import src.config  # noqa: F401  (loads .env as a side effect)

REQUIRED_ENV_VARS = ("SMTP_SERVER", "SMTP_PORT", "EMAIL_SENDER", "EMAIL_PASSWORD", "EMAIL_RECEIVER")


def _missing_env_vars() -> list[str]:
    return [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]


def _picks_table_html(selected_picks: list[dict]) -> str:
    if not selected_picks:
        return "<p>No se seleccionaron picks en esta jornada.</p>"
    rows = "\n".join(
        f'<tr><td style="padding:6px;border-bottom:1px solid #e0e0e0;">{p["match"]}</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;">{p["pick"]}</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;">{p.get("bookmaker", "")}</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;text-align:right;">{p["odds"]:.2f}</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;text-align:right;">{p["edge"] * 100:+.1f}%</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;text-align:right;">{p["kelly"] * 100:.2f}%</td></tr>'
        for p in selected_picks
    )
    return f"""
    <table style="border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:14px;">
      <thead>
        <tr style="background:#1f3a5f;color:#fff;">
          <th style="padding:8px;text-align:left;">Partido</th>
          <th style="padding:8px;text-align:left;">Pick</th>
          <th style="padding:8px;text-align:left;">Bookmaker</th>
          <th style="padding:8px;text-align:right;">Cuota</th>
          <th style="padding:8px;text-align:right;">Edge</th>
          <th style="padding:8px;text-align:right;">Stake (1/4 Kelly)</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """


def _vpn_panel_html(vpn_status: dict) -> str:
    if not vpn_status.get("available", True):
        return (
            '<div style="margin-top:16px;padding:10px;background:#fff3cd;'
            'border:1px solid #ffe69c;font-family:Arial,sans-serif;font-size:13px;">'
            "VPN: estado no disponible (adaptador no encontrado).</div>"
        )
    pct = vpn_status.get("percent_used", 0.0)
    color = "#c0392b" if pct >= 90 else "#e67e22" if pct >= 70 else "#27ae60"
    return f"""
    <div style="margin-top:16px;padding:10px;background:#f4f6f8;
                border-left:4px solid {color};font-family:Arial,sans-serif;font-size:13px;">
      <strong>VPN (TunnelBear)</strong><br>
      Sesion: {vpn_status.get('session_mb', 0):.1f} MB &nbsp;|&nbsp;
      Acumulado del mes: {vpn_status.get('monthly_cumulative_mb', 0):.1f} MB &nbsp;|&nbsp;
      Restante: {vpn_status.get('remaining_mb', 0):.1f} MB
      ({pct:.1f}% consumido)
    </div>
    """


_PANEL = ('<div style="margin-top:16px;padding:10px;background:#f4f6f8;border-left:4px solid #1f3a5f;'
          'font-family:Arial,sans-serif;font-size:13px;">')


def _signals_html(signals: list[dict]) -> str:
    """Picks the model would make in tours excluded from auto-logging --
    tracked for CLV, never bet."""
    if not signals:
        return f"{_PANEL}<strong>Senales del modelo (no apostadas)</strong><br>Ninguna hoy.</div>"
    items = "".join(
        f"<li>{s['match']} &rarr; {s['pick']} @ {s['odds']:.2f} ({s.get('bookmaker', '')}), "
        f"edge {s['edge'] * 100:+.1f}%</li>"
        for s in signals
    )
    return (f"{_PANEL}<strong>Senales del modelo (no apostadas, solo seguimiento de CLV)</strong>"
            f'<ul style="margin:6px 0 0 18px;padding:0;">{items}</ul></div>')


def _tracking_html(tracking: dict | None) -> str:
    """Odds-snapshot health + accumulated CLV (scripts/clv_report.summary)."""
    if tracking is None:
        return f"{_PANEL}<strong>Seguimiento de CLV</strong><br>No disponible hoy.</div>"
    t = tracking
    if not t.get("runs"):
        snap = "Sin snapshots de cuotas todavia."
    else:
        last = t["last_run"].tz_convert("America/Lima").strftime("%Y-%m-%d %H:%M") if t.get("last_run") is not None else "?"
        credits = t.get("credits_remaining")
        snap = (f"Snapshots: {t['runs_24h']} en las ultimas 24 h (ultimo {last}), {t['events']} partidos registrados; "
                f"creditos API restantes: {credits if credits is not None else '?'}")
    if t.get("slope") is None:
        clv = "CLV: aun no hay partidos con linea al escanear y al cierre."
    else:
        lo, hi = t["slope_ci"]
        clv = (f"{t['matches']} partidos con ambas lineas. La linea se mueve hacia el modelo: pendiente "
               f"{t['slope']:+.2f} (IC 95% {lo:+.2f}..{hi:+.2f}; &gt; 0 = informacion que el mercado aun no tenia).")
        if t.get("picks"):
            clo, chi = t["clv_ci"]
            clv += (f"<br>Picks de la regla: N={t['picks']}, CLV medio {t['clv'] * 100:+.1f}% "
                    f"(IC 95% {clo * 100:+.1f}..{chi * 100:+.1f}%).")
    return f"{_PANEL}<strong>Seguimiento de CLV</strong><br>{snap}<br>{clv}</div>"


def build_email_html(
    selected_picks: list[dict], vpn_status: dict,
    signals: list[dict] | None = None, tracking: dict | None = None,
) -> str:
    return f"""
    <html><body style="font-family:Arial,sans-serif;">
      <h2>Picks del dia — ia-tenis</h2>
      {_picks_table_html(selected_picks)}
      {_signals_html(signals or [])}
      {_tracking_html(tracking)}
      {_vpn_panel_html(vpn_status)}
    </body></html>
    """


def send_picks_email(
    selected_picks: list[dict], vpn_status: dict,
    signals: list[dict] | None = None, tracking: dict | None = None,
) -> bool:
    """Send the daily picks summary by email.

    Returns True if sent, False if skipped (missing SMTP config) or the send
    itself failed — never raises, so a notification problem doesn't abort
    the rest of the daily workflow.
    """
    missing = _missing_env_vars()
    if missing:
        print(f"  Aviso: notificacion por correo omitida (faltan variables de entorno: {', '.join(missing)}).")
        return False

    msg = EmailMessage()
    subject = f"ia-tenis: {len(selected_picks)} pick(s) del dia"
    if signals:
        subject += f", {len(signals)} senal(es) en seguimiento"
    msg["Subject"] = subject
    msg["From"] = os.environ["EMAIL_SENDER"]
    msg["To"] = os.environ["EMAIL_RECEIVER"]
    msg.set_content("Ver esta notificacion en un cliente de correo compatible con HTML.")
    msg.add_alternative(build_email_html(selected_picks, vpn_status, signals, tracking), subtype="html")

    try:
        with smtplib.SMTP(os.environ["SMTP_SERVER"], int(os.environ["SMTP_PORT"]), timeout=15) as smtp:
            smtp.starttls()
            smtp.login(os.environ["EMAIL_SENDER"], os.environ["EMAIL_PASSWORD"])
            smtp.send_message(msg)
    except Exception as e:
        print(f"  Aviso: fallo al enviar el correo de picks ({e}).")
        return False

    print(f"  Correo de picks enviado a {os.environ['EMAIL_RECEIVER']}.")
    return True

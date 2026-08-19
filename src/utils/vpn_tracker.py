"""Track VPN data usage against a monthly cap.

Reads cumulative traffic from a Windows network adapter (matched by name,
default "TunnelBear") via PowerShell's Get-NetAdapterStatistics — there is no
Windows equivalent of Linux's /sys/class/net/*/statistics sysfs files, so this
shells out to PowerShell instead of reading a file directly.

Windows' per-adapter byte counters are cumulative since the adapter last came
up (not since boot, and not persisted across disconnects) — they reset to 0
on every VPN reconnect. track_vpn_usage() persists its own running monthly
total in usage_path across calls precisely to survive those resets: each call
diffs the current reading against the last stored one, and if the adapter
counter is now lower than last time (a reset happened), the whole current
reading is counted as new usage rather than producing a negative delta.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent.parent

MB = 1024 * 1024
DEFAULT_USAGE_PATH = _ROOT / "data" / "vpn_usage.json"
DEFAULT_MONTHLY_LIMIT_MB = 2000
DEFAULT_ADAPTER_NAME_PATTERN = "TunnelBear"


def _query_adapter_bytes(
    name_pattern: str = DEFAULT_ADAPTER_NAME_PATTERN, timeout: int = 10,
) -> Optional[tuple[int, int]]:
    """Return (received_bytes, sent_bytes) for the first network adapter whose
    Name or InterfaceDescription contains name_pattern.

    Returns None if no matching adapter is found (VPN not installed or not
    currently connected) or PowerShell itself isn't available/times out —
    callers must treat that as "status unknown", not an error to raise on.
    """
    script = (
        f"$a = Get-NetAdapter | Where-Object {{ $_.Name -like '*{name_pattern}*' "
        f"-or $_.InterfaceDescription -like '*{name_pattern}*' }} | Select-Object -First 1; "
        "if ($a) { Get-NetAdapterStatistics -Name $a.Name | "
        "Select-Object ReceivedBytes,SentBytes | ConvertTo-Json -Compress }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        return None

    out = (result.stdout or "").strip()
    if not out:
        return None
    try:
        data = json.loads(out)
        return int(data["ReceivedBytes"]), int(data["SentBytes"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def track_vpn_usage(
    monthly_limit_mb: float = DEFAULT_MONTHLY_LIMIT_MB,
    usage_path: Path = DEFAULT_USAGE_PATH,
    adapter_name_pattern: str = DEFAULT_ADAPTER_NAME_PATTERN,
    now: Optional[datetime] = None,
) -> dict:
    """Read current VPN adapter traffic, fold it into the persisted monthly
    total at usage_path, and return a usage summary.

    Returns:
        available              – False if the adapter couldn't be read; other
                                  fields fall back to 0/last-known-cumulative
        session_mb              – traffic seen since the last call
        monthly_cumulative_mb
        remaining_mb             – max(monthly_limit_mb - cumulative, 0)
        percent_used
        month                    – "YYYY-MM" the cumulative figure applies to
    """
    now = now or datetime.now()
    month = now.strftime("%Y-%m")
    usage_path = Path(usage_path)
    state = _load_state(usage_path)
    same_month = state.get("month") == month

    stats = _query_adapter_bytes(adapter_name_pattern)
    if stats is None:
        cumulative_mb = state.get("cumulative_mb", 0.0) if same_month else 0.0
        return {
            "available": False,
            "session_mb": 0.0,
            "monthly_cumulative_mb": cumulative_mb,
            "remaining_mb": max(monthly_limit_mb - cumulative_mb, 0.0),
            "percent_used": (cumulative_mb / monthly_limit_mb * 100) if monthly_limit_mb else 0.0,
            "month": month,
        }

    rx_bytes, tx_bytes = stats
    total_bytes = rx_bytes + tx_bytes

    cumulative_mb = state.get("cumulative_mb", 0.0) if same_month else 0.0
    last_total_bytes = state.get("last_total_bytes") if same_month else None

    if last_total_bytes is None or total_bytes < last_total_bytes:
        session_bytes = total_bytes
    else:
        session_bytes = total_bytes - last_total_bytes

    session_mb = session_bytes / MB
    cumulative_mb += session_mb
    remaining_mb = max(monthly_limit_mb - cumulative_mb, 0.0)
    percent_used = (cumulative_mb / monthly_limit_mb * 100) if monthly_limit_mb else 0.0

    _save_state(usage_path, {
        "month": month,
        "cumulative_mb": round(cumulative_mb, 4),
        "last_total_bytes": total_bytes,
        "last_checked": now.isoformat(timespec="seconds"),
        "monthly_limit_mb": monthly_limit_mb,
    })

    return {
        "available": True,
        "session_mb": round(session_mb, 4),
        "monthly_cumulative_mb": round(cumulative_mb, 4),
        "remaining_mb": round(remaining_mb, 4),
        "percent_used": round(percent_used, 2),
        "month": month,
    }

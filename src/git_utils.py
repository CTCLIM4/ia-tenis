"""Shared git introspection helpers — used by src/data/snapshots.py (which
tour/dataset was frozen at which commit) and src/calibration_audit.py
(which commit's model produced a given prediction), so both stamp the same
notion of "current code version" without duplicating the subprocess call.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def current_git_commit() -> str | None:
    """Best-effort short git commit hash for the current checkout.

    Returns None if git is unavailable, this isn't a git checkout, or the
    call times out — callers must treat None as "unknown" and never let
    this failure block whatever they were doing (creating a snapshot,
    logging a prediction).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None

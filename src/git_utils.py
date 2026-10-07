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


def commit_and_push(
    paths: list[str], message: str, remote: str = "origin", branch: str = "master",
) -> bool:
    """Stage the given paths, commit, and push to remote/branch.

    Returns True on success, False if there was nothing to commit or any
    step failed (printed, never raised) — used by scripts/daily_workflow.py
    and scripts/settle_workflow.py, where a failed push must not crash an
    otherwise-successful run; the caller re-runs it manually if needed.
    """
    import subprocess
    try:
        subprocess.run(
            ["git", "add", *paths], cwd=_ROOT, check=True, capture_output=True, text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", *paths],
            cwd=_ROOT, check=True, capture_output=True, text=True,
        )
        if not status.stdout.strip():
            print("  git: nada que commitear (sin cambios).")
            return False
        subprocess.run(
            ["git", "commit", "-m", message], cwd=_ROOT, check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "push", remote, branch], cwd=_ROOT, check=True, capture_output=True, text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        print(f"  Aviso: fallo el comando git {e.cmd}: {stderr}")
        return False
    except Exception as e:
        print(f"  Aviso: fallo git ({e}).")
        return False


def last_commit_time(path: str) -> int | None:
    """Unix time of the last commit touching `path`, or None if never
    committed or git is unavailable."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", path],
            cwd=_ROOT, capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def commit_paths(paths: list[str], message: str) -> bool:
    """Commit only `paths` (local, never pushes). Anything else the user has
    staged stays staged and out of this commit. Returns True if a commit was
    made; failures are printed, never raised."""
    import subprocess
    try:
        subprocess.run(["git", "add", "--", *paths], cwd=_ROOT, check=True, capture_output=True, text=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths],
                              cwd=_ROOT, capture_output=True, text=True)
        if diff.returncode == 0:
            return False
        subprocess.run(["git", "commit", "-m", message, "--only", "--", *paths],
                       cwd=_ROOT, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Aviso: fallo el comando git {e.cmd}: {(e.stderr or '').strip()}")
        return False
    except Exception as e:
        print(f"  Aviso: fallo git ({e}).")
        return False

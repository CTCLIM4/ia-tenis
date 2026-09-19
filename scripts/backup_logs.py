"""Timestamped backups of the append-only log files, before anything
modifies them.

Second line of defense alongside git-tracking prediction_audit_log.csv/
value_bets_log.csv (see .gitignore, tests/test_gitignore_protection.py):
git protects the last *committed* state, but a script can still corrupt
the working copy before anyone commits again -- this catches that moment
too. Prompted by the 2026-09-19 incident that deleted 26 rows from an
untracked prediction_audit_log.csv with no way back
(docs/metrics/2026-09-19-audit-log-protection.md).

Uso:
  python scripts/backup_logs.py
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

BACKUP_DIR = Path("data/logs")
KEEP_LAST = 7

DEFAULT_FILES = (
    Path("data/prediction_audit_log.csv"),
    Path("data/value_bets_log.csv"),
)


def backup_file(
    path, backup_dir=BACKUP_DIR, keep: int = KEEP_LAST, now: Optional[datetime] = None,
) -> Optional[Path]:
    """Copy `path` to backup_dir/backup_{YYYYMMDD_HHMMSS}_{path.name},
    preserving metadata (shutil.copy2). Returns the backup path, or None
    if `path` doesn't exist -- nothing to back up is not an error, and the
    backup directory is left untouched in that case. Rotates: keeps only
    the `keep` most recent backups of this specific filename, oldest
    deleted first.
    """
    path = Path(path)
    if not path.exists():
        return None

    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"backup_{timestamp}_{path.name}"
    shutil.copy2(path, backup_path)
    _rotate(backup_dir, path.name, keep)
    return backup_path


def _rotate(backup_dir: Path, filename: str, keep: int) -> None:
    backups = sorted(backup_dir.glob(f"backup_*_{filename}"), key=lambda p: p.name)
    for stale in backups[:-keep] if len(backups) > keep else []:
        stale.unlink()


def backup_logs(
    files: Iterable = DEFAULT_FILES, backup_dir=BACKUP_DIR, keep: int = KEEP_LAST,
    now: Optional[datetime] = None,
) -> list[Path]:
    """Back up each file in `files`. Returns the backup paths actually
    created, skipping any source file that doesn't exist."""
    created = []
    for f in files:
        result = backup_file(f, backup_dir=backup_dir, keep=keep, now=now)
        if result is not None:
            created.append(result)
    return created


def main() -> None:
    created = backup_logs()
    if not created:
        print("No hay archivos para respaldar.")
        return
    for path in created:
        print(f"Backup creado: {path}")


if __name__ == "__main__":
    main()

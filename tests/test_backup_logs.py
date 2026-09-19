"""Tests for scripts/backup_logs.py -- timestamped backups of the
append-only log files, a second line of defense alongside git-tracking
(tests/test_gitignore_protection.py) after the 2026-09-19 incident that
deleted 26 rows from an untracked prediction_audit_log.csv."""
from __future__ import annotations

from datetime import datetime

import pytest

import scripts.backup_logs as backup_logs


class TestBackupFile:
    def test_backup_creates_timestamped_copy(self, tmp_path):
        src = tmp_path / "prediction_audit_log.csv"
        src.write_text("header\nrow1\n")
        backup_dir = tmp_path / "backups"
        now = datetime(2026, 9, 19, 14, 30, 5)

        result = backup_logs.backup_file(src, backup_dir=backup_dir, now=now)

        assert result is not None
        assert result.exists()
        assert result.name == "backup_20260919_143005_prediction_audit_log.csv"
        assert result.parent == backup_dir

    def test_backup_preserves_original(self, tmp_path):
        src = tmp_path / "value_bets_log.csv"
        original_content = "header\nrow1\nrow2\n"
        src.write_text(original_content)
        backup_dir = tmp_path / "backups"

        backup_logs.backup_file(src, backup_dir=backup_dir, now=datetime(2026, 9, 19))

        assert src.read_text() == original_content

    def test_backup_content_matches_source(self, tmp_path):
        src = tmp_path / "value_bets_log.csv"
        content = "header\nrow1\nrow2\n"
        src.write_text(content)
        backup_dir = tmp_path / "backups"

        result = backup_logs.backup_file(src, backup_dir=backup_dir, now=datetime(2026, 9, 19))

        assert result.read_text() == content

    def test_backup_skips_if_file_missing(self, tmp_path):
        src = tmp_path / "does_not_exist.csv"
        backup_dir = tmp_path / "backups"

        result = backup_logs.backup_file(src, backup_dir=backup_dir, now=datetime(2026, 9, 19))

        assert result is None
        assert not backup_dir.exists()  # must not even create the dir for nothing

    def test_backup_rotation_keeps_last_7(self, tmp_path):
        src = tmp_path / "prediction_audit_log.csv"
        src.write_text("header\n")
        backup_dir = tmp_path / "backups"

        for i in range(10):
            now = datetime(2026, 9, 19, 0, 0, i)  # distinct second each time
            backup_logs.backup_file(src, backup_dir=backup_dir, keep=7, now=now)

        remaining = sorted(backup_dir.glob("backup_*_prediction_audit_log.csv"))
        assert len(remaining) == 7

    def test_backup_rotation_keeps_the_most_recent(self, tmp_path):
        src = tmp_path / "prediction_audit_log.csv"
        src.write_text("header\n")
        backup_dir = tmp_path / "backups"

        for i in range(10):
            now = datetime(2026, 9, 19, 0, 0, i)
            backup_logs.backup_file(src, backup_dir=backup_dir, keep=7, now=now)

        remaining = sorted(p.name for p in backup_dir.glob("backup_*_prediction_audit_log.csv"))
        # Seconds 0-2 (the 3 oldest) must have been rotated out; 3-9 kept.
        assert remaining[0] == "backup_20260919_000003_prediction_audit_log.csv"

    def test_rotation_does_not_touch_other_files_backups(self, tmp_path):
        audit = tmp_path / "prediction_audit_log.csv"
        bets = tmp_path / "value_bets_log.csv"
        audit.write_text("a\n")
        bets.write_text("b\n")
        backup_dir = tmp_path / "backups"

        for i in range(10):
            now = datetime(2026, 9, 19, 0, 0, i)
            backup_logs.backup_file(audit, backup_dir=backup_dir, keep=7, now=now)
        backup_logs.backup_file(bets, backup_dir=backup_dir, keep=7, now=datetime(2026, 9, 19, 1, 0, 0))

        bets_backups = list(backup_dir.glob("backup_*_value_bets_log.csv"))
        assert len(bets_backups) == 1


class TestBackupLogs:
    def test_backs_up_both_default_files(self, tmp_path, monkeypatch):
        audit = tmp_path / "prediction_audit_log.csv"
        bets = tmp_path / "value_bets_log.csv"
        audit.write_text("a\n")
        bets.write_text("b\n")
        backup_dir = tmp_path / "backups"

        created = backup_logs.backup_logs(
            files=(audit, bets), backup_dir=backup_dir, now=datetime(2026, 9, 19),
        )

        assert len(created) == 2

    def test_skips_missing_files_without_crashing(self, tmp_path):
        audit = tmp_path / "prediction_audit_log.csv"
        audit.write_text("a\n")
        missing = tmp_path / "does_not_exist.csv"
        backup_dir = tmp_path / "backups"

        created = backup_logs.backup_logs(
            files=(audit, missing), backup_dir=backup_dir, now=datetime(2026, 9, 19),
        )

        assert len(created) == 1

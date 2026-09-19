"""Tests for scripts/run_prediction.py's --log support."""
from __future__ import annotations

import sys
from datetime import date

import scripts.run_prediction as run_prediction


class TestLogPathForToday:
    def test_creates_logs_directory(self, tmp_path):
        path = run_prediction.log_path_for_today(root=tmp_path)
        assert (tmp_path / "data" / "logs").is_dir()

    def test_uses_todays_date_in_filename(self, tmp_path):
        path = run_prediction.log_path_for_today(root=tmp_path)
        assert path.name == f"daily_{date.today().isoformat()}.log"

    def test_path_is_under_data_logs(self, tmp_path):
        path = run_prediction.log_path_for_today(root=tmp_path)
        assert path.parent == tmp_path / "data" / "logs"


class TestRunWithLogFile:
    def test_without_log_file_behaves_as_before(self):
        # No log_file -- default capture_output=False path, must still work.
        code = run_prediction.run([sys.executable, "-c", "print('hello')"])
        assert code == 0

    def test_writes_subprocess_output_to_log_file(self, tmp_path):
        log_path = tmp_path / "test.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            run_prediction.run(
                [sys.executable, "-c", "print('hello from subprocess')"],
                log_file=log_file,
            )
        content = log_path.read_text(encoding="utf-8")
        assert "hello from subprocess" in content

    def test_writes_command_header_to_log_file(self, tmp_path):
        log_path = tmp_path / "test.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            run_prediction.run(
                [sys.executable, "-c", "print('x')"],
                log_file=log_file,
            )
        content = log_path.read_text(encoding="utf-8")
        assert sys.executable in content

    def test_failure_message_written_to_log_file(self, tmp_path):
        log_path = tmp_path / "test.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            try:
                run_prediction.run(
                    [sys.executable, "-c", "import sys; sys.exit(1)"],
                    log_file=log_file,
                )
            except SystemExit:
                pass
        content = log_path.read_text(encoding="utf-8")
        assert "ERROR" in content

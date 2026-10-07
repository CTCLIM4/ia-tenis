"""Tests for scripts/run_prediction.py's --log support."""
from __future__ import annotations

import sys
from datetime import date

import scripts.run_prediction as run_prediction
import scripts.daily_workflow as daily_workflow


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


def test_runner_dispatches_to_daily_workflow_without_push(monkeypatch):
    calls = []
    monkeypatch.setattr(run_prediction, "run", lambda cmd, **kw: calls.append(cmd) or 0)
    monkeypatch.setattr(sys, "argv", ["run_prediction.py", "--no-download", "--tour", "wta", "--days-ahead", "2"])

    run_prediction.main()

    assert calls[-1] == [
        sys.executable, "-m", "scripts.daily_workflow", "--tour", "wta",
        "--days-ahead", "2", "--no-push",
    ]
    assert not any("src.daily_scanner" in cmd for cmd in calls)
    parsed = daily_workflow.build_parser().parse_args(calls[-1][3:])
    assert (parsed.tour, parsed.days_ahead, parsed.no_push) == ("wta", 2, True)
    assert len(calls) == 3  # pipeline, retrain, daily workflow


def test_runner_dry_run_reaches_daily_workflow(monkeypatch):
    calls = []
    monkeypatch.setattr(run_prediction, "run", lambda cmd, **kw: calls.append(cmd) or 0)
    monkeypatch.setattr(sys, "argv", ["run_prediction.py", "--no-download", "--dry-run", "--tour", "atp"])

    run_prediction.main()

    args = daily_workflow.build_parser().parse_args(calls[-1][3:])
    assert args.dry_run is True
    assert args.no_push is True
    assert args.tour == "atp"

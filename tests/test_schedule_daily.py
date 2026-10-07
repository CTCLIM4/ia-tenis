"""Tests for scripts/schedule_daily.py -- Windows Scheduled Task setup for
scripts/run_prediction.py. Every test mocks subprocess.run; none of them
touch the real schtasks database."""
from __future__ import annotations

import shlex
from pathlib import Path

import pytest

import scripts.run_prediction as run_prediction
import scripts.schedule_daily as schedule_daily


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestValidateTime:
    def test_schedule_validates_time(self):
        schedule_daily._validate_time(8, 0)  # must not raise
        schedule_daily._validate_time(0, 0)
        schedule_daily._validate_time(23, 59)

    def test_rejects_hour_out_of_range(self):
        with pytest.raises(ValueError):
            schedule_daily._validate_time(24, 0)
        with pytest.raises(ValueError):
            schedule_daily._validate_time(-1, 0)

    def test_rejects_minute_out_of_range(self):
        with pytest.raises(ValueError):
            schedule_daily._validate_time(8, 60)
        with pytest.raises(ValueError):
            schedule_daily._validate_time(8, -1)


class TestTaskExists:
    def test_true_when_query_succeeds(self, monkeypatch):
        monkeypatch.setattr(
            schedule_daily.subprocess, "run",
            lambda *a, **kw: _FakeCompletedProcess(returncode=0),
        )
        assert schedule_daily.task_exists("some-task") is True

    def test_false_when_query_fails(self, monkeypatch):
        monkeypatch.setattr(
            schedule_daily.subprocess, "run",
            lambda *a, **kw: _FakeCompletedProcess(returncode=1),
        )
        assert schedule_daily.task_exists("some-task") is False


class TestCreateTask:
    def test_schedule_creates_task(self, monkeypatch, tmp_path):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(schedule_daily.subprocess, "run", fake_run)
        monkeypatch.setattr(schedule_daily, "_ROOT", tmp_path)

        result = schedule_daily.create_task(hour=8, minute=0, task_name="test-task")

        assert result.returncode == 0
        create_calls = [c for c in calls if "/create" in c]
        assert len(create_calls) == 1
        cmd = create_calls[0]
        assert "/tn" in cmd and "test-task" in cmd
        assert "/st" in cmd and "08:00" in cmd
        assert "/sc" in cmd and "daily" in cmd

    def test_schedule_removes_existing_task(self, monkeypatch, tmp_path):
        # Idempotent: creating a task that already exists must delete the
        # old one first, never duplicate it.
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "/query" in cmd:
                return _FakeCompletedProcess(returncode=0)  # task already exists
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(schedule_daily.subprocess, "run", fake_run)
        monkeypatch.setattr(schedule_daily, "_ROOT", tmp_path)

        schedule_daily.create_task(hour=8, minute=0, task_name="test-task")

        delete_calls = [c for c in calls if "/delete" in c]
        assert len(delete_calls) == 1
        # /delete must happen before /create
        delete_idx = calls.index(delete_calls[0])
        create_idx = calls.index(next(c for c in calls if "/create" in c))
        assert delete_idx < create_idx

    def test_skips_delete_when_task_does_not_exist(self, monkeypatch, tmp_path):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "/query" in cmd:
                return _FakeCompletedProcess(returncode=1)  # doesn't exist
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(schedule_daily.subprocess, "run", fake_run)
        monkeypatch.setattr(schedule_daily, "_ROOT", tmp_path)

        schedule_daily.create_task(hour=8, minute=0, task_name="test-task")

        assert not [c for c in calls if "/delete" in c]

    def test_rejects_invalid_time_before_touching_schtasks(self, monkeypatch, tmp_path):
        def boom(*a, **kw):
            raise AssertionError("must not call schtasks with an invalid time")

        monkeypatch.setattr(schedule_daily.subprocess, "run", boom)
        monkeypatch.setattr(schedule_daily, "_ROOT", tmp_path)

        with pytest.raises(ValueError):
            schedule_daily.create_task(hour=25, minute=0, task_name="test-task")

    def test_generated_command_is_accepted_by_run_prediction_argparse(self, monkeypatch, tmp_path):
        # Regression: the /tr command previously hardcoded "--auto-save",
        # but run_prediction.py has no such CLI flag (auto-save is baked
        # into step 4 internally) -- so schtasks ran a command that failed
        # argparse with exit code 2, before any pipeline step ever ran.
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(schedule_daily.subprocess, "run", fake_run)
        monkeypatch.setattr(schedule_daily, "_ROOT", tmp_path)

        schedule_daily.create_task(hour=8, minute=0, task_name="test-task")

        create_cmd = next(c for c in calls if "/create" in c)
        launcher = create_cmd[create_cmd.index("/tr") + 1].strip('"')
        command_str = Path(launcher).read_text(encoding="utf-8").splitlines()[-1]
        parts = shlex.split(command_str, posix=False)
        script_args = [p.strip('"') for p in parts[2:]]

        try:
            run_prediction.build_parser().parse_args(script_args)
        except SystemExit:
            pytest.fail(
                f"run_prediction.py rejects the scheduled command's arguments: {script_args}"
            )

    def test_task_command_fits_schtasks_limit_in_deep_checkout(self, monkeypatch, tmp_path):
        # Regression (2026-10-07): with the full venv + script paths of the
        # production checkout, the /tr value exceeded schtasks' 261-char
        # limit and registration failed. The task must point at a short
        # launcher whose content is the real command, run from the repo root.
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _FakeCompletedProcess(returncode=0)

        deep_root = tmp_path / ("x" * 60) / ("y" * 60)
        monkeypatch.setattr(schedule_daily.subprocess, "run", fake_run)
        monkeypatch.setattr(schedule_daily, "_ROOT", deep_root)
        monkeypatch.setattr(schedule_daily.sys, "executable", str(deep_root / ("venv" * 20) / "python.exe"))

        schedule_daily.create_task(hour=8, minute=0, task_name="test-task")

        create_cmd = next(c for c in calls if "/create" in c)
        task_command = create_cmd[create_cmd.index("/tr") + 1]
        assert len(task_command) <= schedule_daily.MAX_TR_LENGTH
        content = Path(task_command.strip('"')).read_text(encoding="utf-8")
        assert f'cd /d "{deep_root}"' in content
        assert "run_prediction.py" in content.splitlines()[-1]

    def test_creates_log_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            schedule_daily.subprocess, "run",
            lambda cmd, **kw: _FakeCompletedProcess(returncode=0),
        )
        monkeypatch.setattr(schedule_daily, "_ROOT", tmp_path)

        schedule_daily.create_task(hour=8, minute=0, task_name="test-task")

        assert (tmp_path / "data" / "logs").is_dir()


class TestCreateOddsSnapshotTask:
    def _create(self, monkeypatch, root, **kw):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _FakeCompletedProcess(returncode=0)

        monkeypatch.setattr(schedule_daily.subprocess, "run", fake_run)
        monkeypatch.setattr(schedule_daily, "_ROOT", root)
        schedule_daily.create_odds_snapshot_task(task_name="test-snap", **kw)
        return next(c for c in calls if "/create" in c)

    def test_runs_hourly_every_n_hours(self, monkeypatch, tmp_path):
        cmd = self._create(monkeypatch, tmp_path, hour=1, minute=0, every_hours=6)
        assert cmd[cmd.index("/sc") + 1] == "hourly"
        assert cmd[cmd.index("/mo") + 1] == "6"
        assert cmd[cmd.index("/st") + 1] == "01:00"

    def test_rejects_invalid_interval_before_touching_schtasks(self, monkeypatch, tmp_path):
        monkeypatch.setattr(schedule_daily.subprocess, "run",
                            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("schtasks called")))
        with pytest.raises(ValueError):
            schedule_daily.create_odds_snapshot_task(every_hours=0, task_name="test-snap")

    def test_generated_command_is_accepted_by_snapshot_odds_argparse(self, monkeypatch, tmp_path):
        # Same lesson as the daily task: the launcher's real command must
        # parse against the target script's real argparse.
        import scripts.snapshot_odds as snapshot_odds

        deep_root = tmp_path / ("x" * 60) / ("y" * 60)
        monkeypatch.setattr(schedule_daily.sys, "executable", str(deep_root / ("venv" * 20) / "python.exe"))
        cmd = self._create(monkeypatch, deep_root)
        task_command = cmd[cmd.index("/tr") + 1]
        assert len(task_command) <= schedule_daily.MAX_TR_LENGTH
        lines = Path(task_command.strip('"')).read_text(encoding="utf-8").splitlines()
        assert lines[1] == f'cd /d "{deep_root}"'
        parts = shlex.split(lines[-1], posix=False)
        assert parts[1].strip('"').endswith("snapshot_odds.py")
        script_args = [p.strip('"') for p in parts[2:parts.index(">>")]]
        try:
            snapshot_odds.build_parser().parse_args(script_args)
        except SystemExit:
            pytest.fail(f"snapshot_odds.py rejects the scheduled command's arguments: {script_args}")


class TestBatterySettings:
    """Regression (2026-10-07): schtasks' defaults keep a task from starting
    on battery, so on the production laptop it just sat "Queued"."""

    def _calls(self, monkeypatch, tmp_path, create, create_rc=0):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return _FakeCompletedProcess(returncode=create_rc if "/create" in cmd else 0)

        monkeypatch.setattr(schedule_daily.subprocess, "run", fake_run)
        monkeypatch.setattr(schedule_daily, "_ROOT", tmp_path)
        result = create()
        return calls, result

    @pytest.mark.parametrize("which", ["daily", "snapshots"])
    def test_allows_battery_after_creating(self, monkeypatch, tmp_path, which):
        create = (lambda: schedule_daily.create_task(task_name="t")) if which == "daily" \
            else (lambda: schedule_daily.create_odds_snapshot_task(task_name="t"))
        calls, result = self._calls(monkeypatch, tmp_path, create)
        assert result.returncode == 0
        create_idx = next(i for i, c in enumerate(calls) if "/create" in c)
        ps = calls[create_idx + 1]
        assert ps[0] == "powershell"
        script = ps[-1]
        for flag in ("-AllowStartIfOnBatteries", "-DontStopIfGoingOnBatteries", "-StartWhenAvailable"):
            assert flag in script
        assert "-TaskName 't'" in script

    def test_skips_settings_when_create_fails(self, monkeypatch, tmp_path):
        calls, result = self._calls(
            monkeypatch, tmp_path, lambda: schedule_daily.create_task(task_name="t"), create_rc=1)
        assert result.returncode == 1
        assert not [c for c in calls if c[0] == "powershell"]

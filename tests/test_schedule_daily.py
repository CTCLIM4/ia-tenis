"""Tests for scripts/schedule_daily.py -- Windows Scheduled Task setup for
scripts/run_prediction.py. Every test mocks subprocess.run; none of them
touch the real schtasks database."""
from __future__ import annotations

import pytest

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

    def test_creates_log_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            schedule_daily.subprocess, "run",
            lambda cmd, **kw: _FakeCompletedProcess(returncode=0),
        )
        monkeypatch.setattr(schedule_daily, "_ROOT", tmp_path)

        schedule_daily.create_task(hour=8, minute=0, task_name="test-task")

        assert (tmp_path / "data" / "logs").is_dir()

"""Tests for src/git_utils.py — shared git introspection helpers."""
from __future__ import annotations

import subprocess

import src.git_utils as git_utils


class _FakeCompletedProcess:
    def __init__(self, stdout):
        self.stdout = stdout


class TestCurrentGitCommit:
    def test_returns_stripped_short_hash_on_success(self, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: _FakeCompletedProcess("abc1234\n"),
        )
        assert git_utils.current_git_commit() == "abc1234"

    def test_returns_none_when_git_binary_missing(self, monkeypatch):
        def _boom(*a, **kw):
            raise FileNotFoundError("git not found")
        monkeypatch.setattr(subprocess, "run", _boom)
        assert git_utils.current_git_commit() is None

    def test_returns_none_when_not_a_git_repo(self, monkeypatch):
        def _boom(*a, **kw):
            raise subprocess.CalledProcessError(128, ["git"])
        monkeypatch.setattr(subprocess, "run", _boom)
        assert git_utils.current_git_commit() is None

    def test_returns_none_on_timeout(self, monkeypatch):
        def _boom(*a, **kw):
            raise subprocess.TimeoutExpired(["git"], 5)
        monkeypatch.setattr(subprocess, "run", _boom)
        assert git_utils.current_git_commit() is None

    def test_real_invocation_returns_a_string_or_none(self):
        # Sanity check against the real subprocess call (this repo IS a git
        # checkout during tests, so this should return a real short hash) —
        # not mocked, catches a totally broken implementation end-to-end.
        result = git_utils.current_git_commit()
        assert result is None or (isinstance(result, str) and len(result) >= 4)


class _FakeRun:
    """Records every subprocess.run call and returns canned porcelain status."""

    def __init__(self, porcelain_stdout: str = " M data/value_bets_log.csv\n"):
        self.calls: list[list[str]] = []
        self.porcelain_stdout = porcelain_stdout

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if cmd[:2] == ["git", "status"]:
            return _FakeCompletedProcess(self.porcelain_stdout)
        return _FakeCompletedProcess("")


class TestCommitAndPush:
    def test_returns_true_and_runs_add_commit_push_when_there_are_changes(self, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(subprocess, "run", fake)
        result = git_utils.commit_and_push(["data/value_bets_log.csv"], "msg")
        assert result is True
        cmds = [c[1] for c in fake.calls]  # subcommand, e.g. "add"/"status"/"commit"/"push"
        assert cmds == ["add", "status", "commit", "push"]

    def test_returns_false_without_committing_when_nothing_changed(self, monkeypatch):
        fake = _FakeRun(porcelain_stdout="")
        monkeypatch.setattr(subprocess, "run", fake)
        result = git_utils.commit_and_push(["data/value_bets_log.csv"], "msg")
        assert result is False
        cmds = [c[1] for c in fake.calls]
        assert "commit" not in cmds
        assert "push" not in cmds

    def test_returns_false_on_git_failure_without_raising(self, monkeypatch):
        def _boom(*a, **kw):
            raise subprocess.CalledProcessError(1, ["git", "push"], stderr="rejected")
        monkeypatch.setattr(subprocess, "run", _boom)
        result = git_utils.commit_and_push(["data/value_bets_log.csv"], "msg")
        assert result is False

    def test_uses_custom_remote_and_branch(self, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(subprocess, "run", fake)
        git_utils.commit_and_push(["f.txt"], "msg", remote="upstream", branch="main")
        push_cmd = [c for c in fake.calls if c[1] == "push"][0]
        assert push_cmd == ["git", "push", "upstream", "main"]

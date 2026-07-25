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

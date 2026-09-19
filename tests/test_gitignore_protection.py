"""Tests that the real, hand-written-to protection status of the two
append-only log files matches .gitignore's intent: both must be tracked in
git (survive an accidental local rm/overwrite via `git checkout`), same as
value_bets_log.csv already was before this regression
(docs/metrics/2026-09-19-audit-log-protection.md).

These shell out to the real `git` binary against the actual repo working
tree -- deliberately not mocked, since the property under test ("is this
path tracked in THIS repo") is meaningless against a synthetic fixture.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _git_ls_files(path: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", path], cwd=_ROOT, capture_output=True, text=True, check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_prediction_audit_log_is_tracked():
    assert _git_ls_files("data/prediction_audit_log.csv") == ["data/prediction_audit_log.csv"]


def test_value_bets_log_is_tracked():
    # Regression guard: this one was already protected before the incident
    # that prompted this file -- must stay that way.
    assert _git_ls_files("data/value_bets_log.csv") == ["data/value_bets_log.csv"]

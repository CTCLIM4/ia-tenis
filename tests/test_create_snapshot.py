"""Tests for scripts/create_snapshot.py — thin CLI wrapper around
src.data.snapshots.create_snapshot()."""
from __future__ import annotations

import pandas as pd
import pytest

import src.data.snapshots as snapshots
from scripts import create_snapshot as cs


@pytest.fixture()
def fake_data_dirs(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    snapshot_root = tmp_path / "snapshots"
    monkeypatch.setattr(snapshots, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(snapshots, "_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(snapshots, "SNAPSHOT_ROOT", snapshot_root)

    processed_dir.mkdir(parents=True)
    for tour in ("atp", "wta"):
        pd.DataFrame({"match_date": ["2026-07-20"], "winner": ["A"], "loser": ["B"]}).to_csv(
            processed_dir / f"{tour}_features.csv", index=False
        )
    (raw_dir / "tennis_atp_tml").mkdir(parents=True)
    (raw_dir / "tennis_atp_tml" / "2026.csv").write_text("winner_name,loser_name\nA,B\n")
    (raw_dir / "tennis_wta_tduk").mkdir(parents=True)
    (raw_dir / "tennis_wta_tduk" / "2026w.csv").write_text("Winner,Loser\nA,B\n")
    return snapshot_root


class TestCreateSnapshotScript:
    def test_run_creates_snapshot_and_returns_its_path(self, fake_data_dirs):
        result = cs.run(snapshot_id="2026-07-24")
        assert result == fake_data_dirs / "2026-07-24"
        assert (result / "metadata.json").exists()

    def test_run_prints_summary(self, fake_data_dirs, capsys):
        cs.run(snapshot_id="2026-07-24")
        out = capsys.readouterr().out
        assert "2026-07-24" in out
        assert "ATP" in out
        assert "WTA" in out

    def test_run_respects_tours_subset(self, fake_data_dirs):
        result = cs.run(snapshot_id="2026-07-24", tours=("atp",))
        assert (result / "processed" / "atp_features.csv").exists()
        assert not (result / "processed" / "wta_features.csv").exists()

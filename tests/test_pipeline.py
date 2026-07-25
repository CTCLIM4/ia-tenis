"""Tests for src/pipeline.py's --snapshot support in run_pipeline()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.data.snapshots as snapshots
from src.pipeline import run_pipeline


@pytest.fixture()
def fake_snapshot(tmp_path, monkeypatch):
    """A minimal but structurally valid snapshot with a real, if tiny,
    processed features CSV — enough for walk_forward_backtest to run
    end-to-end without touching live data."""
    snapshot_root = tmp_path / "snapshots"
    monkeypatch.setattr(snapshots, "SNAPSHOT_ROOT", snapshot_root)
    snapshot_dir = snapshot_root / "2020-01-01"
    (snapshot_dir / "processed").mkdir(parents=True)

    rng = np.random.default_rng(3)
    n = 500
    elo_diff = rng.normal(0, 150, n)
    df = pd.DataFrame({
        "year": rng.integers(2000, 2015, n),
        "elo_diff": elo_diff,
        "elo_prob": 1 / (1 + 10 ** (-elo_diff / 400)),
        "rank_diff": rng.normal(0, 50, n),
        "form_diff": rng.uniform(-0.5, 0.5, n),
        "surface_form_diff": rng.uniform(-0.5, 0.5, n),
        "h2h_rate": rng.uniform(0.3, 0.7, n),
        "rest_diff": rng.normal(0, 5, n),
        "rolling_elo_diff": rng.normal(0, 30, n),
        "age_multiplier_diff": rng.uniform(-0.3, 0.3, n),
        "rust_factor_diff": rng.uniform(-0.5, 0.5, n),
        "adjusted_elo_diff": elo_diff * rng.uniform(0.6, 1.0, n),
        "outcome": 1,
        "is_mirror": False,
    })
    df.to_csv(snapshot_dir / "processed" / "atp_features.csv", index=False)
    (snapshot_dir / "metadata.json").write_text("{}")
    return snapshot_dir


class TestRunPipelineSnapshot:
    def test_reads_from_snapshot_instead_of_live_loader(self, fake_snapshot, monkeypatch, capsys):
        def _boom(*a, **kw):
            raise AssertionError("should not call the live loader when --snapshot is given")
        monkeypatch.setattr("src.pipeline.load_atp_matches", _boom)

        run_pipeline(tour="atp", warmup_years=5, snapshot="2020-01-01")

        out = capsys.readouterr().out
        assert "snapshot" in out.lower()
        assert "BACKTEST RESULTS" in out

    def test_does_not_write_to_live_processed_dir(self, fake_snapshot, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)  # the non-snapshot path writes to "data/processed" (cwd-relative)

        run_pipeline(tour="atp", warmup_years=5, snapshot="2020-01-01")

        assert not (tmp_path / "data" / "processed").exists()

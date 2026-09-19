"""Tests for src/pipeline.py's --snapshot support in run_pipeline()."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import src.data.snapshots as snapshots
from src.pipeline import run_pipeline


class _StopEarly(Exception):
    """Raised by a fake loader right after capturing its call args, to skip
    the rest of run_pipeline (feature building, backtest) — this test only
    cares about what year range reaches the loader."""


class TestRunPipelineDefaultEndYear:
    def test_defaults_end_year_to_current_year_when_omitted(self, monkeypatch):
        """Regression: run_pipeline's end_year default was hardcoded to 2023
        (and the CLI's positional arg mirrored it), so running
        `python -m src.pipeline atp` with no explicit year range silently
        rebuilt data/processed/atp_features.csv missing every match since
        2024 — discovered when this dropped the just-downloaded 2026 US Open
        results entirely. Must resolve to the actual current year instead."""
        captured = {}

        def _fake_loader(start_year, end_year, raw_dir_override=None):
            captured["start_year"] = start_year
            captured["end_year"] = end_year
            raise _StopEarly

        monkeypatch.setattr("src.pipeline.load_atp_matches", _fake_loader)

        with pytest.raises(_StopEarly):
            run_pipeline(tour="atp", warmup_years=1)

        assert captured["start_year"] == 1990
        assert captured["end_year"] == date.today().year

    def test_respects_an_explicit_end_year(self, monkeypatch):
        captured = {}

        def _fake_loader(start_year, end_year, raw_dir_override=None):
            captured["end_year"] = end_year
            raise _StopEarly

        monkeypatch.setattr("src.pipeline.load_atp_matches", _fake_loader)

        with pytest.raises(_StopEarly):
            run_pipeline(tour="atp", warmup_years=1, end_year=2020)

        assert captured["end_year"] == 2020


class TestRunPipelineDavisDispatch:
    def test_davis_tour_calls_load_davis_cup_matches_not_atp_or_wta(self, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("wrong loader called for tour='davis'")

        monkeypatch.setattr("src.pipeline.load_atp_matches", _boom)
        monkeypatch.setattr("src.pipeline.load_wta_matches", _boom)

        captured = {}

        def _fake_davis_loader(start_year, end_year, raw_dir_override=None):
            captured["start_year"] = start_year
            captured["end_year"] = end_year
            raise _StopEarly

        monkeypatch.setattr("src.pipeline.load_davis_cup_matches", _fake_davis_loader)

        with pytest.raises(_StopEarly):
            run_pipeline(tour="davis", warmup_years=1)

        assert captured["end_year"] == date.today().year


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
        "fatigue_multiplier_diff": rng.uniform(-0.15, 0.15, n),
        "surface_transition_multiplier_diff": rng.uniform(-0.10, 0.10, n),
        "adjusted_elo_diff": elo_diff * rng.uniform(0.6, 1.0, n),
        "h2h_trend": rng.uniform(-0.3, 0.3, n),
        "momentum_3_diff": rng.uniform(-0.5, 0.5, n),
        "momentum_5_diff": rng.uniform(-0.5, 0.5, n),
        "surface_win_rate_trend_diff": rng.uniform(-0.5, 0.5, n),
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

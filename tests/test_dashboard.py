"""Tests for src/dashboard.py -- backtest/calibration/feature-importance/
value-bet-history visualizations. All tests write to tmp_path, never to the
real data/dashboard/."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dashboard import (
    extract_feature_importance,
    plot_backtest_accuracy,
    plot_calibration_curve,
    plot_feature_importance,
    plot_value_bet_history,
)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_valid_png(path) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == _PNG_MAGIC


class TestPlotBacktestAccuracy:
    def test_dashboard_generates_backtest_plot(self, tmp_path):
        results_by_tour = {
            "atp": {2020: {"accuracy": 0.64, "elo_only_accuracy": 0.62, "n_matches": 100},
                    2021: {"accuracy": 0.65, "elo_only_accuracy": 0.63, "n_matches": 110}},
            "wta": {2020: {"accuracy": 0.63, "elo_only_accuracy": 0.61, "n_matches": 90},
                    2021: {"accuracy": 0.66, "elo_only_accuracy": 0.64, "n_matches": 95}},
        }
        out_path = tmp_path / "backtest_accuracy.png"

        result = plot_backtest_accuracy(results_by_tour, out_path)

        assert result == out_path
        assert out_path.exists()
        assert _is_valid_png(out_path)

    def test_creates_parent_directory(self, tmp_path):
        results_by_tour = {"atp": {2020: {"accuracy": 0.64, "elo_only_accuracy": 0.62, "n_matches": 100}}}
        out_path = tmp_path / "nested" / "dir" / "plot.png"

        plot_backtest_accuracy(results_by_tour, out_path)

        assert out_path.exists()


class TestPlotCalibrationCurve:
    def test_dashboard_calibration_curve(self, tmp_path):
        rng = np.random.default_rng(0)
        probs = rng.uniform(0.05, 0.95, 2000)
        y_true = (rng.uniform(0, 1, 2000) < probs).astype(int)
        out_path = tmp_path / "calibration.png"

        result = plot_calibration_curve(probs, y_true, out_path, bins=10)

        assert result == out_path
        assert out_path.exists()
        assert _is_valid_png(out_path)

    def test_raises_on_mismatched_lengths(self, tmp_path):
        with pytest.raises(ValueError):
            plot_calibration_curve(np.array([0.5, 0.6]), np.array([1]), tmp_path / "x.png")


class TestFeatureImportance:
    def _fake_clf(self, n_features=13, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (200, n_features))
        y = (X[:, 0] + rng.normal(0, 0.1, 200) > 0).astype(int)
        clf = make_pipeline(StandardScaler(), LogisticRegression())
        clf.fit(X, y)
        return clf

    def test_dashboard_feature_importance(self):
        feature_names = [f"f{i}" for i in range(13)]
        clf = self._fake_clf(13)

        df = extract_feature_importance(clf, feature_names)

        # Exact correctness, not just "didn't crash": every reported
        # coefficient must match the fitted LR's own coef_ array.
        lr = clf.named_steps["logisticregression"]
        for _, row in df.iterrows():
            idx = feature_names.index(row["feature"])
            assert row["coefficient"] == pytest.approx(lr.coef_[0][idx])
        # Sorted by absolute magnitude, descending.
        abs_coefs = df["coefficient"].abs().tolist()
        assert abs_coefs == sorted(abs_coefs, reverse=True)

    def test_raises_on_feature_count_mismatch(self):
        clf = self._fake_clf(13)
        with pytest.raises(ValueError):
            extract_feature_importance(clf, ["only", "two"])

    def test_plot_feature_importance_generates_file(self, tmp_path):
        feature_names = [f"f{i}" for i in range(13)]
        clf = self._fake_clf(13)
        out_path = tmp_path / "importance.png"

        result = plot_feature_importance(clf, feature_names, out_path)

        assert result == out_path
        assert out_path.exists()
        assert _is_valid_png(out_path)


class TestPlotValueBetHistory:
    def test_generates_file_from_real_shaped_log(self, tmp_path):
        log_path = tmp_path / "value_bets_log.csv"
        pd.DataFrame({
            "date": pd.date_range("2026-07-01", periods=10),
            "tour": ["atp", "wta"] * 5,
            "edge_a": [0.05, 0.03, 0.08, 0.02, 0.06, 0.04, 0.07, 0.03, 0.05, 0.09],
            "edge_b": [0.0] * 10,
            "kelly_a": [0.02, 0.01, 0.03, 0.005, 0.025, 0.015, 0.03, 0.01, 0.02, 0.04],
            "kelly_b": [0.0] * 10,
        }).to_csv(log_path, index=False)
        out_path = tmp_path / "value_bets.png"

        result = plot_value_bet_history(log_path, out_path)

        assert result == out_path
        assert out_path.exists()
        assert _is_valid_png(out_path)

    def test_missing_log_raises_filenotfounderror(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            plot_value_bet_history(tmp_path / "does_not_exist.csv", tmp_path / "out.png")

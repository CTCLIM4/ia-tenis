import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import src.backtest.walkforward as walkforward_module
from src.backtest.walkforward import _FEATURE_COLS, _MIRROR_FLIP_COLS, load_features_with_mirror, walk_forward_backtest


def _synthetic_features(n: int = 2000, n_years: int = 12, start_year: int = 2010) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    years = rng.integers(start_year, start_year + n_years, n)
    elo_diff = rng.normal(50, 150, n)
    elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))
    original = pd.DataFrame(
        {
            "year": years,
            "elo_diff": elo_diff,
            "elo_prob": elo_prob,
            "rank_diff": rng.normal(20, 80, n),
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
            "outcome": 1,
            "is_mirror": False,
        }
    )
    mirror = original.copy()
    for col in (
        "elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff",
        "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
        "fatigue_multiplier_diff", "surface_transition_multiplier_diff", "adjusted_elo_diff",
    ):
        mirror[col] = -mirror[col]
    mirror["elo_prob"] = 1.0 - mirror["elo_prob"]
    mirror["h2h_rate"] = 1.0 - mirror["h2h_rate"]
    mirror["outcome"] = 0
    mirror["is_mirror"] = True
    return pd.concat([original, mirror], ignore_index=True)


class TestLoadFeaturesWithMirror:
    def test_returns_equal_original_and_mirror_counts(self, tmp_path):
        path = tmp_path / "atp_features.csv"
        full = _synthetic_features(n=50)
        full[~full["is_mirror"]].to_csv(path, index=False)

        result = load_features_with_mirror(path)

        n_original = int((~result["is_mirror"]).sum())
        n_mirror = int(result["is_mirror"].sum())
        assert n_original == n_mirror == 50
        assert len(result) == 100

    def test_mirror_rows_have_flipped_diff_columns(self, tmp_path):
        path = tmp_path / "atp_features.csv"
        full = _synthetic_features(n=10)
        full[~full["is_mirror"]].to_csv(path, index=False)

        result = load_features_with_mirror(path)
        original = result[~result["is_mirror"]].reset_index(drop=True)
        mirror = result[result["is_mirror"]].reset_index(drop=True)

        for col in _MIRROR_FLIP_COLS:
            assert (mirror[col] == -original[col]).all()

    def test_mirror_rows_have_outcome_zero_original_have_one(self, tmp_path):
        path = tmp_path / "atp_features.csv"
        full = _synthetic_features(n=10)
        full[~full["is_mirror"]].to_csv(path, index=False)

        result = load_features_with_mirror(path)

        assert (result[result["is_mirror"]]["outcome"] == 0).all()
        assert (result[~result["is_mirror"]]["outcome"] == 1).all()


def test_returns_metrics_dict_with_one_entry_per_test_year():
    df = _synthetic_features()
    results = walk_forward_backtest(df, warmup_years=5)
    assert len(results) > 0
    all_years = sorted(df["year"].unique())
    expected_test_years = all_years[5:]
    assert set(results.keys()) == set(expected_test_years)


def test_each_year_has_required_metrics():
    df = _synthetic_features()
    results = walk_forward_backtest(df, warmup_years=5)
    required = {"accuracy", "log_loss", "brier_score", "n_matches", "elo_only_accuracy", "elo_only_log_loss"}
    for year, m in results.items():
        assert required.issubset(m.keys()), f"Year {year} missing keys: {required - m.keys()}"


def test_metrics_are_in_valid_ranges():
    df = _synthetic_features()
    results = walk_forward_backtest(df, warmup_years=5)
    for year, m in results.items():
        assert 0.0 <= m["accuracy"] <= 1.0, f"Year {year}: accuracy={m['accuracy']}"
        assert m["log_loss"] >= 0.0, f"Year {year}: log_loss={m['log_loss']}"
        assert 0.0 <= m["brier_score"] <= 1.0, f"Year {year}: brier={m['brier_score']}"
        assert 0.0 <= m["elo_only_accuracy"] <= 1.0


def test_more_warmup_years_leaves_fewer_test_years():
    df = _synthetic_features()
    r3 = walk_forward_backtest(df, warmup_years=3)
    r7 = walk_forward_backtest(df, warmup_years=7)
    assert len(r3) >= len(r7)


def test_n_matches_correct_per_year():
    df = _synthetic_features()
    results = walk_forward_backtest(df, warmup_years=5)
    for year, m in results.items():
        expected = int(((df["year"] == year) & (~df["is_mirror"])).sum())
        assert m["n_matches"] == expected


class _SpyScaler:
    """Records exactly what raw data each fit_transform/transform call
    received, then delegates to a real StandardScaler — lets tests assert
    the scaler was fit on X_train only (no leakage from X_test) without
    reimplementing StandardScaler's math."""

    instances: list["_SpyScaler"] = []

    def __init__(self):
        self.fit_transform_calls: list[np.ndarray] = []
        self.transform_calls: list[np.ndarray] = []
        self._real = StandardScaler()
        _SpyScaler.instances.append(self)

    def fit_transform(self, X):
        self.fit_transform_calls.append(np.array(X, copy=True))
        return self._real.fit_transform(X)

    def transform(self, X):
        self.transform_calls.append(np.array(X, copy=True))
        return self._real.transform(X)


class TestFeatureScaling:
    def test_scaler_is_fit_on_train_and_applied_to_test_with_no_leakage(self, monkeypatch):
        _SpyScaler.instances = []
        monkeypatch.setattr(walkforward_module, "StandardScaler", _SpyScaler)

        df = _synthetic_features()
        results = walk_forward_backtest(df, warmup_years=5)

        assert len(_SpyScaler.instances) == len(results), (
            "expected one fresh scaler per evaluated year, matching the "
            "existing per-year LogisticRegression retraining pattern"
        )

        for test_year, spy in zip(sorted(results.keys()), _SpyScaler.instances):
            train_df = df[df["year"] < test_year]
            test_df = df[(df["year"] == test_year) & (~df["is_mirror"])]
            expected_X_train = train_df[_FEATURE_COLS].fillna(0.0).values
            expected_X_test = test_df[_FEATURE_COLS].fillna(0.0).values

            assert len(spy.fit_transform_calls) == 1
            assert len(spy.transform_calls) == 1
            np.testing.assert_array_equal(spy.fit_transform_calls[0], expected_X_train)
            np.testing.assert_array_equal(spy.transform_calls[0], expected_X_test)

    def test_scaled_train_features_are_approximately_standardized(self):
        # Sanity check that scaling actually happens (not silently unused):
        # each feature column of X_train should end up ~zero-mean/unit-variance
        # after StandardScaler, verified directly against the real scaler.
        df = _synthetic_features()
        warmup_years = 5
        test_year = sorted(df[~df["is_mirror"]]["year"].unique())[warmup_years]
        train_df = df[df["year"] < test_year]
        X_train = train_df[_FEATURE_COLS].fillna(0.0).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-8)
        np.testing.assert_allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-8)

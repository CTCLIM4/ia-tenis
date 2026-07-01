import numpy as np
import pandas as pd
import pytest
from src.backtest.walkforward import walk_forward_backtest


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
            "outcome": 1,
            "is_mirror": False,
        }
    )
    mirror = original.copy()
    for col in ("elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff"):
        mirror[col] = -mirror[col]
    mirror["elo_prob"] = 1.0 - mirror["elo_prob"]
    mirror["h2h_rate"] = 1.0 - mirror["h2h_rate"]
    mirror["outcome"] = 0
    mirror["is_mirror"] = True
    return pd.concat([original, mirror], ignore_index=True)


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

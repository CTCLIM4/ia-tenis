"""Tests for src/calibration_metrics.py -- reliability/calibration analysis
against the walk-forward backtest's full mirrored predictions (large N),
never the small, noisy value_bets_log.csv real-bet sample (N=108, high odds
variance -- see docs/metrics/2026-09-18-calibration-investigation.md for why
that distinction matters)."""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration_metrics import calibration_error, compare_shrink_rates


def _perfectly_calibrated(n=6000, seed=0):
    """Synthetic (probs, y_true) where, by construction, the realized
    frequency in every probability band matches the predicted probability."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(0.0, 1.0, n) < probs).astype(int)
    return probs, y


def _overconfident(n=6000, seed=0, bias=0.20):
    """Synthetic data where predicted probabilities are systematically more
    extreme than the true win rate (predictions in [0.5, 1.0] shifted up by
    `bias`, actual outcomes stay at the true, unshifted rate)."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.3, 0.7, n)
    y = (rng.uniform(0.0, 1.0, n) < true_p).astype(int)
    probs = np.clip(true_p + np.where(true_p >= 0.5, bias, -bias), 0.01, 0.99)
    return probs, y


class TestCalibrationError:
    def test_perfectly_calibrated_data_gives_near_zero_error(self):
        probs, y = _perfectly_calibrated()
        err = calibration_error(probs, y, lo=0.20, hi=0.80, bins=6)
        assert err < 0.02  # < 2pp, allowing for sampling noise at N=6000

    def test_overconfident_predictions_show_meaningful_error(self):
        probs, y = _overconfident(bias=0.20)
        err = calibration_error(probs, y, lo=0.20, hi=0.80, bins=6)
        assert err > 0.10  # a 20pp systematic bias must show up clearly

    def test_error_increases_with_bias_magnitude(self):
        probs_small, y_small = _overconfident(bias=0.05, seed=1)
        probs_big, y_big = _overconfident(bias=0.25, seed=1)
        err_small = calibration_error(probs_small, y_small, lo=0.20, hi=0.80, bins=6)
        err_big = calibration_error(probs_big, y_big, lo=0.20, hi=0.80, bins=6)
        assert err_big > err_small

    def test_respects_lo_hi_band(self):
        # Predictions outside [lo, hi] must not influence the result --
        # only predictions inside the requested band are binned.
        rng = np.random.default_rng(2)
        n = 4000
        probs = np.concatenate([rng.uniform(0.3, 0.7, n), np.full(n, 0.99)])
        y = np.concatenate([(rng.uniform(0, 1, n) < probs[:n]).astype(int), np.zeros(n, dtype=int)])
        err_band = calibration_error(probs, y, lo=0.20, hi=0.80, bins=6)
        err_band_only = calibration_error(probs[:n], y[:n], lo=0.20, hi=0.80, bins=6)
        assert err_band == pytest.approx(err_band_only)

    def test_empty_band_returns_nan(self):
        probs = np.array([0.95, 0.97, 0.99])
        y = np.array([1, 0, 1])
        assert np.isnan(calibration_error(probs, y, lo=0.20, hi=0.80, bins=6))


class TestCalibrationByTour:
    """The real per-tour comparison (real ATP/WTA/Davis backtest data) is an
    empirical finding, not a unit-testable invariant -- see
    docs/metrics/2026-09-18-calibration-investigation.md for those numbers.
    This pins down that calibration_error works correctly across datasets
    shaped like the three real tours (sample sizes ATP >> WTA >> Davis)."""

    def test_calibration_by_tour(self):
        sizes = {"atp": 160_000, "wta": 50_000, "davis": 15_000}
        errors = {}
        for tour, n in sizes.items():
            probs, y = _perfectly_calibrated(n=n, seed=hash(tour) % (2**31))
            errors[tour] = calibration_error(probs, y, lo=0.20, hi=0.80, bins=6)
        for tour, err in errors.items():
            assert err < 0.03, f"{tour}: calibration error {err*100:.2f}pp >= 3pp"


class TestCompareShrinkRates:
    def test_returns_one_error_per_rate(self):
        probs, y = _perfectly_calibrated()
        result = compare_shrink_rates(probs, y, rates=(0.0, 0.4, 0.6), shrink_lo=0.10, shrink_hi=0.90)
        assert set(result.keys()) == {0.0, 0.4, 0.6}

    def test_shrink_rate_has_no_effect_inside_the_unshrunk_band(self):
        # apply_shrinkage only transforms predictions outside
        # [shrink_lo, shrink_hi] by construction -- calibration measured
        # strictly inside that band must be identical across every rate.
        probs, y = _perfectly_calibrated()
        result = compare_shrink_rates(
            probs, y, rates=(0.0, 0.4, 0.6, 0.8), shrink_lo=0.10, shrink_hi=0.90,
            eval_lo=0.20, eval_hi=0.80,
        )
        values = list(result.values())
        assert all(v == pytest.approx(values[0]) for v in values)

    def test_shrinking_extreme_overconfidence_reduces_full_range_error(self):
        # Predictions pinned near the extremes (>0.90) that are actually
        # overconfident should calibrate better under a nonzero shrink rate
        # when measured over the FULL range (not the unshrunk band).
        rng = np.random.default_rng(3)
        n = 6000
        true_p = np.clip(rng.uniform(0.85, 0.98, n), 0.01, 0.99)
        y = (rng.uniform(0, 1, n) < (true_p - 0.15)).astype(int)  # true rate is lower
        result = compare_shrink_rates(
            true_p, y, rates=(0.0, 0.6), shrink_lo=0.10, shrink_hi=0.90,
            eval_lo=0.0, eval_hi=1.0,
        )
        assert result[0.6] < result[0.0]

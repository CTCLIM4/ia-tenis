"""Reliability/calibration analysis for the LR model's predicted
probabilities, measured against the walk-forward backtest's large,
mirror-balanced sample -- never against data/value_bets_log.csv (N~100,
high odds variance), which measures realized financial variance, not
calibration. See docs/metrics/2026-09-18-calibration-investigation.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_error(
    probs: np.ndarray, y_true: np.ndarray,
    lo: float = 0.20, hi: float = 0.80, bins: int = 6,
) -> float:
    """Weighted mean absolute calibration error (|mean predicted prob -
    actual win frequency| per bin, weighted by bin size) for predictions
    within [lo, hi]. NaN if no predictions fall in that band.

    This is a reliability measure, not accuracy: a model can be perfectly
    calibrated (this returns ~0) while still being a weak predictor (e.g.
    always predicting the base rate), and vice versa.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    mask = (probs >= lo) & (probs <= hi)
    if not mask.any():
        return float("nan")

    df = pd.DataFrame({"p": probs[mask], "y": y_true[mask]})
    df["bin"] = pd.cut(df["p"], bins=bins)
    grouped = df.groupby("bin", observed=True).agg(
        mean_p=("p", "mean"), mean_y=("y", "mean"), n=("y", "size"),
    )
    if grouped.empty:
        return float("nan")
    weights = grouped["n"] / grouped["n"].sum()
    return float((weights * (grouped["mean_p"] - grouped["mean_y"]).abs()).sum())


def _shrink(p: float, rate: float, shrink_lo: float, shrink_hi: float) -> float:
    if p > shrink_hi:
        return shrink_hi + (p - shrink_hi) * (1 - rate)
    if p < shrink_lo:
        return shrink_lo - (shrink_lo - p) * (1 - rate)
    return p


def compare_shrink_rates(
    probs: np.ndarray, y_true: np.ndarray, rates: tuple,
    shrink_lo: float, shrink_hi: float,
    eval_lo: float = 0.0, eval_hi: float = 1.0, bins: int = 10,
) -> dict:
    """Calibration error (see calibration_error) after applying each
    candidate shrink rate to `probs`, evaluated over [eval_lo, eval_hi].

    Mirrors the shrinkage transform in src.value_analysis.apply_shrinkage
    exactly (reimplemented here, vectorization aside, to avoid importing
    src.value_analysis's module-level _SHRINK_RATE constant into a
    comparison function whose whole point is trying other rates).
    """
    probs = np.asarray(probs)
    out = {}
    for rate in rates:
        shrunk = np.array([_shrink(p, rate, shrink_lo, shrink_hi) for p in probs])
        out[rate] = calibration_error(shrunk, y_true, lo=eval_lo, hi=eval_hi, bins=bins)
    return out

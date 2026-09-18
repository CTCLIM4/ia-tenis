"""End-to-end pipeline: load data -> features -> backtest."""
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtest.walkforward import build_match_features, load_features_with_mirror, walk_forward_backtest
from src.data.loader import load_atp_matches, load_davis_cup_matches, load_wta_matches
from src.data.snapshots import resolve_snapshot_path
from src.features.engineering import FeatureBuilder
from src.models.elo import EloSystem


def run_pipeline(
    tour: str = "atp",
    start_year: int = 1990,
    end_year: Optional[int] = None,
    warmup_years: int = 10,
    eval_start: int = None,
    snapshot: Optional[str] = None,
) -> None:
    if end_year is None:
        end_year = date.today().year

    if snapshot is not None:
        features_path = resolve_snapshot_path(snapshot, tour, "processed")
        print(f"Loading {tour.upper()} features from snapshot '{snapshot}' ({features_path})...")
        match_df = load_features_with_mirror(features_path)
        original = match_df[~match_df["is_mirror"]]
        print(f"  {len(original):,} feature rows ({len(match_df):,} with mirrors) — pinned, not regenerated.")
    else:
        print(f"Loading {tour.upper()} matches {start_year}-{end_year}...")
        loader = (
            load_atp_matches if tour == "atp"
            else load_wta_matches if tour == "wta"
            else load_davis_cup_matches
        )
        df = loader(start_year, end_year)
        print(f"  Loaded {len(df):,} matches across {df['year'].nunique()} seasons")

        elo = EloSystem()
        fb = FeatureBuilder()

        print("Building features (sequential, no lookahead)...")
        match_df = build_match_features(df, elo, fb)
        original = match_df[~match_df["is_mirror"]]
        print(f"  {len(original):,} feature rows ({len(match_df):,} with mirrors)")

        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / f"{tour}_features.csv"
        original.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")

    print(f"\nWalk-forward backtest (warmup={warmup_years} years)...")
    results = walk_forward_backtest(match_df, warmup_years=warmup_years)

    # Optional filter: only report years >= eval_start
    if eval_start is not None:
        results = {y: m for y, m in results.items() if y >= eval_start}
        print(f"  (reporting years >= {eval_start})")

    header = f"{'Year':>6} | {'Acc':>7} | {'LogLoss':>8} | {'Brier':>7} | {'EloAcc':>7} | {'EloLL':>8} | {'N':>6}"
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  BACKTEST RESULTS")
    print(sep)
    print(header)
    print("-" * len(header))
    for year, m in sorted(results.items()):
        print(
            f"{year:>6} | {m['accuracy']:>7.4f} | {m['log_loss']:>8.4f} | "
            f"{m['brier_score']:>7.4f} | {m['elo_only_accuracy']:>7.4f} | "
            f"{m['elo_only_log_loss']:>8.4f} | {m['n_matches']:>6}"
        )
    print(sep)

    accs = [m["accuracy"] for m in results.values()]
    elo_accs = [m["elo_only_accuracy"] for m in results.values()]
    lls = [m["log_loss"] for m in results.values()]
    elo_lls = [m["elo_only_log_loss"] for m in results.values()]
    print(f"\n  Mean Accuracy : {np.mean(accs):.4f}  (Elo-only: {np.mean(elo_accs):.4f})")
    print(f"  Mean Log-Loss : {np.mean(lls):.4f}  (Elo-only: {np.mean(elo_lls):.4f})")
    print(f"  Years tested  : {len(results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ia-tenis pipeline")
    parser.add_argument("tour",        nargs="?", default="atp")
    parser.add_argument("start_year",  nargs="?", type=int, default=1990)
    parser.add_argument("end_year",    nargs="?", type=int, default=None)
    parser.add_argument("--warmup",    type=int,  default=10,   dest="warmup_years")
    parser.add_argument("--eval-start", type=int, default=None, dest="eval_start")
    parser.add_argument("--snapshot",  type=str,  default=None,
                        help="Usar un snapshot pinned (data/snapshots/{id}/) en vez de datos en vivo")
    args = parser.parse_args()

    run_pipeline(
        tour=args.tour,
        start_year=args.start_year,
        end_year=args.end_year,
        warmup_years=args.warmup_years,
        eval_start=args.eval_start,
        snapshot=args.snapshot,
    )

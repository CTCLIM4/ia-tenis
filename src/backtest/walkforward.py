from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from src.features.decay import EloHistoryTracker, calculate_decay_features
from src.features.engineering import FeatureBuilder
from src.models.elo import EloSystem

_FEATURE_COLS = [
    "elo_diff",
    "elo_prob",
    "rank_diff",
    "form_diff",
    "surface_form_diff",
    "h2h_rate",
    "rest_diff",
    "rolling_elo_diff",
    "age_multiplier_diff",
    "rust_factor_diff",
    "adjusted_elo_diff",
]

_MIRROR_FLIP_COLS = (
    "elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff", "adjusted_elo_diff",
)


def load_features_with_mirror(path: Path) -> pd.DataFrame:
    """Load a {tour}_features.csv (as saved by src/pipeline.py's
    run_pipeline — original rows only, is_mirror=False for every row) and
    reconstruct the mirrored rows (loser's perspective, outcome=0) exactly
    as build_match_features() does internally.

    The saved CSV only persists 'original' rows to avoid doubling file size
    on disk — every consumer that needs the full mirrored training set
    (LR fitting in src/value_analysis.py's _train_lr, walk_forward_backtest
    when reading a pinned snapshot in src/pipeline.py) reconstructs it via
    this single function instead of duplicating the mirroring logic.
    """
    df = pd.read_csv(path)
    df["is_mirror"] = False
    mirror = df.copy()
    for col in _MIRROR_FLIP_COLS:
        mirror[col] = -mirror[col]
    mirror["elo_prob"] = 1 - mirror["elo_prob"]
    mirror["h2h_rate"] = 1 - mirror["h2h_rate"]
    mirror["outcome"] = 0
    mirror["is_mirror"] = True
    return pd.concat([df, mirror], ignore_index=True)


def _age_or_none(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def build_match_features(
    df: pd.DataFrame,
    elo_system: EloSystem,
    feature_builder: FeatureBuilder,
    elo_tracker: Optional[EloHistoryTracker] = None,
) -> pd.DataFrame:
    """Process all matches sequentially. Returns feature dataframe with mirror rows.

    elo_tracker: optional EloHistoryTracker, mutated in place as matches are
        processed (same no-lookahead pattern as elo_system/feature_builder).
        Pass one in when the caller needs it afterward (e.g. live prediction
        reusing the full historical replay); otherwise an internal one is
        used and discarded.
    """
    if elo_tracker is None:
        elo_tracker = EloHistoryTracker()

    # Defensive: the sequential no-lookahead processing below assumes
    # chronological order. Callers (src/data/loader.py) already sort, but
    # don't trust that invariant to hold at every call site.
    df = df.sort_values("match_date").reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        surface = row["surface"]
        match_date = row["match_date"].date()
        w_rank = row.get("winner_rank", np.nan)
        l_rank = row.get("loser_rank", np.nan)
        w_age = _age_or_none(row.get("winner_age"))
        l_age = _age_or_none(row.get("loser_age"))

        # Pre-match Elo
        elo_w = elo_system.get_effective_rating(winner, surface)
        elo_l = elo_system.get_effective_rating(loser, surface)
        elo_prob = elo_system.expected_score(elo_w, elo_l)

        # Pre-match contextual features
        wf = feature_builder.get_features(winner, loser, surface, match_date)
        lf = feature_builder.get_features(loser, winner, surface, match_date)

        # Pre-match decay features (rolling form, age, inactivity rust)
        w_decay = calculate_decay_features(
            elo_w, elo_tracker, winner, surface,
            feature_builder.match_dates(winner), match_date, w_age,
        )
        l_decay = calculate_decay_features(
            elo_l, elo_tracker, loser, surface,
            feature_builder.match_dates(loser), match_date, l_age,
        )

        rank_diff = (
            (l_rank - w_rank)
            if (not np.isnan(w_rank) and not np.isnan(l_rank))
            else 0.0
        )

        records.append(
            {
                "match_date": match_date,
                "year": match_date.year,
                "winner": winner,
                "loser": loser,
                "surface": surface,
                "elo_diff": elo_w - elo_l,
                "elo_prob": elo_prob,
                "rank_diff": float(rank_diff),
                "form_diff": wf["recent_win_rate"] - lf["recent_win_rate"],
                "surface_form_diff": wf["recent_win_rate_surface"] - lf["recent_win_rate_surface"],
                "h2h_rate": wf["h2h_win_rate"],
                "rest_diff": wf["rest_days"] - lf["rest_days"],
                "rolling_elo_diff": w_decay["rolling_elo_diff"] - l_decay["rolling_elo_diff"],
                "age_multiplier_diff": w_decay["age_multiplier"] - l_decay["age_multiplier"],
                "rust_factor_diff": w_decay["rust_factor"] - l_decay["rust_factor"],
                "adjusted_elo_diff": w_decay["adjusted_elo_surface"] - l_decay["adjusted_elo_surface"],
                "outcome": 1,
                "is_mirror": False,
            }
        )

        # Post-match state update (no lookahead)
        elo_system.update(winner, loser, surface, match_date)
        feature_builder.update(winner, loser, surface, match_date)
        elo_tracker.record(winner, surface, elo_w)
        elo_tracker.record(loser, surface, elo_l)

    original = pd.DataFrame(records)

    # Mirror rows: swap perspectives so loser is player1 → outcome=0
    mirror = original.copy()
    for col in _MIRROR_FLIP_COLS:
        mirror[col] = -mirror[col]
    mirror["elo_prob"] = 1.0 - mirror["elo_prob"]
    mirror["h2h_rate"] = 1.0 - mirror["h2h_rate"]
    mirror["outcome"] = 0
    mirror["is_mirror"] = True

    return pd.concat([original, mirror], ignore_index=True)


def walk_forward_backtest(
    df: pd.DataFrame,
    warmup_years: int = 10,
) -> Dict[int, dict]:
    """Walk-forward backtest. Train on all years before test_year; evaluate on test_year."""
    all_years = sorted(df[~df["is_mirror"]]["year"].unique())
    test_years = all_years[warmup_years:]

    results: Dict[int, dict] = {}
    for test_year in test_years:
        train_df = df[df["year"] < test_year]
        # Evaluate only on real matches (not mirrors)
        test_df = df[(df["year"] == test_year) & (~df["is_mirror"])]

        if len(train_df) < 200 or len(test_df) < 20:
            continue

        X_train = train_df[_FEATURE_COLS].fillna(0.0).values
        y_train = train_df["outcome"].values
        X_test = test_df[_FEATURE_COLS].fillna(0.0).values
        y_test = test_df["outcome"].values  # always 1

        # Fit only on X_train — X_test must never influence the scaler,
        # or the test year's own distribution would leak into training.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)

        probs = clf.predict_proba(X_test_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
        elo_probs = test_df["elo_prob"].clip(1e-6, 1 - 1e-6).values

        results[test_year] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "log_loss": float(log_loss(y_test, probs, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_test, probs)),
            "n_matches": int(len(test_df)),
            "elo_only_accuracy": float(accuracy_score(y_test, (elo_probs >= 0.5).astype(int))),
            "elo_only_log_loss": float(log_loss(y_test, elo_probs, labels=[0, 1])),
        }

    return results

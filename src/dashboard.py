"""Visualization dashboard: backtest accuracy by year, calibration curves,
LR feature importance, and value-bet history -- written as PNG files.

Usage:
  python -m src.dashboard              # ATP + WTA + Davis, all 4 charts
  python -m src.dashboard --tour wta   # one tour only
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless -- writes files, never opens a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration_metrics import calibration_error

DASHBOARD_DIR = Path("data/dashboard")


def plot_backtest_accuracy(results_by_tour: dict, out_path) -> Path:
    """Line chart of walk-forward accuracy per year, one line per tour, LR
    vs its own Elo-only baseline (dashed)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (tour, results) in enumerate(results_by_tour.items()):
        years = sorted(results.keys())
        color = colors[i % len(colors)]
        ax.plot(years, [results[y]["accuracy"] for y in years],
                marker="o", color=color, label=f"{tour.upper()} LR")
        if all("elo_only_accuracy" in results[y] for y in years):
            ax.plot(years, [results[y]["elo_only_accuracy"] for y in years],
                    linestyle="--", color=color, alpha=0.5, label=f"{tour.upper()} Elo-only")

    ax.set_xlabel("Year")
    ax.set_ylabel("Accuracy")
    ax.set_title("Walk-forward backtest accuracy by year")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_calibration_curve(probs, y_true, out_path, bins: int = 10) -> Path:
    """Predicted probability (bin mean) vs actual win frequency, plus the
    perfect-calibration diagonal for reference."""
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    if len(probs) != len(y_true):
        raise ValueError(f"probs and y_true must be the same length: {len(probs)} != {len(y_true)}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"p": probs, "y": y_true})
    df["bin"] = pd.cut(df["p"], bins=bins)
    grouped = df.groupby("bin", observed=True).agg(
        mean_p=("p", "mean"), mean_y=("y", "mean"), n=("y", "size"),
    ).dropna()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.scatter(grouped["mean_p"], grouped["mean_y"], s=grouped["n"] / grouped["n"].max() * 300 + 20,
               alpha=0.7, label="Observed (bin size = point size)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual win frequency")
    ax.set_title("Calibration curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def extract_feature_importance(clf, feature_names: list) -> pd.DataFrame:
    """LR coefficients (post-StandardScaler, so directly comparable across
    features regardless of each feature's raw units), sorted by absolute
    magnitude descending."""
    lr = clf.named_steps["logisticregression"]
    coefs = lr.coef_[0]
    if len(coefs) != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but the model has {len(coefs)} coefficients"
        )
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def plot_feature_importance(clf, feature_names: list, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = extract_feature_importance(clf, feature_names)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(df))))
    colors = ["#2166ac" if c >= 0 else "#b2182b" for c in df["coefficient"]]
    ax.barh(df["feature"][::-1], df["coefficient"][::-1], color=colors[::-1])
    ax.set_xlabel("Standardized LR coefficient")
    ax.set_title("Feature importance")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_value_bet_history(value_bets_log_path, out_path) -> Path:
    """Three panels: picks per tour, edge distribution, Kelly-stake
    distribution -- from the qualifying side (a or b, whichever has
    positive edge/kelly) of each logged row."""
    value_bets_log_path = Path(value_bets_log_path)
    if not value_bets_log_path.exists():
        raise FileNotFoundError(f"No se encontro {value_bets_log_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(value_bets_log_path)
    side_a = df["kelly_a"] > 0
    edge = df["edge_a"].where(side_a, df["edge_b"])
    kelly = df["kelly_a"].where(side_a, df["kelly_b"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if "tour" in df.columns:
        df["tour"].value_counts().plot(kind="bar", ax=axes[0], color="#2166ac")
    axes[0].set_title("Picks by tour")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")

    axes[1].hist(edge.dropna() * 100, bins=20, color="#2166ac", edgecolor="white")
    axes[1].set_title("Edge distribution")
    axes[1].set_xlabel("Edge (%)")

    axes[2].hist(kelly.dropna() * 100, bins=20, color="#2166ac", edgecolor="white")
    axes[2].set_title("Kelly stake distribution")
    axes[2].set_xlabel("Kelly fraction (%)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    import argparse

    from src.backtest.walkforward import load_features_with_mirror, walk_forward_backtest
    from src.value_analysis import load_model

    parser = argparse.ArgumentParser(description="Genera el dashboard de visualizacion")
    parser.add_argument("--tour", choices=["atp", "wta", "davis", "both"], default="both")
    args = parser.parse_args()
    tours = ("atp", "wta", "davis") if args.tour == "both" else (args.tour,)

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    results_by_tour = {}
    for tour in tours:
        features_path = Path("data/processed") / f"{tour}_features.csv"
        if not features_path.exists():
            print(f"  Aviso: {features_path} no existe, saltando {tour}. Corre: python -m src.pipeline {tour}")
            continue
        print(f"Backtest {tour.upper()}...")
        df = load_features_with_mirror(features_path)
        results = walk_forward_backtest(df, warmup_years=10, return_predictions=True)
        results_by_tour[tour] = results

        probs = np.concatenate([m["probs_calibration"] for m in results.values()])
        y_true = np.concatenate([m["y_true_calibration"] for m in results.values()])
        err = calibration_error(probs, y_true, lo=0.20, hi=0.80, bins=6)
        print(f"  Calibration error [0.20,0.80]: {err * 100:.2f}pp")
        plot_calibration_curve(probs, y_true, DASHBOARD_DIR / f"calibration_{tour}.png")

        _, _, clf, _, _, _ = load_model(tour)
        from src.features import FEATURE_COLS
        plot_feature_importance(clf, FEATURE_COLS, DASHBOARD_DIR / f"feature_importance_{tour}.png")

    if results_by_tour:
        plot_backtest_accuracy(results_by_tour, DASHBOARD_DIR / "backtest_accuracy.png")

    log_path = Path("data/value_bets_log.csv")
    if log_path.exists():
        plot_value_bet_history(log_path, DASHBOARD_DIR / "value_bet_history.png")
    else:
        print(f"  Aviso: {log_path} no existe, saltando historial de picks.")

    print(f"\nDashboard generado en {DASHBOARD_DIR}/")


if __name__ == "__main__":
    main()

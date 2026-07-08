"""Export figures from sections 7, 8, 9 of the exploratory notebook as PNG files."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.linear_model import LogisticRegression
from src.backtest.walkforward import walk_forward_backtest
from src.data.loader import load_atp_matches
from src.models.elo import EloSystem
from src.features.engineering import FeatureBuilder
from src.backtest.walkforward import build_match_features

OUT = Path("notebooks/figures")
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "elo_diff", "elo_prob", "rank_diff",
    "form_diff", "surface_form_diff", "h2h_rate", "rest_diff",
]

# ── cargar datos ─────────────────────────────────────────────────────────────
print("Cargando features CSV...")
df = pd.read_csv("data/processed/atp_features.csv", parse_dates=["match_date"])

mirror = df.copy()
for col in ("elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff"):
    mirror[col] = -mirror[col]
mirror["elo_prob"]  = 1.0 - mirror["elo_prob"]
mirror["h2h_rate"]  = 1.0 - mirror["h2h_rate"]
mirror["outcome"]   = 0
mirror["is_mirror"] = True
full_df = pd.concat([df, mirror], ignore_index=True)

# ── figura 7: coeficientes LR ────────────────────────────────────────────────
print("Figura 7: coeficientes LR...")
results = walk_forward_backtest(full_df, warmup_years=10)
res_df  = pd.DataFrame(results).T.sort_index()
res_df.index = res_df.index.astype(int)

last_year = int(res_df.index[-1])
train_all = full_df[full_df["year"] < last_year]

clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
clf.fit(train_all[FEATURE_COLS].fillna(0), train_all["outcome"])
coefs = pd.Series(clf.coef_[0], index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#e05a4f" if c < 0 else "#5eb56e" for c in coefs]
coefs.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(f"Coeficientes LR — entrenado hasta {last_year-1}, prediciendo {last_year}")
ax.set_xlabel("Coeficiente")
plt.tight_layout()
path7 = OUT / "fig07_feature_importance.png"
fig.savefig(path7, dpi=150)
plt.close(fig)
print(f"  guardado: {path7}")

# ── figura 8: accuracy por superficie ────────────────────────────────────────
print("Figura 8: accuracy por superficie...")
surfaces_main = ["hard", "clay", "grass"]
surf_results  = {}

for surf in surfaces_main:
    sub = full_df[full_df["surface"] == surf]
    r = walk_forward_backtest(sub, warmup_years=10)
    if r:
        surf_results[surf] = {
            "accuracy":          np.mean([m["accuracy"]          for m in r.values()]),
            "elo_only_accuracy": np.mean([m["elo_only_accuracy"]  for m in r.values()]),
        }

surf_df = pd.DataFrame(surf_results).T
print(surf_df.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(surfaces_main))
w = 0.35
ax.bar(x - w/2, surf_df.loc[surfaces_main, "accuracy"],          width=w, label="LR + features", color="steelblue")
ax.bar(x + w/2, surf_df.loc[surfaces_main, "elo_only_accuracy"], width=w, label="Solo Elo",      color="coral")
ax.set_xticks(x)
ax.set_xticklabels(surfaces_main)
ax.set_ylabel("Accuracy media (2000-2023)")
ax.set_title("Accuracy por superficie")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=1))
ax.legend()
ax.set_ylim(0.60, 0.76)
plt.tight_layout()
path8 = OUT / "fig08_accuracy_by_surface.png"
fig.savefig(path8, dpi=150)
plt.close(fig)
print(f"  guardado: {path8}")

# ── figura 9: top Elo por superficie ─────────────────────────────────────────
print("Figura 9: top Elo por superficie (reconstruyendo EloSystem)...")
raw = load_atp_matches(1990, 2023)
elo = EloSystem()
fb  = FeatureBuilder()
_   = build_match_features(raw, elo, fb)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, surf, col in zip(
    axes,
    ["hard",    "clay",    "grass"],
    ["#4f86c6", "#e05a4f", "#5eb56e"],
):
    top = (
        pd.Series({k[0]: v for k, v in elo.surface_ratings.items() if k[1] == surf})
        .sort_values(ascending=False)
        .head(12)
    )
    ax.barh(top.index[::-1], top.values[::-1], color=col)
    ax.set_title(f"Top Elo - {surf.capitalize()}")
    ax.set_xlabel("Elo rating")
    ax.axvline(1500, color="gray", linestyle="--", linewidth=0.8)

plt.suptitle("Top jugadores por Elo de superficie (cierre 2023)", y=1.02, fontsize=13)
plt.tight_layout()
path9 = OUT / "fig09_top_elo_by_surface.png"
fig.savefig(path9, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  guardado: {path9}")

print("\nExportacion completada.")
print(f"  Directorio: {OUT.resolve()}")

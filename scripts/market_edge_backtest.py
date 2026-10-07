"""Backtest the model's betting edge against historical WTA market odds.

Uso:
  python -m scripts.market_edge_backtest              # WTA
  python -m scripts.market_edge_backtest --tour atp   # needs data/raw/tennis_atp_tduk_odds/

Global calibration (src/calibration_metrics.py) asks "is P(win) right on
average?". This asks the betting question: on the matches where the model
disagrees with the market by enough to bet, who is right? Out-of-sample WTA
predictions (same walk-forward as src.backtest.walkforward, warmup 10 years,
production shrinkage) are joined to Tennis-Data's historical odds (average,
Bet365, max; Pinnacle until 2025) and the production selection rule is
replayed: MIN_EDGE <= p - 1/odds <= MAX_SUSPICIOUS_EDGE, Kelly/4 capped at
KELLY_CAP.

The model's ATP data (Tennismylife) carries no odds, so ATP odds come from
Tennis-Data's ATP files, downloaded once (2026-10-07) from
https://www.tennis-data.co.uk/data.php into data/raw/tennis_atp_tduk_odds/
(gitignored, not refreshed by scripts/download_data.py). Names differ
("Carlos Alcaraz" vs "Alcaraz C.") and Tennismylife dates every match with
the tournament start, so ATP rows join on (last surname token, first
initial) for both players with the Tennis-Data date 3 days before to 20
days after, keeping only 1:1 pairs.

See docs/metrics/2026-10-07-model-vs-market-edge.md.
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.daily_workflow import KELLY_DIVISOR, MIN_EDGE
from src.backtest.walkforward import _FEATURE_COLS, load_features_with_mirror
from src.config import MAX_SUSPICIOUS_EDGE
from src.name_matching import normalize_name
from src.value_analysis import KELLY_CAP, apply_shrinkage

WARMUP_YEARS = 10
ODDS_COLS = ["PSW", "PSL", "B365W", "B365L", "MaxW", "MaxL", "AvgW", "AvgL"]
TOURS = {
    "wta": {"features": "data/processed/wta_features.csv", "odds_dir": "data/raw/tennis_wta_tduk", "odds_glob": "20*w.xls*"},
    "atp": {"features": "data/processed/atp_features.csv", "odds_dir": "data/raw/tennis_atp_tduk_odds", "odds_glob": "20*.xls*"},
}
ATP_DATE_WINDOW_DAYS = (-3, 20)
_TD_INITIALS = re.compile(r"\s((?:[A-Za-z]\.-?\s?)+)$")


def out_of_sample_predictions(features_path: Path) -> pd.DataFrame:
    df = load_features_with_mirror(features_path)
    years = sorted(df[~df["is_mirror"]]["year"].unique())
    frames = []
    for test_year in years[WARMUP_YEARS:]:
        train = df[df["year"] < test_year]
        test = df[(df["year"] == test_year) & ~df["is_mirror"]]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[_FEATURE_COLS].fillna(0.0).values)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_train, train["outcome"].values)
        probs = clf.predict_proba(scaler.transform(test[_FEATURE_COLS].fillna(0.0).values))[:, 1]
        out = test[["match_date", "year", "winner", "loser"] + _FEATURE_COLS].copy()
        out["p"] = [apply_shrinkage(p) for p in probs]
        frames.append(out)
    pred = pd.concat(frames, ignore_index=True)
    pred["match_date"] = pd.to_datetime(pred["match_date"]).dt.normalize()
    return pred


def historical_odds(raw_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.glob(pattern)):
        x = pd.read_excel(path)
        frames.append(x[["Date", "Winner", "Loser"] + [c for c in ODDS_COLS if c in x]])
    odds = pd.concat(frames, ignore_index=True)
    odds["match_date"] = pd.to_datetime(odds["Date"], errors="coerce").dt.normalize()
    odds = odds.rename(columns={"Winner": "winner", "Loser": "loser"}).drop(columns="Date")
    for col in ODDS_COLS:
        odds[col] = pd.to_numeric(odds.get(col), errors="coerce")
    # A (date, winner, loser) key seen twice can't be attributed safely.
    return odds.drop_duplicates(["match_date", "winner", "loser"], keep=False)


def player_key(name: str) -> str:
    """'Bautista Agut R.' and 'Roberto Bautista Agut' -> 'agut|r'."""
    name = str(name).strip()
    m = _TD_INITIALS.search(name)
    if m:  # Tennis-Data: surname then initials
        surname, initials = name[: m.start()], m.group(1)
    else:  # Tennismylife: given name(s) then surname
        parts = name.split(" ", 1)
        initials, surname = parts[0], (parts[1] if len(parts) > 1 else parts[0])
    tokens = normalize_name(surname).replace("-", " ").split()
    return (tokens[-1] if tokens else "") + "|" + normalize_name(initials)[:1]


def join_odds(pred: pd.DataFrame, odds: pd.DataFrame, tour: str) -> pd.DataFrame:
    if tour == "wta":
        return pred.merge(odds, on=["match_date", "winner", "loser"], how="left")
    pred = pred.assign(kw=pred["winner"].map(player_key), kl=pred["loser"].map(player_key), pid=range(len(pred)))
    odds = odds.drop(columns=["winner", "loser"]).assign(
        kw=odds["winner"].map(player_key), kl=odds["loser"].map(player_key), oid=range(len(odds)))
    pairs = pred.merge(odds.rename(columns={"match_date": "odds_date"}), on=["kw", "kl"])
    days = (pairs["odds_date"] - pairs["match_date"]).dt.days
    lo, hi = ATP_DATE_WINDOW_DAYS
    pairs = pairs[(days >= lo) & (days <= hi)]
    pairs = pairs[pairs.groupby("pid")["oid"].transform("size").eq(1)
                  & pairs.groupby("oid")["pid"].transform("size").eq(1)]
    return pred.merge(pairs[["pid"] + [c for c in ODDS_COLS if c in pairs]], on="pid", how="left")


def _devig(own: pd.Series, other: pd.Series) -> pd.Series:
    return (1 / own) / (1 / own + 1 / other)


def bet_sides(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per side of each match: winner side (y=1) and loser side (y=0)."""
    sides = []
    for own, other, p, y in (("W", "L", matches["p"], 1), ("L", "W", 1 - matches["p"], 0)):
        sides.append(pd.DataFrame({
            "year": matches["year"], "p": p, "y": y,
            "q": _devig(matches[f"Avg{own}"], matches[f"Avg{other}"]),
            "q_ps": _devig(matches[f"PS{own}"], matches[f"PS{other}"]),
            "avg": matches[f"Avg{own}"], "b365": matches[f"B365{own}"], "max": matches[f"Max{own}"],
        }))
    return pd.concat(sides, ignore_index=True)


def select_bets(sides: pd.DataFrame, price: str) -> pd.DataFrame:
    odds = sides[price]
    edge = sides["p"] - 1 / odds
    bets = sides[(odds > 1) & (edge >= MIN_EDGE) & (edge <= MAX_SUSPICIOUS_EDGE)].copy()
    bets["odds"] = bets[price]
    bets["edge"] = edge[bets.index]
    bets["ev"] = bets["p"] * (bets["odds"] - 1) - (1 - bets["p"])
    bets["stake"] = np.minimum(bets["edge"] / (bets["odds"] - 1), KELLY_CAP) / KELLY_DIVISOR
    bets["ret"] = np.where(bets["y"] == 1, bets["odds"] - 1, -1.0)
    return bets


ANCHORED_MIN_TRAIN_YEARS = 3


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def anchored_predictions(matches: pd.DataFrame) -> pd.DataFrame:
    """Market-anchored models, walk-forward over the years that have odds
    (train on all earlier odds years, at least ANCHORED_MIN_TRAIN_YEARS):

      M1: logit(q)                      -- the market alone, recalibrated
      M2: logit(q) + logit(p_model)     -- market + current model's OOS p
      M3: logit(q) + the 15 features    -- market + raw features (L2, C=0.1)

    No intercept, and every input flips sign for the other side (elo_prob and
    h2h_rate are centred at 0.5 so their loser view is an exact negation, as
    in load_features_with_mirror), so P(A) + P(B) = 1 by construction.
    """
    feats = matches[_FEATURE_COLS].fillna({"elo_prob": 0.5, "h2h_rate": 0.5}).fillna(0.0).copy()
    feats["elo_prob"] -= 0.5
    feats["h2h_rate"] -= 0.5
    F = feats.to_numpy()
    sides = bet_sides(matches)  # winner rows then loser rows, same order as F / -F
    X_feat = np.vstack([F, -F])
    lq, lp = _logit(sides["q"]), _logit(sides["p"])
    years = sorted(sides["year"].unique())
    out = []
    for test_year in years[ANCHORED_MIN_TRAIN_YEARS:]:
        tr = (sides["year"] < test_year).to_numpy()
        te = (sides["year"] == test_year).to_numpy()
        res = sides[te].copy()
        for name, X, C in (("M1", lq[:, None], 1e6), ("M2", np.column_stack([lq, lp]), 1e6)):
            clf = LogisticRegression(fit_intercept=False, C=C, max_iter=1000).fit(X[tr], sides["y"][tr])
            res[name] = clf.predict_proba(X[te])[:, 1]
        scaler = StandardScaler(with_mean=False).fit(X_feat[tr])  # scale only: keeps the sign symmetry
        X3 = np.column_stack([lq, scaler.transform(X_feat)])
        clf = LogisticRegression(fit_intercept=False, C=0.1, max_iter=2000).fit(X3[tr], sides["y"][tr])
        res["M3"] = clf.predict_proba(X3[te])[:, 1]
        out.append(res)
    return pd.concat(out, ignore_index=True)


def _bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _summary(bets: pd.DataFrame, by: str) -> pd.DataFrame:
    return bets.groupby(by, observed=True).agg(
        N=("y", "size"), model_p=("p", "mean"), market_q=("q", "mean"),
        actual=("y", "mean"), EV=("ev", "mean"), ROI=("ret", "mean"),
    ).round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest del edge del modelo contra cuotas historicas")
    parser.add_argument("--tour", choices=sorted(TOURS), default="wta")
    parser.add_argument("--anchored", action="store_true",
                        help="Tambien evalua modelos anclados al mercado (M1/M2/M3)")
    args = parser.parse_args()
    tour = args.tour
    cfg = TOURS[tour]
    warnings.filterwarnings("ignore")  # openpyxl "unknown extension" noise
    pred = out_of_sample_predictions(_ROOT / cfg["features"])
    matches = join_odds(pred, historical_odds(_ROOT / cfg["odds_dir"], cfg["odds_glob"]), tour)
    matches = matches[matches["AvgW"].gt(1) & matches["AvgL"].gt(1)]
    sides = bet_sides(matches)
    print(f"Out-of-sample {tour.upper()} {int(matches['year'].min())}-{int(matches['year'].max())}: "
          f"{len(pred)} predictions, {len(matches)} with market odds\n")

    print("A. Accuracy over every side of every match")
    print(f"  model                  log-loss {log_loss(sides['y'], sides['p']):.4f}  "
          f"brier {brier_score_loss(sides['y'], sides['p']):.4f}")
    print(f"  market avg (de-vigged) log-loss {log_loss(sides['y'], sides['q']):.4f}  "
          f"brier {brier_score_loss(sides['y'], sides['q']):.4f}")
    ps = sides.dropna(subset=["q_ps"])
    print(f"  Pinnacle subset: model {log_loss(ps['y'], ps['p']):.4f} vs Pinnacle {log_loss(ps['y'], ps['q_ps']):.4f}")
    blends = {a: log_loss(sides["y"], a * sides["p"] + (1 - a) * sides["q"]) for a in np.linspace(0, 1, 21)}
    best = min(blends, key=blends.get)
    print(f"  best blend a*model + (1-a)*market: a={best:.2f} (log-loss {blends[best]:.4f})\n")

    print(f"B. Production rule ({MIN_EDGE:.0%} <= edge <= {MAX_SUSPICIOUS_EDGE:.0%}, "
          f"Kelly/{KELLY_DIVISOR} cap {KELLY_CAP:.0%})")
    for price, label in (("avg", "market average odds"), ("b365", "Bet365"), ("max", "best odds (optimistic)")):
        bets = select_bets(sides, price)
        lo, hi = _bootstrap_ci(bets["ret"].to_numpy())
        kelly_roi = (bets["stake"] * bets["ret"]).sum() / bets["stake"].sum()
        print(f"  [{label}] N={len(bets)}  model p {bets['p'].mean():.3f} | market q {bets['q'].mean():.3f} | "
              f"actual {bets['y'].mean():.3f}\n"
              f"    EV theoretical {bets['ev'].mean():+.1%} | ROI flat {bets['ret'].mean():+.1%} "
              f"(95% CI {lo:+.1%}..{hi:+.1%}) | ROI Kelly-weighted {kelly_roi:+.1%}")
    base = sides.assign(ret=np.where(sides["y"] == 1, sides["avg"] - 1, -1.0))
    print(f"  baseline, bet every side at avg odds: ROI {base['ret'].mean():+.1%}\n")

    bets = select_bets(sides, "avg")
    bets["odds_band"] = pd.cut(bets["odds"], [1, 1.5, 2, 3, 5, 100])
    bets["edge_band"] = pd.cut(bets["edge"], [MIN_EDGE, 0.05, 0.07, MAX_SUSPICIOUS_EDGE])
    print("C. Selected bets (avg odds) by odds band\n" + _summary(bets, "odds_band").to_string() + "\n")
    print("D. By edge band\n" + _summary(bets, "edge_band").to_string() + "\n")
    print("E. By year\n" + _summary(bets, "year")[["N", "EV", "ROI"]].T.to_string())

    if not args.anchored:
        return
    anchored = anchored_predictions(matches)
    print(f"\nF. Market-anchored models, test years {int(anchored['year'].min())}-{int(anchored['year'].max())}")
    print("  log-loss: " + " | ".join(
        f"{name} {log_loss(anchored['y'], anchored[col]):.4f}"
        for name, col in (("market", "q"), ("model", "p"), ("M1", "M1"), ("M2", "M2"), ("M3", "M3"))))
    for price in ("avg", "max"):
        print(f"  [{price} odds] production rule")
        for col in ("p", "M1", "M2", "M3"):
            bets = select_bets(anchored.assign(p=anchored[col]), price)
            if bets.empty:
                print(f"    {col:3s} N=0")
                continue
            lo, hi = _bootstrap_ci(bets["ret"].to_numpy())
            print(f"    {col:3s} N={len(bets):6d} ROI flat {bets['ret'].mean():+.1%} (95% CI {lo:+.1%}..{hi:+.1%})")
        every = anchored[anchored[price] > 1]
        print(f"    bet every side ROI {np.where(every['y'] == 1, every[price] - 1, -1.0).mean():+.1%}")


if __name__ == "__main__":
    main()

"""Do ATP serve/return statistics add information the market lacks?

Uso:
  python -m scripts.serve_stats_edge
  python -m scripts.serve_stats_edge --half-life 365 --prior-points 400

Research only: nothing here feeds production (FEATURE_COLS is unchanged).
Requires data/processed/atp_features.csv (python -m src.pipeline atp) and the
Tennis-Data ATP odds in data/raw/tennis_atp_tduk_odds/ (see
scripts/market_edge_backtest.py).

Tennismylife rows carry per-match serve stats (svpt, 1stWon, 2ndWon, aces,
double faults, break points). One sequential pass over load_atp_matches()
-- the same leak-free order the pipeline uses, so its rows line up 1:1 with
atp_features.csv -- keeps exponentially decayed, opponent-adjusted serve and
return skills per player (overall and per surface). Every feature is read
before the match updates the state.

Opponent adjustment (online Barnett-Clarke): the serve points A was
*expected* to win against B are avg_spw - (B's return skill - avg_rpw); A's
serve skill is avg_spw plus A's decayed excess over that expectation, shrunk
toward the average with `prior_points` pseudo-points. Same for return.

Features (winner's view; all flip sign for the mirror row):
  serve_adj_diff, return_adj_diff      -- skill differences
  serve_model_logit                    -- logit of the match-win probability
                                          from an iid point model (best-of 3/5)
  surface_serve_model_logit            -- same with surface-specific skills
  ace_rate_diff, df_rate_diff, bp_save_diff

Two questions, both walk-forward out of sample:
  A. Without the market: do the 15 production features + serve features beat
     the 15 alone?
  B. With the market (the one that matters for betting): does logit(q) +
     serve features beat logit(q) alone (M1), consistently by year, and does
     the production rule make money with it?
"""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.market_edge_backtest import (
    ANCHORED_MIN_TRAIN_YEARS, TOURS, WARMUP_YEARS, _bootstrap_ci, _logit, bet_sides,
    historical_odds, join_odds, select_bets,
)
from src.backtest.walkforward import _FEATURE_COLS
from src.data.loader import load_atp_matches
from src.value_analysis import apply_shrinkage

SERVE_COLS = [
    "serve_adj_diff", "return_adj_diff", "serve_model_logit", "surface_serve_model_logit",
    "ace_rate_diff", "df_rate_diff", "bp_save_diff",
]
GLOBAL_HALF_LIFE_DAYS = 730
INITIAL_AVG_SPW = 0.62


# --- iid point model -------------------------------------------------------

def _game(p: float) -> float:
    q = 1 - p
    deuce = p * p / (1 - 2 * p * q)
    return p**4 * (1 + 4 * q + 10 * q * q) + 20 * p**3 * q**3 * deuce


def _tiebreak(pa: float, pb: float) -> float:
    """A serves first point, then serve alternates every two points."""
    @lru_cache(maxsize=None)
    def win(a: int, b: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        n = a + b
        if a >= 6 and a == b:  # 6-6, 7-7, ...: one point on each serve until someone leads by 2
            p1 = pa if ((n + 1) // 2) % 2 == 0 else 1 - pb
            p2 = pa if ((n + 2) // 2) % 2 == 0 else 1 - pb
            return p1 * p2 / (p1 * p2 + (1 - p1) * (1 - p2))
        p = pa if ((n + 1) // 2) % 2 == 0 else 1 - pb
        return p * win(a + 1, b) + (1 - p) * win(a, b + 1)
    return win(0, 0)


def _set(pa: float, pb: float) -> float:
    ga, gb = _game(pa), 1 - _game(pb)  # P(A wins a game) on A's / B's serve
    tb = _tiebreak(pa, pb)

    @lru_cache(maxsize=None)
    def win(a: int, b: int) -> float:
        if (a >= 6 and a - b >= 2) or a == 7:
            return 1.0
        if (b >= 6 and b - a >= 2) or b == 7:
            return 0.0
        if a == 6 and b == 6:
            return tb
        g = ga if (a + b) % 2 == 0 else gb
        return g * win(a + 1, b) + (1 - g) * win(a, b + 1)
    return win(0, 0)


def match_win_prob(pa: float, pb: float, best_of: int) -> float:
    """P(A wins) given P(A wins a point on serve)=pa, P(B ...)=pb."""
    pa, pb = min(max(pa, 0.3), 0.95), min(max(pb, 0.3), 0.95)
    s = _set(pa, pb)
    need = 3 if best_of == 5 else 2
    return sum(math.comb(need - 1 + k, k) * s**need * (1 - s) ** k for k in range(need))


# --- sequential serve-skill state ------------------------------------------

class _Decayed:
    """Exponentially decayed sums keyed by name."""
    __slots__ = ("v", "last")

    def __init__(self):
        self.v: dict[str, float] = defaultdict(float)
        self.last = None

    def at(self, day, half_life: float) -> "_Decayed":
        if self.last is not None and day > self.last:
            f = 0.5 ** ((day - self.last).days / half_life)
            for k in self.v:
                self.v[k] *= f
        self.last = day if self.last is None or day > self.last else self.last
        return self


def build_serve_features(df: pd.DataFrame, half_life: float, prior_points: float) -> pd.DataFrame:
    players: dict = defaultdict(_Decayed)
    surfaces: dict = defaultdict(_Decayed)
    glob = _Decayed()
    glob.v["won"], glob.v["pts"] = INITIAL_AVG_SPW * 1000, 1000.0
    K = prior_points

    def skills(st: _Decayed, avg_spw: float) -> tuple[float, float]:
        v = st.v
        s = avg_spw + (v["sv_won"] - v["sv_exp"]) / (v["sv_pts"] + K)
        r = (1 - avg_spw) + (v["rt_won"] - v["rt_exp"]) / (v["rt_pts"] + K)
        return s, r

    def rates(st: _Decayed) -> tuple[float, float, float]:
        v = st.v
        ace = (v["aces"] + 0.07 * K) / (v["sv_pts"] + K)
        dfr = (v["dfs"] + 0.035 * K) / (v["sv_pts"] + K)
        bps = (v["bp_saved"] + 0.6 * 20) / (v["bp_faced"] + 20)
        return ace, dfr, bps

    cols = ["match_date", "surface", "winner_name", "loser_name", "best_of",
            "w_svpt", "w_1stWon", "w_2ndWon", "w_ace", "w_df", "w_bpSaved", "w_bpFaced",
            "l_svpt", "l_1stWon", "l_2ndWon", "l_ace", "l_df", "l_bpSaved", "l_bpFaced"]
    rows = []
    for m in df[cols].itertuples(index=False):
        day = m.match_date.date()
        glob.at(day, GLOBAL_HALF_LIFE_DAYS)
        avg = glob.v["won"] / glob.v["pts"]
        W = players[m.winner_name].at(day, half_life)
        L = players[m.loser_name].at(day, half_life)
        Ws = surfaces[(m.winner_name, m.surface)].at(day, half_life)
        Ls = surfaces[(m.loser_name, m.surface)].at(day, half_life)
        sw, rw = skills(W, avg)
        sl, rl = skills(L, avg)
        ssw, srw = skills(Ws, avg)
        ssl, srl = skills(Ls, avg)
        bo = 5 if m.best_of == 5 else 3
        p_overall = match_win_prob(sw - (rl - (1 - avg)), sl - (rw - (1 - avg)), bo)
        p_surface = match_win_prob(ssw - (srl - (1 - avg)), ssl - (srw - (1 - avg)), bo)
        aw, dw, bw = rates(W)
        al, dl, bl = rates(L)
        rows.append((sw - sl, rw - rl, float(_logit(p_overall)), float(_logit(p_surface)),
                     aw - al, dw - dl, bw - bl))

        if not (m.w_svpt > 0 and m.l_svpt > 0):
            continue  # walkover / no stats: nothing to learn
        w_won = m.w_1stWon + m.w_2ndWon
        l_won = m.l_1stWon + m.l_2ndWon
        exp_w = avg - (rl - (1 - avg))  # winner's expected serve-point rate vs this returner
        exp_l = avg - (rw - (1 - avg))
        for P, S, sv_pts, sv_won, sv_exp_rate, o_pts, o_won, o_exp_rate, ace, dfs, bps, bpf in (
            (W, Ws, m.w_svpt, w_won, exp_w, m.l_svpt, l_won, exp_l, m.w_ace, m.w_df, m.w_bpSaved, m.w_bpFaced),
            (L, Ls, m.l_svpt, l_won, exp_l, m.w_svpt, w_won, exp_w, m.l_ace, m.l_df, m.l_bpSaved, m.l_bpFaced),
        ):
            upd = {
                "sv_pts": sv_pts, "sv_won": sv_won, "sv_exp": sv_pts * sv_exp_rate,
                "rt_pts": o_pts, "rt_won": o_pts - o_won, "rt_exp": o_pts * (1 - o_exp_rate),
            }
            for st in (P, S):
                for k, x in upd.items():
                    st.v[k] += x
            for k, x in (("aces", ace), ("dfs", dfs), ("bp_saved", bps), ("bp_faced", bpf)):
                if pd.notna(x):
                    P.v[k] += x
        glob.v["won"] += w_won + l_won
        glob.v["pts"] += m.w_svpt + m.l_svpt
    return pd.DataFrame(rows, columns=SERVE_COLS, index=df.index)


# --- evaluation ------------------------------------------------------------

def _with_mirror(df: pd.DataFrame, flip: list[str]) -> pd.DataFrame:
    orig = df.assign(outcome=1, is_mirror=False)
    mirror = orig.copy()
    for c in flip:
        mirror[c] = -mirror[c]
    mirror["outcome"], mirror["is_mirror"] = 0, True
    return pd.concat([orig, mirror], ignore_index=True)


def walk_forward(feats: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    """OOS p for each feature set; 'centred' copies of elo_prob/h2h_rate make
    every input sign-symmetric so the mirror is an exact negation."""
    feats = feats.copy()
    feats["elo_prob"] = feats["elo_prob"].fillna(0.5) - 0.5
    feats["h2h_rate"] = feats["h2h_rate"].fillna(0.5) - 0.5
    all_cols = sorted({c for cols in feature_sets.values() for c in cols})
    feats[all_cols] = feats[all_cols].fillna(0.0)
    full = _with_mirror(feats, all_cols)
    years = sorted(feats["year"].unique())
    out = []
    for test_year in years[WARMUP_YEARS:]:
        train = full[full["year"] < test_year]
        test = feats[feats["year"] == test_year][["match_date", "year", "winner", "loser"]].copy()
        X_test_all = feats[feats["year"] == test_year]
        for name, cols in feature_sets.items():
            scaler = StandardScaler().fit(train[cols].values)
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(train[cols].values), train["outcome"])
            probs = clf.predict_proba(scaler.transform(X_test_all[cols].values))[:, 1]
            test[name] = [apply_shrinkage(p) for p in probs]
        out.append(test)
    pred = pd.concat(out, ignore_index=True)
    pred["match_date"] = pd.to_datetime(pred["match_date"]).dt.normalize()
    return pred


def anchored(matches: pd.DataFrame, models: dict[str, list[str] | None]) -> pd.DataFrame:
    """Walk-forward logit(q) + extra columns, no intercept (see
    market_edge_backtest.anchored_predictions). None = market only (M1)."""
    sides = bet_sides(matches)
    lq = _logit(sides["q"])
    years = sorted(sides["year"].unique())
    res_all = []
    for test_year in years[ANCHORED_MIN_TRAIN_YEARS:]:
        tr = (sides["year"] < test_year).to_numpy()
        te = (sides["year"] == test_year).to_numpy()
        res = sides[te].copy()
        for name, cols in models.items():
            if cols is None:
                X, C = lq[:, None], 1e6
            else:
                F = matches[cols].fillna(0.0).to_numpy()
                Xf = np.vstack([F, -F])
                Xf = StandardScaler(with_mean=False).fit(Xf[tr]).transform(Xf)
                X, C = np.column_stack([lq, Xf]), 0.1
            clf = LogisticRegression(fit_intercept=False, C=C, max_iter=2000).fit(X[tr], sides["y"][tr])
            res[name] = clf.predict_proba(X[te])[:, 1]
        res_all.append(res)
    return pd.concat(res_all, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valor de las estadisticas de saque/resto ATP frente al mercado")
    parser.add_argument("--half-life", type=float, default=270.0, help="vida media del decaimiento (dias)")
    parser.add_argument("--prior-points", type=float, default=300.0, help="pseudo-puntos de shrinkage")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    cfg = TOURS["atp"]

    raw = load_atp_matches(1990)
    feats = pd.read_csv(_ROOT / cfg["features"])
    if len(raw) != len(feats) or not (raw["winner_name"].values == feats["winner"].values).all():
        sys.exit("atp_features.csv no esta alineado con load_atp_matches(1990): regenera con python -m src.pipeline atp")
    serve = build_serve_features(raw, args.half_life, args.prior_points)
    feats = pd.concat([feats, serve.reset_index(drop=True)], axis=1)

    sets = {
        "p": list(_FEATURE_COLS),
        "p_serve": list(_FEATURE_COLS) + SERVE_COLS,
        "p_serve_only": SERVE_COLS,
    }
    pred = walk_forward(feats, sets)
    print(f"Saque/resto ATP (vida media {args.half_life:.0f} d, prior {args.prior_points:.0f} pts)\n")
    print(f"A. Sin mercado, walk-forward {int(pred['year'].min())}-{int(pred['year'].max())}, N={len(pred)}")
    y1 = np.ones(len(pred))
    for name in sets:
        p = pred[name].to_numpy()
        ll = log_loss(np.r_[y1, 1 - y1], np.r_[p, 1 - p])
        print(f"  {name:13s} log-loss {ll:.4f}  accuracy {(p > 0.5).mean():.4f}")

    keep = ["match_date", "winner", "loser"] + SERVE_COLS
    pred = pred.merge(feats.assign(match_date=pd.to_datetime(feats["match_date"]).dt.normalize())[keep]
                      .drop_duplicates(["match_date", "winner", "loser"], keep=False),
                      on=["match_date", "winner", "loser"], how="inner")
    matches = join_odds(pred, historical_odds(_ROOT / cfg["odds_dir"], cfg["odds_glob"]), "atp")
    matches = matches[matches["AvgW"].gt(1) & matches["AvgL"].gt(1)].reset_index(drop=True)
    sides = bet_sides(matches)
    print(f"\n   con cuotas: {len(matches)} partidos")
    for name in sets:
        ps = np.r_[matches[name], 1 - matches[name]]
        print(f"  {name:13s} log-loss {log_loss(sides['y'], ps):.4f}")
    print(f"  mercado       log-loss {log_loss(sides['y'], sides['q']):.4f}")

    models = {
        "M1": None,
        "M_logit": ["serve_model_logit", "surface_serve_model_logit"],
        "M_serve": SERVE_COLS,
    }
    an = anchored(matches, models)
    print(f"\nB. Anclado al mercado, test {int(an['year'].min())}-{int(an['year'].max())}, N lados={len(an)}")
    base = log_loss(an["y"], an["M1"])
    print(f"  mercado {log_loss(an['y'], an['q']):.4f} | M1 {base:.4f}")
    for name in models:
        if name == "M1":
            continue
        ll = log_loss(an["y"], an[name])
        by_year = an.groupby("year").apply(
            lambda g: log_loss(g["y"], g["M1"], labels=[0, 1]) - log_loss(g["y"], g[name], labels=[0, 1]))
        print(f"  {name:8s} {ll:.4f}  mejora sobre M1 {1000 * (base - ll):+.2f} mll; "
              f"mejora en {int((by_year > 0).sum())}/{len(by_year)} anos")
    for price in ("avg", "max"):
        print(f"  [{price}] regla de produccion")
        for name in models:
            bets = select_bets(an.assign(p=an[name]), price)
            if bets.empty:
                print(f"    {name:8s} N=0")
                continue
            lo, hi = _bootstrap_ci(bets["ret"].to_numpy())
            print(f"    {name:8s} N={len(bets):6d} ROI plano {bets['ret'].mean():+.1%} (IC 95% {lo:+.1%}..{hi:+.1%})")


if __name__ == "__main__":
    main()

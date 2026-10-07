"""Closing-line value (CLV) of the model's predictions, from the odds
snapshots recorded by scripts/snapshot_odds.py.

Uso:
  python -m scripts.clv_report
  python -m scripts.clv_report --max-close-gap-hours 8

ROI needs thousands of settled bets to separate skill from luck. CLV doesn't
wait for results: the closing line is the market's best estimate, so
taking prices that beat it is the standard sign of a real edge, and it
shows up in hundreds of matches.

Each prediction in data/prediction_audit_log.csv (the daily scan logs every
evaluated match, bet or not) is joined to its snapshot event by tour, both
players' (last surname token, initial) key in either order -- the same key
as scripts/market_edge_backtest.py -- and a commence date within a day of
match_date. For every event:

  q_scan   fair (de-vigged) probability at the last snapshot run at or
           before the scan timestamp -- what the market said when the model
           made its call
  q_close  fair probability from each bookmaker's last pre-match price,
           averaged over bookmakers; only kept when a snapshot run happened
           within --max-close-gap-hours of the start

Reported:
  A. Does the model predict line moves? Slope of (q_close - q_scan) on
     (p_model - q_scan) over every side of every match. A slope > 0 means
     the line moves toward the model after the scan -- information the
     market hadn't priced yet, the only way betting early can pay.
  B. CLV of the production rule's picks (3% <= p - 1/odds <= 10% at the
     scan's logged best odds): mean of odds * q_close - 1, the expected
     return if the closing line is the truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.daily_workflow import MIN_EDGE
from scripts.market_edge_backtest import _bootstrap_ci, player_key
from src.config import MAX_SUSPICIOUS_EDGE
from src.data.timezone_utils import LIMA_TZ

SNAPSHOT_DIR = _ROOT / "data" / "odds_snapshots"
AUDIT_LOG = _ROOT / "data" / "prediction_audit_log.csv"
DEFAULT_MAX_CLOSE_GAP_HOURS = 8.0


def load_snapshots(snapshot_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    files = sorted(p for p in snapshot_dir.glob("*.csv") if p.name != "runs.csv")
    runs_path = snapshot_dir / "runs.csv"
    if not files or not runs_path.exists():
        return pd.DataFrame(), pd.Series(dtype="datetime64[ns, UTC]")
    snaps = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    snaps["snapshot_utc"] = pd.to_datetime(snaps["snapshot_utc"], utc=True)
    snaps["commence_time"] = pd.to_datetime(snaps["commence_time"], utc=True)
    runs = pd.to_datetime(pd.read_csv(runs_path)["snapshot_utc"], utc=True).sort_values().reset_index(drop=True)
    return snaps, runs


def fair_home_prob(rows: pd.DataFrame) -> float:
    """Mean over bookmakers of the de-vigged home probability."""
    ih, ia = 1 / rows["price_home"], 1 / rows["price_away"]
    return float((ih / (ih + ia)).mean())


def fair_at(event_rows: pd.DataFrame, when: pd.Timestamp) -> Optional[float]:
    """Fair home probability as of `when`: each bookmaker's latest stored
    price at or before it (rows are change-only, so the latest row is the
    price still current)."""
    upto = event_rows[event_rows["snapshot_utc"] <= when]
    if upto.empty:
        return None
    return fair_home_prob(upto.sort_values("snapshot_utc").groupby("bookmaker").tail(1))


def event_lines(snaps: pd.DataFrame, runs: pd.Series, max_close_gap_hours: float) -> pd.DataFrame:
    """One row per event: keys, commence time and the closing fair line
    (None when no run came close enough to the start)."""
    out = []
    for eid, ev in snaps.groupby("event_id"):
        first = ev.iloc[0]
        commence = ev["commence_time"].max()  # a rescheduled match keeps its latest start
        before = runs[runs < commence]
        gap_h = (commence - before.iloc[-1]).total_seconds() / 3600 if len(before) else np.inf
        out.append({
            "event_id": eid, "tour": first["sport_key"].split("_")[1],
            "home": first["home"], "away": first["away"],
            "kh": player_key(first["home"]), "ka": player_key(first["away"]),
            "commence": commence, "close_gap_h": gap_h,
            "q_close_home": fair_at(ev, commence) if gap_h <= max_close_gap_hours else None,
        })
    return pd.DataFrame(out)


def join_audit(audit: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Attach event_id + orientation (a_is_home) to audit rows; rows that
    match zero or several events are dropped."""
    a = audit.copy()
    a["row"] = range(len(a))
    a["ka"], a["kb"] = a["player_a"].map(player_key), a["player_b"].map(player_key)
    a["match_date"] = pd.to_datetime(a["match_date"]).dt.date
    ev = events.assign(lima_date=events["commence"].dt.tz_convert(LIMA_TZ).dt.date)
    pairs = []
    for a_is_home, (left, right) in ((True, ("kh", "ka")), (False, ("ka", "kh"))):
        m = a.merge(ev.rename(columns={left: "ka", right: "kb", "tour": "ev_tour"}), on=["ka", "kb"])
        m = m[(m["ev_tour"] == m["tour"])
              & ((pd.to_datetime(m["lima_date"]) - pd.to_datetime(m["match_date"])).dt.days.abs() <= 1)]
        pairs.append(m.assign(a_is_home=a_is_home))
    m = pd.concat(pairs, ignore_index=True)
    m = m[m.groupby("row")["event_id"].transform("nunique").eq(1)]
    return m.drop_duplicates("row")


def sides(joined: pd.DataFrame, snaps: pd.DataFrame, runs: pd.Series) -> pd.DataFrame:
    """Two rows per audit prediction (player a, player b) with p, the logged
    best odds, q_scan and q_close from that side's view."""
    by_event = dict(tuple(snaps.groupby("event_id")))
    out = []
    for r in joined.itertuples(index=False):
        scan = pd.Timestamp(r.timestamp).tz_localize(LIMA_TZ).tz_convert("UTC")
        before = runs[runs <= scan]
        q_scan_home = fair_at(by_event[r.event_id], before.iloc[-1]) if len(before) else None
        if q_scan_home is None or r.q_close_home is None or pd.isna(r.q_close_home):
            continue
        qa_scan = q_scan_home if r.a_is_home else 1 - q_scan_home
        qa_close = r.q_close_home if r.a_is_home else 1 - r.q_close_home
        for p, odds, q_scan, q_close in ((r.p_a_cal, r.odds_a, qa_scan, qa_close),
                                          (r.p_b_cal, r.odds_b, 1 - qa_scan, 1 - qa_close)):
            out.append({"event_id": r.event_id, "tour": r.tour, "p": p, "odds": odds,
                        "q_scan": q_scan, "q_close": q_close})
    return pd.DataFrame(out)


def line_move_slope(s: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """OLS slope (no intercept: both sides are mirrored) of the line move on
    the model's disagreement, with an event-clustered bootstrap CI."""
    x = (s["p"] - s["q_scan"]).to_numpy()
    y = (s["q_close"] - s["q_scan"]).to_numpy()
    slope = float(x @ y / (x @ x)) if x @ x > 0 else float("nan")
    groups = s.groupby("event_id").indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        xx, yy = x[idx], y[idx]
        if xx @ xx > 0:
            boots.append(xx @ yy / (xx @ xx))
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return slope, float(lo), float(hi)


def rule_picks(s: pd.DataFrame) -> pd.DataFrame:
    edge = s["p"] - 1 / s["odds"]
    picks = s[(s["odds"] > 1) & (edge >= MIN_EDGE) & (edge <= MAX_SUSPICIOUS_EDGE)].copy()
    picks["clv"] = picks["odds"] * picks["q_close"] - 1
    return picks


def summary(snapshot_dir: Path = SNAPSHOT_DIR, audit_log: Path = AUDIT_LOG,
            max_close_gap_hours: float = DEFAULT_MAX_CLOSE_GAP_HOURS, n_boot: int = 2000) -> dict:
    """Everything the report prints, as a dict (also used by the daily email).
    Keys are always present; values are None when there isn't data yet."""
    out = {"runs": 0, "last_run": None, "runs_24h": 0, "credits_remaining": None, "events": 0,
           "events_with_close": 0, "audit_rows": 0, "joined": 0, "matches": 0,
           "slope": None, "slope_ci": None, "move": None, "picks": 0, "clv": None, "clv_ci": None,
           "clv_by_tour": {}, "pick_p": None, "pick_q_scan": None, "pick_q_close": None}
    snaps, runs = load_snapshots(snapshot_dir)
    if snaps.empty:
        return out
    run_log = pd.read_csv(snapshot_dir / "runs.csv")
    last = runs.iloc[-1]
    out.update(runs=len(runs), last_run=last, runs_24h=int((runs > last - pd.Timedelta(hours=24)).sum()))
    credits = pd.to_numeric(run_log.get("credits_remaining", pd.Series(dtype=float)), errors="coerce").dropna()
    out["credits_remaining"] = int(credits.iloc[-1]) if len(credits) else None

    events = event_lines(snaps, runs, max_close_gap_hours)
    out.update(events=len(events), events_with_close=int(events["q_close_home"].notna().sum()))
    if not audit_log.exists():
        return out
    audit = pd.read_csv(audit_log)
    audit = audit[pd.to_datetime(audit["timestamp"]) >= runs.iloc[0].tz_convert(LIMA_TZ).tz_localize(None)]
    joined = join_audit(audit, events) if len(audit) else audit
    s = sides(joined, snaps, runs) if len(joined) else pd.DataFrame()
    out.update(audit_rows=len(audit), joined=len(joined), matches=int(s["event_id"].nunique()) if len(s) else 0)
    if s.empty:
        return out
    slope, lo, hi = line_move_slope(s, n_boot=n_boot)
    out.update(slope=slope, slope_ci=(lo, hi), move=float(np.abs(s["q_close"] - s["q_scan"]).mean()))
    picks = rule_picks(s)
    out["picks"] = len(picks)
    if len(picks):
        out.update(clv=float(picks["clv"].mean()), clv_ci=_bootstrap_ci(picks["clv"].to_numpy()),
                   clv_by_tour={t: (len(g), float(g["clv"].mean())) for t, g in picks.groupby("tour")},
                   pick_p=float(picks["p"].mean()), pick_q_scan=float(picks["q_scan"].mean()),
                   pick_q_close=float(picks["q_close"].mean()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CLV del modelo frente a la cuota de cierre")
    parser.add_argument("--max-close-gap-hours", type=float, default=DEFAULT_MAX_CLOSE_GAP_HOURS,
                        help="maximo entre el ultimo snapshot y el inicio para aceptar la cuota de cierre")
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--audit-log", type=Path, default=AUDIT_LOG)
    args = parser.parse_args()

    r = summary(args.snapshot_dir, args.audit_log, args.max_close_gap_hours)
    if not r["runs"]:
        sys.exit("Sin snapshots todavia (python -m scripts.snapshot_odds).")
    print(f"Snapshots: {r['runs']} corridas, {r['events']} partidos "
          f"({r['events_with_close']} con cierre a <= {args.max_close_gap_hours:g} h)")
    print(f"Auditoria desde el primer snapshot: {r['audit_rows']} predicciones, {r['joined']} cruzadas con un "
          f"partido, {r['matches']} con linea al escanear y al cierre\n")
    if r["slope"] is None:
        print("Aun no hay partidos con ambas lineas; vuelve a correrlo en unos dias.")
        return

    lo, hi = r["slope_ci"]
    print("A. El modelo predice el movimiento de la linea?")
    print(f"  movimiento medio |q_cierre - q_escaneo| {r['move']:.4f}")
    print(f"  pendiente de (q_cierre - q_escaneo) sobre (p_modelo - q_escaneo): {r['slope']:+.3f} "
          f"(IC 95% {lo:+.3f}..{hi:+.3f})")
    print("  > 0 con IC por encima de 0: la linea se mueve hacia el modelo (informacion no incorporada aun)\n")

    print(f"B. Picks de la regla de produccion ({MIN_EDGE:.0%} <= edge <= {MAX_SUSPICIOUS_EDGE:.0%}), N={r['picks']}")
    if not r["picks"]:
        return
    clo, chi = r["clv_ci"]
    print(f"  CLV medio {r['clv']:+.1%} (IC 95% {clo:+.1%}..{chi:+.1%}); p modelo {r['pick_p']:.3f} | "
          f"q escaneo {r['pick_q_scan']:.3f} | q cierre {r['pick_q_close']:.3f}")
    for tour, (n, c) in r["clv_by_tour"].items():
        print(f"  {tour.upper()}: N={n} CLV {c:+.1%}")


if __name__ == "__main__":
    main()

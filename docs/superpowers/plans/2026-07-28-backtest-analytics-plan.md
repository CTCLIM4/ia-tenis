# Módulo 2 (Backtest Analytics) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/backtest_analytics.py`, a console dashboard that reads `data/value_bets_log.csv` and `data/prediction_audit_log.csv` and reports financial, risk, and calibration metrics on resolved bets.

**Architecture:** One self-contained module (`load_*` → `compute_*` pure functions → `print_report` → `run_report` orchestrator → `main()` CLI), following the existing single-file-per-pipeline pattern (`daily_scanner.py`, `calibration_audit.py`). Uses `pandas`, already a hard dependency.

**Tech Stack:** Python, pandas, argparse, pytest.

**Spec:** `docs/superpowers/specs/2026-07-28-backtest-analytics-design.md` — read this first for the formulas and rationale; this plan implements it verbatim.

---

## Task 1: `load_resolved_bets` — filter + derive bet-side columns

**Files:**
- Create: `src/backtest_analytics.py`
- Create: `tests/test_backtest_analytics.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for src/backtest_analytics.py — Módulo 2 (backtest analytics dashboard)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_analytics import load_resolved_bets

_BETS_HEADER = "tour,match_date,status,result,kelly_a,kelly_b,odds_a,odds_b,ev_a,ev_b,profit"

_BETS_ROWS = [
    # Row A: real, resolved, side A bet, win
    "atp,2026-01-01,ok,A_win,0.05,0.0,2.0,1.9,0.1,-0.05,0.05",
    # Row B: real, resolved, side B bet, win
    "wta,2026-01-02,ok,B_win,0.0,0.03,1.8,2.1,-0.02,0.08,0.03",
    # Row C: excluded — missing Elo
    "atp,2026-01-04,invalid_missing_elo,A_win,0.05,0.0,2.0,1.9,0.1,-0.05,0.05",
    # Row D: excluded — still pending
    "atp,2026-01-05,ok,pending,0.05,0.0,2.0,1.9,0.1,-0.05,",
    # Row E: excluded — test fixture row
    "atp,2026-01-06,test_never_played,excluded,0.0,0.0,1.5,2.5,0.0,0.0,",
]


def _write_bets_csv(tmp_path, rows=_BETS_ROWS):
    path = tmp_path / "value_bets_log.csv"
    path.write_text(_BETS_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


class TestLoadResolvedBets:
    def test_filters_to_status_ok_and_resolved_result(self, tmp_path):
        df = load_resolved_bets(_write_bets_csv(tmp_path))
        assert len(df) == 2
        assert set(df["result"]) == {"A_win", "B_win"}

    def test_derives_bet_side_a(self, tmp_path):
        df = load_resolved_bets(_write_bets_csv(tmp_path))
        row = df[df["result"] == "A_win"].iloc[0]
        assert row["bet_side"] == "a"
        assert row["stake"] == pytest.approx(0.05)
        assert row["odds_taken"] == pytest.approx(2.0)
        assert row["ev_theoretical"] == pytest.approx(0.1)
        assert row["win"] == True  # noqa: E712

    def test_derives_bet_side_b(self, tmp_path):
        df = load_resolved_bets(_write_bets_csv(tmp_path))
        row = df[df["result"] == "B_win"].iloc[0]
        assert row["bet_side"] == "b"
        assert row["stake"] == pytest.approx(0.03)
        assert row["odds_taken"] == pytest.approx(2.1)
        assert row["ev_theoretical"] == pytest.approx(0.08)
        assert row["win"] == True  # noqa: E712

    def test_tour_filter(self, tmp_path):
        df = load_resolved_bets(_write_bets_csv(tmp_path), tour="atp")
        assert len(df) == 1
        assert df.iloc[0]["tour"] == "atp"

    def test_sorted_by_match_date(self, tmp_path):
        rows = list(reversed(_BETS_ROWS[:2]))  # B before A in the file
        df = load_resolved_bets(_write_bets_csv(tmp_path, rows))
        assert list(df["result"]) == ["A_win", "B_win"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest_analytics'` (module doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

Create `src/backtest_analytics.py`:

```python
"""Módulo 2: Dashboard de Rendimiento y Análisis de Banca.

Lee el histórico acumulado de data/value_bets_log.csv (apuestas logueadas)
y data/prediction_audit_log.csv (población completa de predicciones
evaluadas) y genera un reporte de consola con métricas financieras, de
riesgo, y una comparación de calibración (EV teórico vs beneficio real).

Uso:
  python -m src.backtest_analytics                    # banca $1000, ambos tours
  python -m src.backtest_analytics --tour atp
  python -m src.backtest_analytics --bankroll 5000

Ver docs/superpowers/specs/2026-07-28-backtest-analytics-design.md para el
diseño completo (fórmulas, filtros, y por qué).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

DEFAULT_BANKROLL = 1000.0


def load_resolved_bets(path: str, tour: Optional[str] = None) -> pd.DataFrame:
    """Load value_bets_log.csv and filter down to real, resolved bets.

    Keeps only status=='ok' and result in {'A_win','B_win'} — excludes test
    rows, missing-Elo rows, and unresolved/discarded rows, which would
    otherwise contaminate financial/risk metrics with data that isn't a real
    settled bet.

    Derives per-row bet_side/stake/odds_taken/ev_theoretical/win from
    whichever side (a/b) actually has kelly_<side> > 0 — deterministic since
    edge_a + edge_b <= 0 always (bookmaker vig), so at most one side can have
    positive Kelly for a given row.
    """
    df = pd.read_csv(path)
    df = df[(df["status"] == "ok") & (df["result"].isin(["A_win", "B_win"]))].copy()
    if tour and tour != "both":
        df = df[df["tour"] == tour]

    bet_side = df["kelly_a"].gt(0).map({True: "a", False: "b"})
    df["bet_side"] = bet_side
    df["stake"] = df["kelly_a"].where(bet_side == "a", df["kelly_b"])
    df["odds_taken"] = df["odds_a"].where(bet_side == "a", df["odds_b"])
    df["ev_theoretical"] = df["ev_a"].where(bet_side == "a", df["ev_b"])
    df["win"] = df["profit"] > 0
    df["match_date"] = pd.to_datetime(df["match_date"])

    return df.sort_values("match_date").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_analytics.py tests/test_backtest_analytics.py
git commit -m "feat: add load_resolved_bets to src/backtest_analytics.py"
```

---

## Task 2: `compute_financial_metrics`

**Files:**
- Modify: `src/backtest_analytics.py`
- Modify: `tests/test_backtest_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest_analytics.py`:

```python
from src.backtest_analytics import compute_financial_metrics  # add to existing import block


def _synthetic_resolved_df():
    """4 resolved bets with hand-computed expected metrics — see design spec
    Task 2/3/4 comments below for the arithmetic behind each assertion."""
    return pd.DataFrame({
        "match_date": pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04",
        ]),
        "profit":         [0.05, 0.03, -0.04, -0.02],
        "stake":          [0.05, 0.03, 0.04, 0.02],
        "ev_theoretical": [0.10, 0.08, 0.05, 0.02],
        "win":            [True, True, False, False],
    })


class TestComputeFinancialMetrics:
    def test_metrics(self):
        df = _synthetic_resolved_df()
        m = compute_financial_metrics(df, bankroll=1000.0)
        # total_profit=0.02, total_stake=0.14 -> roi = 0.02/0.14*100
        assert m["roi_pct"] == pytest.approx(14.285714, rel=1e-4)
        assert m["win_rate_pct"] == pytest.approx(50.0)
        assert m["wins"] == 2
        assert m["losses"] == 2
        assert m["net_profit_usd"] == pytest.approx(20.0)
        assert m["avg_stake_usd"] == pytest.approx(35.0)
        assert m["avg_stake_pct"] == pytest.approx(3.5)
        assert m["num_bets"] == 4

    def test_zero_stake_does_not_divide_by_zero(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2026-01-01"]),
            "profit": [0.0], "stake": [0.0],
            "ev_theoretical": [0.0], "win": [False],
        })
        m = compute_financial_metrics(df, bankroll=1000.0)
        assert m["roi_pct"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestComputeFinancialMetrics -v`
Expected: FAIL — `ImportError: cannot import name 'compute_financial_metrics'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest_analytics.py`:

```python
def compute_financial_metrics(df: pd.DataFrame, bankroll: float) -> dict:
    total_profit = df["profit"].sum()
    total_stake = df["stake"].sum()
    wins = int(df["win"].sum())
    return {
        "roi_pct": (total_profit / total_stake * 100) if total_stake else 0.0,
        "win_rate_pct": df["win"].mean() * 100,
        "wins": wins,
        "losses": len(df) - wins,
        "net_profit_usd": total_profit * bankroll,
        "avg_stake_usd": df["stake"].mean() * bankroll,
        "avg_stake_pct": df["stake"].mean() * 100,
        "num_bets": len(df),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestComputeFinancialMetrics -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_analytics.py tests/test_backtest_analytics.py
git commit -m "feat: add compute_financial_metrics to src/backtest_analytics.py"
```

---

## Task 3: `compute_risk_metrics` (equity curve, drawdown, variance, streaks)

**Files:**
- Modify: `src/backtest_analytics.py`
- Modify: `tests/test_backtest_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest_analytics.py`:

```python
from src.backtest_analytics import compute_risk_metrics  # add to existing import block


class TestComputeRiskMetrics:
    def test_metrics(self):
        df = _synthetic_resolved_df()
        r = compute_risk_metrics(df, bankroll=1000.0)
        # equity = 1000 + cumsum([.05,.03,-.04,-.02])*1000 = [1050,1080,1040,1020]
        # running_max = [1050,1080,1080,1080]; drawdown = [0,0,-40,-60]
        assert r["max_drawdown_usd"] == pytest.approx(-60.0)
        # -60 / 1080 * 100
        assert r["max_drawdown_pct"] == pytest.approx(-5.555556, rel=1e-4)
        # sample variance (ddof=1) of [0.05,0.03,-0.04,-0.02]
        assert r["variance"] == pytest.approx(0.0017667, rel=1e-3)
        # win sequence [T,T,F,F]
        assert r["max_win_streak"] == 2
        assert r["max_loss_streak"] == 2

    def test_streak_breaks_correctly(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime([f"2026-01-0{i}" for i in range(1, 6)]),
            "profit": [0.01, 0.01, 0.01, -0.01, 0.01],
            "stake": [0.01] * 5,
            "ev_theoretical": [0.05] * 5,
            "win": [True, True, True, False, True],
        })
        r = compute_risk_metrics(df, bankroll=1000.0)
        assert r["max_win_streak"] == 3
        assert r["max_loss_streak"] == 1

    def test_single_row_variance_is_zero(self):
        df = pd.DataFrame({
            "match_date": pd.to_datetime(["2026-01-01"]),
            "profit": [0.02], "stake": [0.02],
            "ev_theoretical": [0.02], "win": [True],
        })
        r = compute_risk_metrics(df, bankroll=1000.0)
        assert r["variance"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestComputeRiskMetrics -v`
Expected: FAIL — `ImportError: cannot import name 'compute_risk_metrics'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest_analytics.py`:

```python
def _max_streak(values: list, target: bool) -> int:
    best = current = 0
    for v in values:
        if v == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def compute_risk_metrics(df: pd.DataFrame, bankroll: float) -> dict:
    ordered = df.sort_values("match_date")
    equity = bankroll + ordered["profit"].cumsum() * bankroll
    running_max = equity.cummax()
    drawdown_usd = equity - running_max

    max_drawdown_usd = drawdown_usd.min()
    trough_idx = drawdown_usd.idxmin()
    peak_at_trough = running_max.loc[trough_idx]
    max_drawdown_pct = (max_drawdown_usd / peak_at_trough * 100) if peak_at_trough else 0.0

    variance = ordered["profit"].var(ddof=1) if len(ordered) > 1 else 0.0

    win_sequence = ordered["win"].tolist()
    return {
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "variance": 0.0 if pd.isna(variance) else variance,
        "max_win_streak": _max_streak(win_sequence, True),
        "max_loss_streak": _max_streak(win_sequence, False),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestComputeRiskMetrics -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_analytics.py tests/test_backtest_analytics.py
git commit -m "feat: add compute_risk_metrics to src/backtest_analytics.py"
```

---

## Task 4: `compute_calibration_metrics`

**Files:**
- Modify: `src/backtest_analytics.py`
- Modify: `tests/test_backtest_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest_analytics.py`:

```python
from src.backtest_analytics import compute_calibration_metrics  # add to existing import block


class TestComputeCalibrationMetrics:
    def test_metrics(self):
        df = _synthetic_resolved_df()
        c = compute_calibration_metrics(df)
        # ev_theoretical mean = (.10+.08+.05+.02)/4 = 0.0625
        assert c["ev_theoretical_avg"] == pytest.approx(0.0625)
        # real_return per row = profit/stake = [1.0, 1.0, -1.0, -1.0] -> mean 0.0
        assert c["real_return_avg"] == pytest.approx(0.0)
        assert c["difference"] == pytest.approx(-0.0625)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestComputeCalibrationMetrics -v`
Expected: FAIL — `ImportError: cannot import name 'compute_calibration_metrics'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest_analytics.py`:

```python
def compute_calibration_metrics(df: pd.DataFrame) -> dict:
    ev_theoretical_avg = df["ev_theoretical"].mean()
    real_return_avg = (df["profit"] / df["stake"]).mean()
    return {
        "ev_theoretical_avg": ev_theoretical_avg,
        "real_return_avg": real_return_avg,
        "difference": real_return_avg - ev_theoretical_avg,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestComputeCalibrationMetrics -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_analytics.py tests/test_backtest_analytics.py
git commit -m "feat: add compute_calibration_metrics to src/backtest_analytics.py"
```

---

## Task 5: `load_audit_log` + `compute_audit_coverage`

**Files:**
- Modify: `src/backtest_analytics.py`
- Modify: `tests/test_backtest_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest_analytics.py`:

```python
from src.backtest_analytics import (  # add to existing import block
    compute_audit_coverage,
    load_audit_log,
)

_AUDIT_HEADER = "tour,decision"
_AUDIT_ROWS = [
    "atp,logged", "wta,logged",
    "atp,passed_low_edge",
    "wta,blocked_suspicious", "atp,blocked_suspicious", "atp,blocked_suspicious",
]


def _write_audit_csv(tmp_path):
    path = tmp_path / "prediction_audit_log.csv"
    path.write_text(_AUDIT_HEADER + "\n" + "\n".join(_AUDIT_ROWS) + "\n", encoding="utf-8")
    return str(path)


class TestLoadAuditLog:
    def test_loads_all_rows(self, tmp_path):
        df = load_audit_log(_write_audit_csv(tmp_path))
        assert len(df) == 6


class TestComputeAuditCoverage:
    def test_counts_by_decision(self, tmp_path):
        audit_df = load_audit_log(_write_audit_csv(tmp_path))
        cov = compute_audit_coverage(audit_df)
        assert cov["total"] == 6
        assert cov["by_decision"] == {
            "logged": 2,
            "passed_low_edge": 1,
            "passed_user_declined": 0,
            "blocked_suspicious": 3,
            "invalid_missing_elo": 0,
        }
        assert cov["logged"] == 2
        assert cov["passed"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestComputeAuditCoverage -v`
Expected: FAIL — `ImportError: cannot import name 'load_audit_log'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest_analytics.py`:

```python
_DECISION_CATEGORIES = [
    "logged", "passed_low_edge", "passed_user_declined",
    "blocked_suspicious", "invalid_missing_elo",
]


def load_audit_log(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_audit_coverage(audit_df: pd.DataFrame) -> dict:
    counts = audit_df["decision"].value_counts()
    by_decision = {cat: int(counts.get(cat, 0)) for cat in _DECISION_CATEGORIES}
    total = len(audit_df)
    logged = by_decision["logged"]
    return {
        "total": total,
        "by_decision": by_decision,
        "logged": logged,
        "passed": total - logged,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestLoadAuditLog tests/test_backtest_analytics.py::TestComputeAuditCoverage -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_analytics.py tests/test_backtest_analytics.py
git commit -m "feat: add load_audit_log and compute_audit_coverage to src/backtest_analytics.py"
```

---

## Task 6: `print_report`

**Files:**
- Modify: `src/backtest_analytics.py`
- Modify: `tests/test_backtest_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest_analytics.py`:

```python
from src.backtest_analytics import print_report  # add to existing import block


class TestPrintReport:
    def test_report_contains_all_sections(self, capsys):
        df = _synthetic_resolved_df()
        financial = compute_financial_metrics(df, bankroll=1000.0)
        risk = compute_risk_metrics(df, bankroll=1000.0)
        calibration = compute_calibration_metrics(df)
        coverage = {
            "total": 6,
            "by_decision": {
                "logged": 2, "passed_low_edge": 1, "passed_user_declined": 0,
                "blocked_suspicious": 3, "invalid_missing_elo": 0,
            },
            "logged": 2, "passed": 4,
        }

        print_report(df, financial, risk, calibration, coverage, bankroll=1000.0)
        out = capsys.readouterr().out

        assert "REPORTE DE RENDIMIENTO Y BANCA" in out
        assert "$1,000.00" in out
        assert "2026-01-01 a 2026-01-04" in out
        assert "METRICAS FINANCIERAS" in out
        assert "METRICAS DE RIESGO" in out
        assert "CALIBRACION" in out
        assert "COBERTURA DEL AUDIT LOG" in out
        assert "Muestra pequena" in out  # N=4 < 30

    def test_report_without_coverage(self, capsys):
        df = _synthetic_resolved_df()
        financial = compute_financial_metrics(df, bankroll=1000.0)
        risk = compute_risk_metrics(df, bankroll=1000.0)
        calibration = compute_calibration_metrics(df)

        print_report(df, financial, risk, calibration, None, bankroll=1000.0)
        out = capsys.readouterr().out

        assert "COBERTURA DEL AUDIT LOG: no disponible" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestPrintReport -v`
Expected: FAIL — `ImportError: cannot import name 'print_report'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest_analytics.py`:

```python
SMALL_SAMPLE_THRESHOLD = 30


def print_report(
    df: pd.DataFrame,
    financial: dict,
    risk: dict,
    calibration: dict,
    coverage: Optional[dict],
    bankroll: float,
) -> None:
    n = len(df)
    period_start = df["match_date"].min().date().isoformat()
    period_end = df["match_date"].max().date().isoformat()

    print("=" * 60)
    print(" REPORTE DE RENDIMIENTO Y BANCA - Modulo 2")
    print(f" Banca base: ${bankroll:,.2f} | Periodo: {period_start} a {period_end} "
          f"| Apuestas: {n}")
    print("=" * 60)

    if n < SMALL_SAMPLE_THRESHOLD:
        print(f"\n  *** Muestra pequena (N={n} < {SMALL_SAMPLE_THRESHOLD}) - "
              "metricas poco confiables todavia. ***")

    print("\nMETRICAS FINANCIERAS")
    print(f"  ROI acumulado:            {financial['roi_pct']:+.2f}%")
    print(f"  Win Rate:                   {financial['win_rate_pct']:.1f}%   "
          f"({financial['wins']}W / {financial['losses']}L)")
    print(f"  Beneficio Neto:           {financial['net_profit_usd']:+,.2f}$")
    print(f"  Stake promedio:            {financial['avg_stake_usd']:,.2f}$   "
          f"({financial['avg_stake_pct']:.2f}% banca)")
    print(f"  Apuestas procesadas:      {financial['num_bets']:>5}")

    print("\nMETRICAS DE RIESGO")
    print(f"  Drawdown maximo:          {risk['max_drawdown_usd']:+,.2f}$   "
          f"({risk['max_drawdown_pct']:+.1f}%)")
    print(f"  Varianza (profit):        {risk['variance']:.5f}")
    print(f"  Racha ganadora maxima:    {risk['max_win_streak']:>5}")
    print(f"  Racha perdedora maxima:   {risk['max_loss_streak']:>5}")

    print("\nCALIBRACION: EV TEORICO vs BENEFICIO REAL")
    print(f"  EV teorico promedio (por unidad):    "
          f"{calibration['ev_theoretical_avg'] * 100:+.1f}%")
    print(f"  Retorno real promedio (por unidad):  "
          f"{calibration['real_return_avg'] * 100:+.1f}%")
    print(f"  Diferencia (real - teorico):         "
          f"{calibration['difference'] * 100:+.1f} pp")

    if coverage is not None:
        print("\nCOBERTURA DEL AUDIT LOG (prediction_audit_log.csv)")
        print(f"  Predicciones evaluadas totales: {coverage['total']:>4}")
        for cat, count in coverage["by_decision"].items():
            pct = (count / coverage["total"] * 100) if coverage["total"] else 0.0
            print(f"    {cat:<22}{count:>4} ({pct:.1f}%)")
        print(f"  Logged (apostadas) vs resto:  {coverage['logged']} vs {coverage['passed']}")
    else:
        print("\nCOBERTURA DEL AUDIT LOG: no disponible "
              "(prediction_audit_log.csv no encontrado).")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestPrintReport -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_analytics.py tests/test_backtest_analytics.py
git commit -m "feat: add print_report to src/backtest_analytics.py"
```

---

## Task 7: `run_report` orchestration + error handling

**Files:**
- Modify: `src/backtest_analytics.py`
- Modify: `tests/test_backtest_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest_analytics.py`:

```python
from src.backtest_analytics import run_report  # add to existing import block


class TestRunReport:
    def test_missing_bets_file(self, tmp_path, capsys):
        run_report(bets_path=str(tmp_path / "nope.csv"), audit_path=str(tmp_path / "nope2.csv"))
        out = capsys.readouterr().out
        assert "No se encontro" in out

    def test_zero_resolved_bets(self, tmp_path, capsys):
        rows = ["atp,2026-01-05,ok,pending,0.05,0.0,2.0,1.9,0.1,-0.05,"]
        bets_path = _write_bets_csv(tmp_path, rows)
        run_report(bets_path=bets_path, audit_path=str(tmp_path / "nope.csv"))
        out = capsys.readouterr().out
        assert "No hay apuestas resueltas todavia" in out

    def test_normal_run_without_audit_file(self, tmp_path, capsys):
        bets_path = _write_bets_csv(tmp_path)  # 2 resolved rows (A_win, B_win)
        run_report(bets_path=bets_path, audit_path=str(tmp_path / "missing_audit.csv"))
        out = capsys.readouterr().out
        assert "REPORTE DE RENDIMIENTO Y BANCA" in out
        assert "Aviso: no se encontro" in out
        assert "COBERTURA DEL AUDIT LOG: no disponible" in out

    def test_normal_run_with_audit_file(self, tmp_path, capsys):
        bets_path = _write_bets_csv(tmp_path)
        audit_path = _write_audit_csv(tmp_path)
        run_report(bets_path=bets_path, audit_path=audit_path, bankroll=500.0, tour="atp")
        out = capsys.readouterr().out
        assert "REPORTE DE RENDIMIENTO Y BANCA" in out
        assert "$500.00" in out
        assert "COBERTURA DEL AUDIT LOG (prediction_audit_log.csv)" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestRunReport -v`
Expected: FAIL — `ImportError: cannot import name 'run_report'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest_analytics.py`:

```python
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BETS_PATH = str(_ROOT / "data" / "value_bets_log.csv")
DEFAULT_AUDIT_PATH = str(_ROOT / "data" / "prediction_audit_log.csv")


def run_report(
    bets_path: str = DEFAULT_BETS_PATH,
    audit_path: str = DEFAULT_AUDIT_PATH,
    bankroll: float = DEFAULT_BANKROLL,
    tour: Optional[str] = None,
) -> None:
    if not Path(bets_path).exists():
        print(f"No se encontro {bets_path}. Corre el modulo de value bets primero "
              "para generar historial.")
        return

    df = load_resolved_bets(bets_path, tour)
    if df.empty:
        print("No hay apuestas resueltas todavia (status='ok' y result en A_win/B_win).")
        return

    financial = compute_financial_metrics(df, bankroll)
    risk = compute_risk_metrics(df, bankroll)
    calibration = compute_calibration_metrics(df)

    coverage = None
    if Path(audit_path).exists():
        audit_df = load_audit_log(audit_path)
        coverage = compute_audit_coverage(audit_df)
    else:
        print(f"Aviso: no se encontro {audit_path} - se omite la seccion de "
              "cobertura del audit log.")

    print_report(df, financial, risk, calibration, coverage, bankroll)
```

Move the `_ROOT`/`DEFAULT_BETS_PATH`/`DEFAULT_AUDIT_PATH` constants to the top of the file (next to `DEFAULT_BANKROLL`) rather than inline here — this snippet shows where they're introduced, but place them alongside the other module-level constants for readability.

- [ ] **Step 4: Run test to verify it passes**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestRunReport -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_analytics.py tests/test_backtest_analytics.py
git commit -m "feat: add run_report orchestration to src/backtest_analytics.py"
```

---

## Task 8: CLI `main()`

**Files:**
- Modify: `src/backtest_analytics.py`
- Modify: `tests/test_backtest_analytics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backtest_analytics.py`:

```python
import sys  # add to existing import block

from src.backtest_analytics import main  # add to existing import block


class TestMain:
    def test_cli_wires_args_into_run_report(self, tmp_path, capsys, monkeypatch):
        bets_path = _write_bets_csv(tmp_path)
        audit_path = _write_audit_csv(tmp_path)
        argv = [
            "backtest_analytics",
            "--bankroll", "2000",
            "--tour", "wta",
            "--bets-file", bets_path,
            "--audit-file", audit_path,
        ]
        monkeypatch.setattr(sys, "argv", argv)

        main()
        out = capsys.readouterr().out

        assert "$2,000.00" in out
        assert "Apuestas: 1" in out  # only the wta row (B_win) survives --tour wta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestMain -v`
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/backtest_analytics.py`:

```python
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dashboard de rendimiento y analisis de banca (Modulo 2)"
    )
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--tour", choices=["atp", "wta", "both"], default="both")
    parser.add_argument("--bets-file", default=DEFAULT_BETS_PATH)
    parser.add_argument("--audit-file", default=DEFAULT_AUDIT_PATH)
    args = parser.parse_args()

    tour = None if args.tour == "both" else args.tour
    run_report(args.bets_file, args.audit_file, args.bankroll, tour)


if __name__ == "__main__":
    main()
```

Move the `import argparse` line to the top of the file with the other imports rather than leaving it inline here.

- [ ] **Step 4: Run test to verify it passes**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_backtest_analytics.py::TestMain -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_analytics.py tests/test_backtest_analytics.py
git commit -m "feat: add CLI entry point to src/backtest_analytics.py"
```

---

## Task 9: Full suite + real-data smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run the full project test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no regressions in any other module

- [ ] **Step 2: Smoke test against the real repo CSVs**

Run: `./tenis-env/Scripts/python.exe -m src.backtest_analytics`
Expected: prints a full report (as of this plan being written, `data/value_bets_log.csv` has 4 rows with `status=='ok'` and `result` resolved, and `data/prediction_audit_log.csv` has 2 rows both `decision=='logged'` — confirm the printed numbers match what those rows produce, and confirm the small-sample warning appears since N=4 < 30).

- [ ] **Step 3: Smoke test the CLI flags**

Run: `./tenis-env/Scripts/python.exe -m src.backtest_analytics --tour atp --bankroll 500`
Expected: report reflects only ATP resolved rows and a $500 base bankroll in the header

- [ ] **Step 4: Fix anything the smoke test surfaces, then final commit if needed**

If Steps 2-3 reveal a formatting or logic issue not caught by the unit tests, fix it in `src/backtest_analytics.py` and commit:

```bash
git add src/backtest_analytics.py
git commit -m "fix: address issue found in backtest_analytics real-data smoke test"
```

If nothing needed fixing, no commit is required for this task.

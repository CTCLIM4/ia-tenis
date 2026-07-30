# Módulo 4 (Bankroll Simulation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/bankroll_simulation.py`, a Monte Carlo simulator that projects future bankroll trajectories under different Kelly-fraction multipliers (full/half/quarter), calibrated from the edge/odds profile of already-resolved bets in `data/value_bets_log.csv`.

**Architecture:** One self-contained module (`estimate_bet_profile` → `sample_bet` → `kelly_stake` → `simulate_path` → `run_monte_carlo` → `print_report` → `run_simulation` orchestrator → `main()` CLI), following the existing single-file-per-pipeline pattern (`backtest_analytics.py`, `daily_scanner.py`). Reuses `backtest_analytics.load_resolved_bets`/`SMALL_SAMPLE_THRESHOLD` and `value_analysis.KELLY_CAP` rather than duplicating them.

**Tech Stack:** Python, pandas, numpy (`np.random.Generator`), argparse, pytest.

**Spec:** `docs/superpowers/specs/2026-07-29-bankroll-simulation-design.md` — read this first for the formulas and rationale; this plan implements it verbatim.

---

## Task 1: `estimate_bet_profile`

**Files:**
- Create: `src/bankroll_simulation.py`
- Create: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for src/bankroll_simulation.py — Módulo 4 (bankroll Monte Carlo simulation)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.bankroll_simulation import estimate_bet_profile


def _resolved_df_for_profile():
    """3 resolved bets shaped like backtest_analytics.load_resolved_bets output
    (has bet_side/odds_taken derived, plus the raw edge_a/edge_b columns)."""
    return pd.DataFrame({
        "bet_side":  ["a", "b", "a"],
        "edge_a":    [0.05, -0.05, 0.03],
        "edge_b":    [-0.1, 0.08, -0.06],
        "odds_taken": [2.0, 1.8, 2.5],
    })


class TestEstimateBetProfile:
    def test_computes_mean_and_std_from_bet_side_edge(self):
        profile = estimate_bet_profile(_resolved_df_for_profile())
        # edge_taken = [0.05, 0.08, 0.03] (side actually bet on each row)
        assert profile["mean_edge"] == pytest.approx(0.0533333, rel=1e-4)
        assert profile["std_edge"] == pytest.approx(0.0251661, rel=1e-4)
        assert profile["mean_odds"] == pytest.approx(2.1)
        assert profile["std_odds"] == pytest.approx(0.3605551, rel=1e-4)
        assert profile["n"] == 3

    def test_single_row_has_zero_std(self):
        df = pd.DataFrame({
            "bet_side": ["a"], "edge_a": [0.05], "edge_b": [-0.1],
            "odds_taken": [2.0],
        })
        profile = estimate_bet_profile(df)
        assert profile["mean_edge"] == pytest.approx(0.05)
        assert profile["std_edge"] == 0.0
        assert profile["mean_odds"] == pytest.approx(2.0)
        assert profile["std_odds"] == 0.0
        assert profile["n"] == 1

    def test_empty_df_raises(self):
        df = pd.DataFrame({"bet_side": [], "edge_a": [], "edge_b": [], "odds_taken": []})
        with pytest.raises(ValueError):
            estimate_bet_profile(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.bankroll_simulation'`

- [ ] **Step 3: Write minimal implementation**

Create `src/bankroll_simulation.py`:

```python
"""Módulo 4: Simulación Monte Carlo de Banca (Kelly).

Proyecta trayectorias futuras de banca bajo distintas fracciones de Kelly
(completo/medio/cuarto), calibradas con el perfil de edge/odds de las
apuestas ya resueltas en data/value_bets_log.csv, para estimar banca
esperada (P10/P50/P90) y probabilidad de ruina a N apuestas.

Uso:
  python -m src.bankroll_simulation
  python -m src.bankroll_simulation --bets 100 --simulations 20000
  python -m src.bankroll_simulation --mean-edge 0.05 --std-edge 0.02 \
      --mean-odds 2.0 --std-odds 0.3   # perfil manual, sin leer el CSV

Ver docs/superpowers/specs/2026-07-29-bankroll-simulation-design.md para el
diseño completo (fórmulas y rationale).
"""
from __future__ import annotations

import pandas as pd


def estimate_bet_profile(df: pd.DataFrame) -> dict:
    """Derive an edge/odds sampling profile from resolved bets.

    Expects the DataFrame shape produced by
    backtest_analytics.load_resolved_bets (has bet_side and odds_taken
    already derived, plus the raw edge_a/edge_b columns). std falls back to
    0.0 when there's only one row (nothing to estimate dispersion from).
    """
    if df.empty:
        raise ValueError("No hay apuestas resueltas de donde estimar el perfil.")

    edge_taken = df["edge_a"].where(df["bet_side"] == "a", df["edge_b"])
    n = len(df)
    return {
        "mean_edge": edge_taken.mean(),
        "std_edge": edge_taken.std(ddof=1) if n > 1 else 0.0,
        "mean_odds": df["odds_taken"].mean(),
        "std_odds": df["odds_taken"].std(ddof=1) if n > 1 else 0.0,
        "n": n,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "feat: add estimate_bet_profile to src/bankroll_simulation.py"
```

---

## Task 2: `sample_bet`

**Files:**
- Modify: `src/bankroll_simulation.py`
- Modify: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bankroll_simulation.py`:

```python
from src.bankroll_simulation import sample_bet  # add to existing import block


class TestSampleBet:
    def test_deterministic_when_std_is_zero(self):
        profile = {"mean_edge": 0.05, "std_edge": 0.0, "mean_odds": 2.0, "std_odds": 0.0, "n": 5}
        rng = np.random.default_rng(0)
        p_win, odds = sample_bet(rng, profile)
        # odds collapses to mean_odds exactly (sigma=0 lognormal)
        assert odds == pytest.approx(2.0)
        # edge collapses to mean_edge exactly (sigma=0 normal); implied=1/2.0=0.5
        assert p_win == pytest.approx(0.55)

    def test_edge_floor_prevents_nonpositive_edge(self):
        profile = {"mean_edge": -0.02, "std_edge": 0.0, "mean_odds": 2.0, "std_odds": 0.0, "n": 5}
        rng = np.random.default_rng(0)
        p_win, odds = sample_bet(rng, profile)
        # edge floored to 0.001; implied=0.5 -> p_win=0.501
        assert p_win == pytest.approx(0.501)

    def test_seed_reproducibility_with_dispersion(self):
        profile = {"mean_edge": 0.05, "std_edge": 0.02, "mean_odds": 2.0, "std_odds": 0.3, "n": 10}
        r1 = sample_bet(np.random.default_rng(42), profile)
        r2 = sample_bet(np.random.default_rng(42), profile)
        assert r1 == r2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestSampleBet -v`
Expected: FAIL — `ImportError: cannot import name 'sample_bet'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/bankroll_simulation.py` (add `import math` and `import numpy as np` to the top imports):

```python
def sample_bet(rng: np.random.Generator, profile: dict) -> tuple[float, float]:
    """Draw one synthetic (p_win, odds) pair from the calibrated profile.

    edge ~ Normal(mean_edge, std_edge), floored above 0 (the historical log
    only ever contains positive-edge bets by construction of
    value_analysis.calculate_value, so a non-positive draw isn't a
    realistic sample — it's floored rather than treated as "skip this bet",
    which would complicate comparing Kelly fractions on equal footing).

    odds ~ Lognormal fit by method of moments to have the given mean/std
    (collapses to exactly mean_odds when std_odds == 0, since sigma
    becomes 0). p_win is implied_prob (1/odds) + edge, clipped away from
    0/1 so Kelly can't degenerate.
    """
    edge = rng.normal(profile["mean_edge"], profile["std_edge"])
    edge = max(edge, 0.001)

    mean_odds = profile["mean_odds"]
    std_odds = profile["std_odds"]
    sigma2 = math.log(1 + (std_odds / mean_odds) ** 2)
    mu = math.log(mean_odds) - sigma2 / 2
    odds = rng.lognormal(mu, math.sqrt(sigma2))
    odds = max(odds, 1.01)

    implied_prob = 1 / odds
    p_win = min(max(implied_prob + edge, 0.01), 0.99)
    return p_win, odds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestSampleBet -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "feat: add sample_bet to src/bankroll_simulation.py"
```

---

## Task 3: `kelly_stake`

**Files:**
- Modify: `src/bankroll_simulation.py`
- Modify: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bankroll_simulation.py`:

```python
from src.bankroll_simulation import kelly_stake  # add to existing import block


class TestKellyStake:
    def test_positive_edge_scales_with_multiplier(self):
        # p_win=0.55, odds=2.0 -> implied=0.5, edge=0.05, raw=0.05/(1.0)=0.05
        assert kelly_stake(0.55, 2.0, kelly_multiplier=1.0) == pytest.approx(0.05)
        assert kelly_stake(0.55, 2.0, kelly_multiplier=0.5) == pytest.approx(0.025)

    def test_negative_edge_returns_zero(self):
        # p_win=0.3, odds=2.0 -> implied=0.5, edge=-0.2 -> no bet
        assert kelly_stake(0.3, 2.0, kelly_multiplier=1.0) == 0.0

    def test_cap_dominates_at_high_edge(self):
        # p_win=0.9, odds=1.5 -> implied=0.6667, edge=0.2333, raw=0.4667
        # full Kelly and 1/4 Kelly both still hit the 5% cap
        assert kelly_stake(0.9, 1.5, kelly_multiplier=1.0) == pytest.approx(0.05)
        assert kelly_stake(0.9, 1.5, kelly_multiplier=0.25) == pytest.approx(0.05)
        # 1/10 Kelly finally drops below the cap: 0.1 * 0.4667 = 0.04667
        assert kelly_stake(0.9, 1.5, kelly_multiplier=0.1) == pytest.approx(0.046667, rel=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestKellyStake -v`
Expected: FAIL — `ImportError: cannot import name 'kelly_stake'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/bankroll_simulation.py` (add `from src.value_analysis import KELLY_CAP` to the top imports):

```python
def kelly_stake(p_win: float, odds: float, kelly_multiplier: float) -> float:
    """Same convention as value_analysis.calculate_value's Kelly formula,
    with an extra multiplier applied before the cap (so half/quarter Kelly
    only differ from full Kelly when the cap isn't already binding)."""
    implied_prob = 1 / odds
    edge = p_win - implied_prob
    raw_kelly = edge / (odds - 1) if edge > 0 else 0.0
    return min(kelly_multiplier * raw_kelly, KELLY_CAP)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestKellyStake -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "feat: add kelly_stake to src/bankroll_simulation.py"
```

---

## Task 4: `simulate_path`

**Files:**
- Modify: `src/bankroll_simulation.py`
- Modify: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bankroll_simulation.py`:

```python
from src.bankroll_simulation import simulate_path  # add to existing import block


class _FixedRNG:
    """Deterministic stand-in for np.random.Generator. normal()/lognormal()
    collapse to their location parameter (only used with std=0 profiles in
    these tests), and random() replays a fixed win/loss sequence — lets us
    hand-verify the bankroll arithmetic exactly instead of trusting opaque
    Generator internals."""

    def __init__(self, wins):
        self._wins = list(wins)
        self._i = 0

    def normal(self, loc, scale):
        return loc

    def lognormal(self, mean, sigma):
        return math.exp(mean)

    def random(self):
        win = self._wins[self._i]
        self._i += 1
        return 0.0 if win else 0.999


class TestSimulatePath:
    def test_hand_verified_bankroll_sequence(self):
        # profile is deterministic (std=0 both): every sampled bet is
        # p_win=0.55, odds=2.0 -> kelly_stake(1.0) = 0.05 (at the cap)
        profile = {"mean_edge": 0.05, "std_edge": 0.0, "mean_odds": 2.0, "std_odds": 0.0, "n": 5}
        rng = _FixedRNG(wins=[True, False, True])
        path = simulate_path(rng, profile, kelly_multiplier=1.0, n_bets=3, initial_bankroll=1000.0)

        # bet1: stake=50,  win  -> 1000 + 50*(2.0-1)   = 1050.0
        # bet2: stake=52.5,lose -> 1050 - 52.5         = 997.5
        # bet3: stake=49.875,win-> 997.5 + 49.875*1.0  = 1047.375
        expected = np.array([1000.0, 1050.0, 997.5, 1047.375])
        np.testing.assert_allclose(path, expected)

    def test_bankroll_never_negative(self):
        # extreme profile: huge edge relative to odds still caps at 5%/bet,
        # so bankroll shrinks but must never cross zero even on an
        # all-losses run.
        profile = {"mean_edge": 0.05, "std_edge": 0.0, "mean_odds": 2.0, "std_odds": 0.0, "n": 5}
        rng = _FixedRNG(wins=[False] * 50)
        path = simulate_path(rng, profile, kelly_multiplier=1.0, n_bets=50, initial_bankroll=1000.0)
        assert (path >= 0.0).all()

    def test_same_seed_reproducible(self):
        profile = {"mean_edge": 0.05, "std_edge": 0.02, "mean_odds": 2.0, "std_odds": 0.3, "n": 10}
        p1 = simulate_path(np.random.default_rng(7), profile, 1.0, n_bets=20, initial_bankroll=1000.0)
        p2 = simulate_path(np.random.default_rng(7), profile, 1.0, n_bets=20, initial_bankroll=1000.0)
        np.testing.assert_array_equal(p1, p2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestSimulatePath -v`
Expected: FAIL — `ImportError: cannot import name 'simulate_path'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/bankroll_simulation.py`:

```python
def simulate_path(
    rng: np.random.Generator,
    profile: dict,
    kelly_multiplier: float,
    n_bets: int,
    initial_bankroll: float,
) -> np.ndarray:
    """Simulate one bankroll trajectory of n_bets synthetic bets.

    Stakes a fraction of the CURRENT bankroll each bet (compounding), not
    a fraction of the fixed initial bankroll — this is the whole point of
    the Kelly criterion (geometric growth) and is why half-Kelly reduces
    ruin risk so much in practice. This differs deliberately from Módulo
    2's linear equity curve, which is correct for reporting a fixed
    historical sequence but not for a forward-looking Kelly simulation.
    """
    bankroll = initial_bankroll
    path = [bankroll]
    for _ in range(n_bets):
        p_win, odds = sample_bet(rng, profile)
        stake_frac = kelly_stake(p_win, odds, kelly_multiplier)
        stake_usd = stake_frac * bankroll
        if rng.random() < p_win:
            bankroll += stake_usd * (odds - 1)
        else:
            bankroll -= stake_usd
        bankroll = max(bankroll, 0.0)
        path.append(bankroll)
    return np.array(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestSimulatePath -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "feat: add simulate_path to src/bankroll_simulation.py"
```

---

## Task 5: `run_monte_carlo`

**Files:**
- Modify: `src/bankroll_simulation.py`
- Modify: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bankroll_simulation.py`:

```python
from src.bankroll_simulation import run_monte_carlo  # add to existing import block

_MC_PROFILE = {"mean_edge": 0.05, "std_edge": 0.02, "mean_odds": 2.0, "std_odds": 0.3, "n": 10}


class TestRunMonteCarlo:
    def test_output_structure_and_sane_ranges(self):
        result = run_monte_carlo(
            _MC_PROFILE, kelly_multipliers=[1.0, 0.5, 0.25],
            n_simulations=200, n_bets=15, initial_bankroll=1000.0,
            ruin_threshold=0.5, seed=42,
        )
        assert set(result["results"].keys()) == {1.0, 0.5, 0.25}
        assert result["n_simulations"] == 200
        assert result["n_bets"] == 15
        assert result["initial_bankroll"] == 1000.0
        assert result["ruin_threshold"] == 0.5
        assert result["profile"] == _MC_PROFILE

        for stats in result["results"].values():
            assert stats["p10"] <= stats["p50"] <= stats["p90"]
            assert stats["p10"] >= 0.0
            assert 0.0 <= stats["ruin_probability"] <= 1.0
            assert stats["median_max_drawdown_pct"] <= 0.0

    def test_same_seed_is_reproducible(self):
        kwargs = dict(
            kelly_multipliers=[1.0], n_simulations=50, n_bets=10,
            initial_bankroll=1000.0, ruin_threshold=0.5, seed=7,
        )
        r1 = run_monte_carlo(_MC_PROFILE, **kwargs)
        r2 = run_monte_carlo(_MC_PROFILE, **kwargs)
        assert r1 == r2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestRunMonteCarlo -v`
Expected: FAIL — `ImportError: cannot import name 'run_monte_carlo'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/bankroll_simulation.py`:

```python
def run_monte_carlo(
    profile: dict,
    kelly_multipliers: list[float],
    n_simulations: int,
    n_bets: int,
    initial_bankroll: float,
    ruin_threshold: float,
    seed: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    results = {}
    for km in kelly_multipliers:
        finals = []
        max_drawdowns_pct = []
        ruined = 0
        for _ in range(n_simulations):
            path = simulate_path(rng, profile, km, n_bets, initial_bankroll)
            finals.append(path[-1])
            running_max = np.maximum.accumulate(path)
            drawdown_pct = np.where(running_max > 0, (path - running_max) / running_max, 0.0)
            max_drawdowns_pct.append(drawdown_pct.min())
            if path.min() < ruin_threshold * initial_bankroll:
                ruined += 1
        finals = np.array(finals)
        results[km] = {
            "p10": float(np.percentile(finals, 10)),
            "p50": float(np.percentile(finals, 50)),
            "p90": float(np.percentile(finals, 90)),
            "ruin_probability": ruined / n_simulations,
            "median_max_drawdown_pct": float(np.median(max_drawdowns_pct)) * 100,
        }

    return {
        "profile": profile,
        "results": results,
        "n_simulations": n_simulations,
        "n_bets": n_bets,
        "initial_bankroll": initial_bankroll,
        "ruin_threshold": ruin_threshold,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestRunMonteCarlo -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "feat: add run_monte_carlo to src/bankroll_simulation.py"
```

---

## Task 6: `print_report`

**Files:**
- Modify: `src/bankroll_simulation.py`
- Modify: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bankroll_simulation.py`:

```python
from src.bankroll_simulation import print_report  # add to existing import block


def _mc_result(n=7):
    return {
        "profile": {"mean_edge": 0.058, "std_edge": 0.041, "mean_odds": 2.03, "std_odds": 0.60, "n": n},
        "results": {
            1.0:  {"p10": 612.34, "p50": 1340.55, "p90": 2850.90, "ruin_probability": 0.184, "median_max_drawdown_pct": -34.2},
            0.5:  {"p10": 780.11, "p50": 1190.20, "p90": 1950.44, "ruin_probability": 0.062, "median_max_drawdown_pct": -19.8},
            0.25: {"p10": 890.02, "p50": 1080.15, "p90": 1420.33, "ruin_probability": 0.011, "median_max_drawdown_pct": -10.5},
        },
        "n_simulations": 10000, "n_bets": 50,
        "initial_bankroll": 1000.0, "ruin_threshold": 0.5,
    }


class TestPrintReport:
    def test_report_contains_expected_sections(self, capsys):
        print_report(_mc_result())
        out = capsys.readouterr().out
        assert "SIMULACION MONTE CARLO DE BANCA" in out
        assert "$1,000.00" in out
        assert "10000 simulaciones x 50 apuestas" in out
        assert "N=7" in out
        assert "Perfil estimado de muestra chica" in out
        assert "Kelly completo" in out
        assert "1/2 Kelly" in out
        assert "1/4 Kelly" in out
        assert "Prob. de ruina" in out

    def test_manual_profile_omits_n_and_warning(self, capsys):
        mc_result = _mc_result(n="manual")
        print_report(mc_result)
        out = capsys.readouterr().out
        assert "N=" not in out
        assert "muestra chica" not in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestPrintReport -v`
Expected: FAIL — `ImportError: cannot import name 'print_report'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/bankroll_simulation.py` (add `from src.backtest_analytics import SMALL_SAMPLE_THRESHOLD` to the top imports):

```python
_KELLY_LABELS = {1.0: "Kelly completo", 0.5: "1/2 Kelly", 0.25: "1/4 Kelly"}


def _kelly_label(km: float) -> str:
    return _KELLY_LABELS.get(km, f"{km:g}x Kelly")


def print_report(mc_result: dict) -> None:
    profile = mc_result["profile"]
    results = mc_result["results"]
    kellys = list(results.keys())

    print("=" * 60)
    print(" SIMULACION MONTE CARLO DE BANCA - Modulo 4")
    print(f" Banca inicial: ${mc_result['initial_bankroll']:,.2f} | "
          f"{mc_result['n_simulations']} simulaciones x {mc_result['n_bets']} apuestas")
    print("=" * 60)

    n = profile["n"]
    n_suffix = f" | N={n}" if n != "manual" else ""
    print(f"\n Perfil: edge medio {profile['mean_edge'] * 100:.1f}% "
          f"(sd {profile['std_edge'] * 100:.1f}%) | odds medias "
          f"{profile['mean_odds']:.2f} (sd {profile['std_odds']:.2f}){n_suffix}")

    if n != "manual" and n < SMALL_SAMPLE_THRESHOLD:
        print(f"\n  *** Perfil estimado de muestra chica (N={n} < "
              f"{SMALL_SAMPLE_THRESHOLD}) - usar con cautela. ***")

    labels = [_kelly_label(km) for km in kellys]
    col_width = max(14, max(len(l) for l in labels) + 2)

    header = " " * 28 + "".join(f"{l:>{col_width}}" for l in labels)
    print("\n" + header)

    def _row(title, fmt_fn):
        cells = "".join(f"{fmt_fn(results[km]):>{col_width}}" for km in kellys)
        print(f"  {title:<26}{cells}")

    _row("Banca final P10", lambda r: f"${r['p10']:,.2f}")
    _row("Banca final P50 (mediana)", lambda r: f"${r['p50']:,.2f}")
    _row("Banca final P90", lambda r: f"${r['p90']:,.2f}")
    _row("Prob. de ruina", lambda r: f"{r['ruin_probability'] * 100:.1f}%")
    _row("Drawdown maximo esperado", lambda r: f"{r['median_max_drawdown_pct']:.1f}%")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestPrintReport -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "feat: add print_report to src/bankroll_simulation.py"
```

---

## Task 7: `run_simulation` orchestration + error handling

**Files:**
- Modify: `src/bankroll_simulation.py`
- Modify: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bankroll_simulation.py`:

```python
from src.bankroll_simulation import run_simulation  # add to existing import block

_RESOLVED_BETS_HEADER = "tour,match_date,status,result,kelly_a,kelly_b,odds_a,odds_b,ev_a,ev_b,profit,edge_a,edge_b"
_RESOLVED_BETS_ROWS = [
    "atp,2026-01-01,ok,A_win,0.05,0.0,2.0,1.9,0.1,-0.05,0.05,0.05,-0.08",
    "wta,2026-01-02,ok,B_win,0.0,0.03,1.8,2.1,-0.02,0.08,0.03,-0.04,0.08",
]


def _write_resolved_bets_csv(tmp_path, rows=_RESOLVED_BETS_ROWS):
    path = tmp_path / "value_bets_log.csv"
    path.write_text(_RESOLVED_BETS_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


class TestRunSimulation:
    def test_missing_bets_file_without_overrides(self, tmp_path, capsys):
        run_simulation(bets_path=str(tmp_path / "nope.csv"), n_simulations=10, n_bets=5)
        out = capsys.readouterr().out
        assert "No se encontro" in out

    def test_zero_resolved_bets_without_overrides(self, tmp_path, capsys):
        rows = ["atp,2026-01-05,ok,pending,0.05,0.0,2.0,1.9,0.1,-0.05,,0.05,-0.08"]
        bets_path = _write_resolved_bets_csv(tmp_path, rows)
        run_simulation(bets_path=bets_path, n_simulations=10, n_bets=5)
        out = capsys.readouterr().out
        assert "No hay apuestas resueltas todavia" in out

    def test_partial_overrides_rejected(self, tmp_path, capsys):
        run_simulation(
            bets_path=str(tmp_path / "nope.csv"), n_simulations=10, n_bets=5,
            mean_edge=0.05, std_edge=None, mean_odds=None, std_odds=None,
        )
        out = capsys.readouterr().out
        assert "Debes dar los 4 overrides" in out

    def test_full_overrides_skip_missing_file(self, tmp_path, capsys):
        run_simulation(
            bets_path=str(tmp_path / "nope.csv"), n_simulations=20, n_bets=5, seed=1,
            mean_edge=0.05, std_edge=0.01, mean_odds=2.0, std_odds=0.2,
        )
        out = capsys.readouterr().out
        assert "SIMULACION MONTE CARLO" in out

    def test_normal_run_from_real_bets_csv(self, tmp_path, capsys):
        bets_path = _write_resolved_bets_csv(tmp_path)  # 2 resolved rows
        run_simulation(bets_path=bets_path, n_simulations=20, n_bets=5, seed=3)
        out = capsys.readouterr().out
        assert "SIMULACION MONTE CARLO" in out
        assert "N=2" in out

    def test_rejects_non_positive_n_bets(self, tmp_path):
        bets_path = _write_resolved_bets_csv(tmp_path)
        with pytest.raises(ValueError):
            run_simulation(bets_path=bets_path, n_simulations=20, n_bets=0)

    def test_rejects_non_positive_n_simulations(self, tmp_path):
        bets_path = _write_resolved_bets_csv(tmp_path)
        with pytest.raises(ValueError):
            run_simulation(bets_path=bets_path, n_simulations=0, n_bets=5)

    def test_rejects_ruin_threshold_out_of_range(self, tmp_path):
        bets_path = _write_resolved_bets_csv(tmp_path)
        with pytest.raises(ValueError):
            run_simulation(bets_path=bets_path, n_simulations=20, n_bets=5, ruin_threshold=1.5)
        with pytest.raises(ValueError):
            run_simulation(bets_path=bets_path, n_simulations=20, n_bets=5, ruin_threshold=0.0)

    def test_rejects_non_positive_kelly_multiplier(self, tmp_path):
        bets_path = _write_resolved_bets_csv(tmp_path)
        with pytest.raises(ValueError):
            run_simulation(
                bets_path=bets_path, n_simulations=20, n_bets=5,
                kelly_multipliers=[1.0, 0.0],
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestRunSimulation -v`
Expected: FAIL — `ImportError: cannot import name 'run_simulation'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/bankroll_simulation.py` (add `from pathlib import Path` and
`from src.backtest_analytics import load_resolved_bets` — alongside the
`SMALL_SAMPLE_THRESHOLD` import already added in Task 6 — to the top imports):

```python
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BETS_PATH = str(_ROOT / "data" / "value_bets_log.csv")
DEFAULT_BANKROLL = 1000.0
DEFAULT_N_BETS = 50
DEFAULT_N_SIMULATIONS = 10000
DEFAULT_RUIN_THRESHOLD = 0.5
DEFAULT_KELLY_MULTIPLIERS = [1.0, 0.5, 0.25]


def run_simulation(
    bets_path: str = DEFAULT_BETS_PATH,
    tour: str | None = None,
    bankroll: float = DEFAULT_BANKROLL,
    n_bets: int = DEFAULT_N_BETS,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    kelly_multipliers: list[float] | None = None,
    ruin_threshold: float = DEFAULT_RUIN_THRESHOLD,
    seed: int | None = None,
    mean_edge: float | None = None,
    std_edge: float | None = None,
    mean_odds: float | None = None,
    std_odds: float | None = None,
) -> None:
    if kelly_multipliers is None:
        kelly_multipliers = list(DEFAULT_KELLY_MULTIPLIERS)

    if n_bets <= 0:
        raise ValueError(f"n_bets debe ser > 0, recibido {n_bets}")
    if n_simulations <= 0:
        raise ValueError(f"n_simulations debe ser > 0, recibido {n_simulations}")
    if not (0 < ruin_threshold < 1):
        raise ValueError(f"ruin_threshold debe estar en (0, 1), recibido {ruin_threshold}")
    if any(km <= 0 for km in kelly_multipliers):
        raise ValueError(f"kelly_multipliers deben ser > 0, recibido {kelly_multipliers}")

    overrides = [mean_edge, std_edge, mean_odds, std_odds]
    n_overrides = sum(o is not None for o in overrides)
    if n_overrides not in (0, 4):
        print("Debes dar los 4 overrides (--mean-edge --std-edge --mean-odds "
              "--std-odds) juntos, o ninguno.")
        return

    if n_overrides == 4:
        profile = {
            "mean_edge": mean_edge, "std_edge": std_edge,
            "mean_odds": mean_odds, "std_odds": std_odds, "n": "manual",
        }
    else:
        if not Path(bets_path).exists():
            print(f"No se encontro {bets_path} y no diste overrides manuales "
                  "(--mean-edge/--std-edge/--mean-odds/--std-odds).")
            return
        df = load_resolved_bets(bets_path, tour)
        if df.empty:
            print("No hay apuestas resueltas todavia (status='ok' y result en "
                  "A_win/B_win) para estimar el perfil; usa los overrides "
                  "manuales o registra mas apuestas primero.")
            return
        profile = estimate_bet_profile(df)

    mc_result = run_monte_carlo(
        profile, kelly_multipliers, n_simulations, n_bets, bankroll,
        ruin_threshold, seed,
    )
    print_report(mc_result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestRunSimulation -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "feat: add run_simulation orchestration to src/bankroll_simulation.py"
```

---

## Task 8: CLI `main()`

**Files:**
- Modify: `src/bankroll_simulation.py`
- Modify: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bankroll_simulation.py`:

```python
import sys  # add to existing import block

from src.bankroll_simulation import main  # add to existing import block


class TestMain:
    def test_cli_wires_args_from_bets_file(self, tmp_path, capsys, monkeypatch):
        bets_path = _write_resolved_bets_csv(tmp_path)
        argv = [
            "bankroll_simulation",
            "--bankroll", "500",
            "--bets-file", bets_path,
            "--bets", "5",
            "--simulations", "20",
            "--kelly-fractions", "1.0,0.5",
            "--seed", "9",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        main()
        out = capsys.readouterr().out

        assert "SIMULACION MONTE CARLO" in out
        assert "$500.00" in out
        assert "Kelly completo" in out
        assert "1/2 Kelly" in out
        assert "1/4 Kelly" not in out  # only 2 fractions requested

    def test_cli_with_manual_overrides(self, tmp_path, capsys, monkeypatch):
        argv = [
            "bankroll_simulation",
            "--bets-file", str(tmp_path / "nope.csv"),
            "--mean-edge", "0.05", "--std-edge", "0.01",
            "--mean-odds", "2.0", "--std-odds", "0.2",
            "--bets", "5", "--simulations", "20", "--seed", "1",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        main()
        out = capsys.readouterr().out
        assert "SIMULACION MONTE CARLO" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestMain -v`
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/bankroll_simulation.py` (add `import argparse` to the top imports):

```python
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulacion Monte Carlo de banca bajo Kelly (Modulo 4)"
    )
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL)
    parser.add_argument("--tour", choices=["atp", "wta", "both"], default="both")
    parser.add_argument("--bets-file", default=DEFAULT_BETS_PATH)
    parser.add_argument("--bets", type=int, default=DEFAULT_N_BETS)
    parser.add_argument("--simulations", type=int, default=DEFAULT_N_SIMULATIONS)
    parser.add_argument("--kelly-fractions", default="1.0,0.5,0.25")
    parser.add_argument("--ruin-threshold", type=float, default=DEFAULT_RUIN_THRESHOLD)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mean-edge", type=float, default=None)
    parser.add_argument("--std-edge", type=float, default=None)
    parser.add_argument("--mean-odds", type=float, default=None)
    parser.add_argument("--std-odds", type=float, default=None)
    args = parser.parse_args()

    tour = None if args.tour == "both" else args.tour
    kelly_multipliers = [float(x) for x in args.kelly_fractions.split(",")]

    run_simulation(
        bets_path=args.bets_file,
        tour=tour,
        bankroll=args.bankroll,
        n_bets=args.bets,
        n_simulations=args.simulations,
        kelly_multipliers=kelly_multipliers,
        ruin_threshold=args.ruin_threshold,
        seed=args.seed,
        mean_edge=args.mean_edge,
        std_edge=args.std_edge,
        mean_odds=args.mean_odds,
        std_odds=args.std_odds,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestMain -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "feat: add CLI entry point to src/bankroll_simulation.py"
```

---

## Task 9: Full suite + real-data smoke test

**Files:** none (verification only)

- [ ] **Step 1: Run the full project test suite**

Run: `./tenis-env/Scripts/python.exe -m pytest -q`
Expected: all tests pass, no regressions in any other module

- [ ] **Step 2: Smoke test against the real repo CSV**

Run: `./tenis-env/Scripts/python.exe -m src.bankroll_simulation --simulations 5000 --bets 30 --seed 42`
Expected: prints a full report using the real `data/value_bets_log.csv` (7
resolved rows as of this plan being written — confirm "N=7" and the
small-sample warning appear), with 3 columns (Kelly completo / 1/2 Kelly /
1/4 Kelly). Visually confirm `Prob. de ruina` decreases and `Banca final
P50` moves toward the initial bankroll as the fraction shrinks from full
to 1/4 Kelly — this directional check is a manual eyeball here rather than
a pytest assertion, since asserting strict cross-fraction monotonicity in
an automated test would require sharing random draws across fractions
(see spec's "Fuera de alcance") and risks flakiness otherwise.

- [ ] **Step 3: Smoke test manual-override mode**

Run: `./tenis-env/Scripts/python.exe -m src.bankroll_simulation --mean-edge 0.05 --std-edge 0.02 --mean-odds 2.0 --std-odds 0.3 --simulations 5000 --bets 30 --seed 42`
Expected: report prints without reading `data/value_bets_log.csv` at all,
profile line has no `N=` suffix and no small-sample warning.

- [ ] **Step 4: Fix anything the smoke tests surface, then final commit if needed**

If Steps 2-3 reveal a formatting or logic issue not caught by the unit
tests, fix it in `src/bankroll_simulation.py` and commit:

```bash
git add src/bankroll_simulation.py
git commit -m "fix: address issue found in bankroll_simulation real-data smoke test"
```

If nothing needed fixing, no commit is required for this task.

---

## Task 10: Validate manual overrides / bankroll, and harden `--kelly-fractions` parsing

**Added after the final whole-module review** (Tasks 1-9 were already complete,
tested, and committed): `run_simulation` validates `n_bets`/`n_simulations`/
`ruin_threshold`/`kelly_multipliers` but not the four manual override values
or `bankroll`, so bad CLI input crashes with a raw Python traceback instead
of a clean error — e.g. `--std-edge -0.02` raises `ValueError: scale < 0`
from deep inside `sample_bet`, and `--bankroll -50` silently produces a
nonsensical negative-bankroll report plus a `RuntimeWarning`. Separately,
`--kelly-fractions` is parsed by hand (`[float(x) for x in ...split(",")]`)
with no error handling, so malformed input (`"1.0,abc"`) also raises a raw
traceback instead of argparse's usual clean "invalid value" message.

**Files:**
- Modify: `src/bankroll_simulation.py`
- Modify: `tests/test_bankroll_simulation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bankroll_simulation.py`:

```python
class TestRunSimulationValidatesMoreInputs:
    def test_rejects_non_positive_bankroll(self, tmp_path):
        bets_path = _write_resolved_bets_csv(tmp_path)
        with pytest.raises(ValueError):
            run_simulation(bets_path=bets_path, n_simulations=20, n_bets=5, bankroll=0.0)
        with pytest.raises(ValueError):
            run_simulation(bets_path=bets_path, n_simulations=20, n_bets=5, bankroll=-50.0)

    def test_rejects_manual_mean_odds_not_greater_than_one(self, tmp_path):
        with pytest.raises(ValueError):
            run_simulation(
                bets_path=str(tmp_path / "nope.csv"), n_simulations=20, n_bets=5,
                mean_edge=0.05, std_edge=0.01, mean_odds=0.8, std_odds=0.2,
            )

    def test_rejects_negative_manual_std_edge(self, tmp_path):
        with pytest.raises(ValueError):
            run_simulation(
                bets_path=str(tmp_path / "nope.csv"), n_simulations=20, n_bets=5,
                mean_edge=0.05, std_edge=-0.01, mean_odds=2.0, std_odds=0.2,
            )

    def test_rejects_negative_manual_std_odds(self, tmp_path):
        with pytest.raises(ValueError):
            run_simulation(
                bets_path=str(tmp_path / "nope.csv"), n_simulations=20, n_bets=5,
                mean_edge=0.05, std_edge=0.01, mean_odds=2.0, std_odds=-0.2,
            )


class TestMainRejectsMalformedKellyFractions:
    def test_cli_exits_cleanly_on_malformed_kelly_fractions(self, tmp_path, capsys, monkeypatch):
        bets_path = _write_resolved_bets_csv(tmp_path)
        argv = [
            "bankroll_simulation",
            "--bets-file", bets_path,
            "--kelly-fractions", "1.0,abc",
        ]
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            main()
        err = capsys.readouterr().err
        assert "Traceback" not in err
        assert "kelly-fractions" in err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestRunSimulationValidatesMoreInputs tests/test_bankroll_simulation.py::TestMainRejectsMalformedKellyFractions -v`
Expected: FAIL — the 4 `run_simulation` tests fail because no `ValueError` is
raised for `bankroll`/`mean_odds`/`std_edge`/`std_odds` yet (either no
exception, or an unrelated exception raised from deep inside `sample_bet`
instead of a clean validation `ValueError` in `run_simulation` itself); the
CLI test fails because `main()` raises an unhandled `ValueError` (not
`SystemExit`) on malformed `--kelly-fractions`.

- [ ] **Step 3: Write minimal implementation**

In `src/bankroll_simulation.py`, extend the existing validation block near
the top of `run_simulation` (right after the `n_bets`/`n_simulations`/
`ruin_threshold`/`kelly_multipliers` checks added in Task 7) to also
validate `bankroll`:

```python
    if bankroll <= 0:
        raise ValueError(f"bankroll debe ser > 0, recibido {bankroll}")
```

Then, inside the `if n_overrides == 4:` branch (where the manual profile
dict is built), validate the three override fields that have a meaningful
valid range before building the profile:

```python
    if n_overrides == 4:
        if mean_odds <= 1:
            raise ValueError(f"mean_odds debe ser > 1, recibido {mean_odds}")
        if std_edge < 0:
            raise ValueError(f"std_edge debe ser >= 0, recibido {std_edge}")
        if std_odds < 0:
            raise ValueError(f"std_odds debe ser >= 0, recibido {std_odds}")
        profile = {
            "mean_edge": mean_edge, "std_edge": std_edge,
            "mean_odds": mean_odds, "std_odds": std_odds, "n": "manual",
        }
```

(`mean_edge` is intentionally left unvalidated — `sample_bet`'s existing
0.001 floor already handles any value, including negative ones, safely.)

In `main()`, wrap the `--kelly-fractions` parsing in a try/except that
calls `parser.error(...)` (argparse's standard way to print a clean usage
message and exit with status 2, instead of letting a raw `ValueError`
propagate):

```python
    try:
        kelly_multipliers = [float(x) for x in args.kelly_fractions.split(",")]
    except ValueError:
        parser.error(
            f"--kelly-fractions invalido: '{args.kelly_fractions}' "
            "(debe ser una lista de numeros separados por comas, ej. '1.0,0.5,0.25')"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py::TestRunSimulationValidatesMoreInputs tests/test_bankroll_simulation.py::TestMainRejectsMalformedKellyFractions -v`
Expected: PASS (5 tests)

Then run the full file to confirm no regressions:

Run: `./tenis-env/Scripts/python.exe -m pytest tests/test_bankroll_simulation.py -v`
Expected: PASS (32 tests: 27 pre-existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add src/bankroll_simulation.py tests/test_bankroll_simulation.py
git commit -m "fix: validate manual overrides/bankroll and harden --kelly-fractions parsing"
```

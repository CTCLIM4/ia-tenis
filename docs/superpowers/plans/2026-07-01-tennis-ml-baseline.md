# Tennis ML Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validated tennis match outcome prediction model using surface-specific Elo with temporal decay, additional features (ranking, H2H, form, rest), and walk-forward backtesting on Jeff Sackmann's historical ATP/WTA data.

**Architecture:** Sequential Elo rating system processes historical matches to produce per-player per-surface ratings; a feature builder computes contextual signals alongside Elo; a walk-forward backtest trains logistic regression calibration on lagged years and evaluates on future years — zero lookahead throughout.

**Tech Stack:** Python 3.14, pandas, numpy, scikit-learn, scipy, pytest, Jupyter

---

## File Map

```
D:\ia-tenis\
├── data/
│   ├── raw/
│   │   ├── tennis_atp/         # Jeff Sackmann ATP repo (cloned)
│   │   └── tennis_wta/         # Jeff Sackmann WTA repo (cloned)
│   └── processed/              # CSV outputs of feature pipeline
├── scripts/
│   └── download_data.py        # Clone / pull Jeff Sackmann repos
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # Load & clean raw CSV files
│   ├── models/
│   │   ├── __init__.py
│   │   └── elo.py              # EloSystem: per-surface Elo + decay
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py      # FeatureBuilder: H2H, form, rest days
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── walkforward.py      # build_match_features + walk_forward_backtest
│   └── pipeline.py             # CLI entry point
├── tests/
│   ├── conftest.py             # shared fixtures
│   ├── test_elo.py
│   ├── test_features.py
│   ├── test_loader.py
│   └── test_backtest.py
├── notebooks/
│   └── 01_exploratory.ipynb
├── pyproject.toml              # pytest config (pythonpath = ["."])
├── requirements.txt
└── .gitignore
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/features/__init__.py`, `src/backtest/__init__.py`
- Create: `scripts/` dir, `notebooks/` dir, `data/raw/` dir, `data/processed/` dir

- [ ] **Step 1: Create directory structure**

```powershell
cd D:\ia-tenis
New-Item -ItemType Directory -Force data\raw, data\processed, scripts, notebooks, tests
New-Item -ItemType Directory -Force src\data, src\models, src\features, src\backtest
```

- [ ] **Step 2: Create `requirements.txt`**

```
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
scipy>=1.13.0
pytest>=8.0.0
jupyter>=1.0.0
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 4: Create `.gitignore`**

```
tenis-env/
data/raw/
data/processed/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 5: Create all `__init__.py` files**

Create empty `__init__.py` in: `src/`, `src/data/`, `src/models/`, `src/features/`, `src/backtest/`, `tests/`

```powershell
@("src", "src\data", "src\models", "src\features", "src\backtest", "tests") | ForEach-Object {
    New-Item -ItemType File -Path "$_\__init__.py" -Force
}
```

- [ ] **Step 6: Install dependencies**

```powershell
D:\ia-tenis\tenis-env\Scripts\pip.exe install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 7: Init git and commit**

```powershell
git init
git add requirements.txt pyproject.toml .gitignore src tests scripts notebooks
git commit -m "chore: project scaffolding — dirs, requirements, pytest config"
```

---

## Task 2: Data Download Script

**Files:**
- Create: `scripts/download_data.py`

- [ ] **Step 1: Write `scripts/download_data.py`**

```python
"""Clone or update Jeff Sackmann tennis datasets into data/raw/."""
import subprocess
import sys
from pathlib import Path

DATA_RAW = Path("data/raw")

REPOS = {
    "tennis_atp": "https://github.com/JeffSackmann/tennis_atp.git",
    "tennis_wta": "https://github.com/JeffSackmann/tennis_wta.git",
}


def download():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        dest = DATA_RAW / name
        if dest.exists():
            print(f"{name}: already exists, pulling latest...")
            subprocess.run(["git", "-C", str(dest), "pull"], check=True)
        else:
            print(f"Cloning {name} (shallow)...")
            subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)
    print("Done.")


if __name__ == "__main__":
    download()
```

- [ ] **Step 2: Run the download script**

```powershell
cd D:\ia-tenis
D:\ia-tenis\tenis-env\Scripts\python.exe scripts/download_data.py
```

Expected output:
```
Cloning tennis_atp (shallow)...
Cloning tennis_wta (shallow)...
Done.
```

After cloning, verify:
```powershell
Get-ChildItem data\raw\tennis_atp\*.csv | Select-Object -First 5
```
Expected: files like `atp_matches_1968.csv`, `atp_matches_2023.csv`, etc.

- [ ] **Step 3: Commit**

```powershell
git add scripts/download_data.py
git commit -m "feat: add Jeff Sackmann dataset download script"
```

---

## Task 3: Data Loader (TDD)

**Files:**
- Create: `src/data/loader.py`
- Create: `tests/test_loader.py`

Jeff Sackmann CSV column reference:
- `tourney_date`: integer YYYYMMDD (e.g. 20230116)
- `surface`: 'Clay', 'Hard', 'Grass', 'Carpet'
- `winner_name`, `loser_name`: player names
- `winner_rank`, `loser_rank`: ATP/WTA rank (can be NaN)

- [ ] **Step 1: Write failing tests in `tests/test_loader.py`**

```python
import pandas as pd
import pytest
from io import StringIO
from src.data.loader import _clean

SAMPLE_CSV = (
    "tourney_id,tourney_name,surface,draw_size,tourney_level,tourney_date,"
    "match_num,winner_id,winner_seed,winner_entry,winner_name,winner_hand,"
    "winner_ht,winner_ioc,winner_age,winner_rank,winner_rank_points,"
    "loser_id,loser_seed,loser_entry,loser_name,loser_hand,loser_ht,"
    "loser_ioc,loser_age,loser_rank,loser_rank_points,score,best_of,round,minutes\n"
    "2023-1,AO,Hard,128,G,20230117,1,101,,,Djokovic,R,188,SRB,35.7,1,10000,"
    "102,,,Murray,R,190,GBR,35.5,5,500,6-3 6-4,5,R32,85\n"
    "2023-1,AO,Clay,128,G,20230116,2,103,,,Nadal,R,185,ESP,36.7,2,8000,"
    "104,,,Federer,R,185,SUI,41.7,3,7000,7-5 6-3,5,R32,92\n"
)


def test_clean_parses_dates():
    df = _clean(pd.read_csv(StringIO(SAMPLE_CSV)))
    assert str(df["match_date"].dtype) == "datetime64[ns]"
    assert df.iloc[0]["match_date"].year == 2023


def test_clean_normalizes_surface_to_lowercase():
    df = _clean(pd.read_csv(StringIO(SAMPLE_CSV)))
    assert set(df["surface"].unique()).issubset({"clay", "hard", "grass", "carpet", "unknown"})


def test_clean_sorts_by_date_ascending():
    df = _clean(pd.read_csv(StringIO(SAMPLE_CSV)))
    dates = df["match_date"].tolist()
    assert dates == sorted(dates)


def test_clean_drops_rows_with_missing_player_names():
    raw = pd.read_csv(StringIO(SAMPLE_CSV))
    raw.loc[0, "winner_name"] = None
    df = _clean(raw)
    assert len(df) == 1


def test_clean_coerces_rank_to_float_and_allows_nan():
    raw = pd.read_csv(StringIO(SAMPLE_CSV))
    raw["winner_rank"] = "N/A"
    df = _clean(raw)
    assert df["winner_rank"].isna().all()


def test_clean_adds_tour_column_when_provided():
    raw = pd.read_csv(StringIO(SAMPLE_CSV))
    raw["tour"] = "atp"
    df = _clean(raw)
    assert "tour" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe tests/test_loader.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `src.data.loader` does not exist yet.

- [ ] **Step 3: Write `src/data/loader.py`**

```python
from pathlib import Path
from typing import Optional
import pandas as pd

RAW_DATA_DIR = Path("data/raw")

_SURFACE_MAP = {
    "Clay": "clay",
    "Hard": "hard",
    "Grass": "grass",
    "Carpet": "carpet",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["match_date"] = pd.to_datetime(
        df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["surface"] = df["surface"].map(_SURFACE_MAP).fillna("unknown")
    df = df.dropna(subset=["winner_name", "loser_name", "match_date"])
    for col in ("winner_rank", "loser_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")
    return df.sort_values("match_date").reset_index(drop=True)


def _load_tour(tour: str, start_year: int, end_year: int) -> pd.DataFrame:
    prefix = "atp" if tour == "atp" else "wta"
    tour_dir = RAW_DATA_DIR / f"tennis_{tour}"
    frames = []
    for year in range(start_year, end_year + 1):
        path = tour_dir / f"{prefix}_matches_{year}.csv"
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            df["year"] = year
            df["tour"] = tour
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No {tour.upper()} match files found in {tour_dir} "
            f"for years {start_year}-{end_year}. Run scripts/download_data.py first."
        )
    return _clean(pd.concat(frames, ignore_index=True))


def load_atp_matches(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    return _load_tour("atp", start_year, end_year)


def load_wta_matches(start_year: int = 1990, end_year: int = 2024) -> pd.DataFrame:
    return _load_tour("wta", start_year, end_year)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe tests/test_loader.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/data/loader.py tests/test_loader.py
git commit -m "feat: data loader with surface normalization and date parsing"
```

---

## Task 4: Elo Rating System (TDD)

**Files:**
- Create: `src/models/elo.py`
- Create: `tests/test_elo.py`

**Algorithm (FiveThirtyEight-inspired):**
- Each player has a `general_rating` (updates on every match) and `surface_rating[surface]` (updates on surface matches only).
- Effective rating for prediction: `w * surface + (1-w) * general`, where `w = n_surface / (n_surface + 250)` grows with surface experience.
- K-factor: 32 for < 30 matches, 24 for 30–99, 16 for 100+.
- Yearly decay at first match of new calendar year: rating regressed 25% towards 1500 (`0.75 * r + 0.25 * 1500`).

- [ ] **Step 1: Write failing tests in `tests/test_elo.py`**

```python
import pytest
from datetime import date
from src.models.elo import EloSystem


def test_new_player_starts_at_initial_rating():
    elo = EloSystem()
    assert elo.get_general_rating("Unknown Player") == 1500.0


def test_winner_gains_rating_loser_loses():
    elo = EloSystem()
    elo.update("A", "B", "hard", date(2023, 1, 1))
    assert elo.get_general_rating("A") > 1500.0
    assert elo.get_general_rating("B") < 1500.0


def test_ratings_are_conserved_equal_players():
    elo = EloSystem()
    elo.update("A", "B", "hard", date(2023, 1, 1))
    total = elo.get_general_rating("A") + elo.get_general_rating("B")
    assert abs(total - 3000.0) < 1e-9


def test_upset_win_gives_larger_elo_gain():
    elo = EloSystem()
    d = date(2022, 6, 1)
    for _ in range(8):
        elo.update("Strong", "Weak", "hard", d)

    weak_before = elo.get_general_rating("Weak")
    # Weak player (underdog) wins — should gain > half of K
    prob_weak_wins = elo.update("Weak", "Strong", "hard", date(2022, 7, 1))
    gain = elo.get_general_rating("Weak") - weak_before

    assert prob_weak_wins < 0.5          # Elo says strong player was favourite
    assert gain > elo.k * 0.5           # upset → bigger gain than 50-50


def test_surface_ratings_independent():
    elo = EloSystem()
    elo.update("A", "B", "clay", date(2023, 1, 1))
    assert elo.get_surface_rating("A", "hard") == 1500.0  # hard untouched
    assert elo.get_surface_rating("A", "clay") != 1500.0  # clay changed


def test_blend_weight_increases_with_surface_experience():
    elo = EloSystem()
    w0 = elo.get_blend_weight("A", "clay")
    elo.update("A", "B", "clay", date(2023, 1, 1))
    w1 = elo.get_blend_weight("A", "clay")
    assert w1 > w0
    assert w0 == 0.0   # zero experience at start


def test_yearly_decay_regresses_towards_1500():
    elo = EloSystem()
    for i in range(20):
        elo.update("A", "B", "hard", date(2022, (i % 11) + 1, 1))
    rating_end_2022 = elo.get_general_rating("A")
    assert rating_end_2022 > 1500.0

    # First match in 2023 triggers decay
    elo.update("A", "B", "hard", date(2023, 1, 15))
    rating_after_decay = elo.get_general_rating("A")

    # Decayed but still above 1500 (decay only regresses partially)
    assert 1500.0 < rating_after_decay < rating_end_2022


def test_k_factor_decreases_with_experience():
    elo = EloSystem()
    k_new = elo.get_k("new_player")   # 0 matches
    elo.match_counts["veteran"] = 150
    k_veteran = elo.get_k("veteran")
    assert k_veteran < k_new


def test_expected_score_sums_to_one():
    elo = EloSystem()
    p_ab = elo.expected_score(1600.0, 1400.0)
    p_ba = elo.expected_score(1400.0, 1600.0)
    assert abs(p_ab + p_ba - 1.0) < 1e-10
    assert p_ab > 0.5


def test_update_returns_win_probability_for_winner():
    elo = EloSystem()
    prob = elo.update("A", "B", "hard", date(2023, 1, 1))
    assert 0.0 < prob < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe tests/test_elo.py -v
```

Expected: `ModuleNotFoundError` for `src.models.elo`.

- [ ] **Step 3: Write `src/models/elo.py`**

```python
from datetime import date
from typing import Dict, Optional, Tuple


class EloSystem:
    def __init__(
        self,
        k: float = 32.0,
        initial_rating: float = 1500.0,
        decay_factor: float = 0.75,
        surface_blend_cap: int = 250,
    ):
        self.k = k
        self.initial_rating = initial_rating
        self.decay_factor = decay_factor          # fraction of rating retained at year boundary
        self.surface_blend_cap = surface_blend_cap  # w = n / (n + cap)

        self.general_ratings: Dict[str, float] = {}
        self.surface_ratings: Dict[Tuple[str, str], float] = {}
        self.match_counts: Dict[str, int] = {}
        self.surface_match_counts: Dict[Tuple[str, str], int] = {}
        self.current_year: Dict[str, Optional[int]] = {}

    # ── accessors ──────────────────────────────────────────────────────────────

    def get_general_rating(self, player: str) -> float:
        return self.general_ratings.get(player, self.initial_rating)

    def get_surface_rating(self, player: str, surface: str) -> float:
        return self.surface_ratings.get((player, surface), self.initial_rating)

    def get_blend_weight(self, player: str, surface: str) -> float:
        n = self.surface_match_counts.get((player, surface), 0)
        return n / (n + self.surface_blend_cap)

    def get_effective_rating(self, player: str, surface: str) -> float:
        w = self.get_blend_weight(player, surface)
        return w * self.get_surface_rating(player, surface) + (1 - w) * self.get_general_rating(player)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def get_k(self, player: str) -> float:
        n = self.match_counts.get(player, 0)
        if n < 30:
            return self.k
        if n < 100:
            return self.k * 0.75
        return self.k * 0.5

    # ── internal ───────────────────────────────────────────────────────────────

    def _apply_yearly_decay(self, player: str, match_year: int) -> None:
        prev = self.current_year.get(player)
        if prev is not None and match_year > prev:
            r = self.general_ratings.get(player, self.initial_rating)
            self.general_ratings[player] = (
                self.decay_factor * r + (1 - self.decay_factor) * self.initial_rating
            )
            for key in list(self.surface_ratings):
                if key[0] == player:
                    sr = self.surface_ratings[key]
                    self.surface_ratings[key] = (
                        self.decay_factor * sr + (1 - self.decay_factor) * self.initial_rating
                    )
        self.current_year[player] = match_year

    # ── public update ──────────────────────────────────────────────────────────

    def update(self, winner: str, loser: str, surface: str, match_date: date) -> float:
        """Process one match. Returns Elo win-probability for the winner (pre-update)."""
        self._apply_yearly_decay(winner, match_date.year)
        self._apply_yearly_decay(loser, match_date.year)

        # Pre-match prediction
        ra = self.get_effective_rating(winner, surface)
        rb = self.get_effective_rating(loser, surface)
        prob_winner = self.expected_score(ra, rb)

        k_w = self.get_k(winner)
        k_l = self.get_k(loser)

        # Update general ratings
        ga = self.get_general_rating(winner)
        gb = self.get_general_rating(loser)
        ea = self.expected_score(ga, gb)
        self.general_ratings[winner] = ga + k_w * (1.0 - ea)
        self.general_ratings[loser] = gb + k_l * (ea - 1.0)   # = k_l * (0 - (1-ea))

        # Update surface ratings
        sa = self.get_surface_rating(winner, surface)
        sb = self.get_surface_rating(loser, surface)
        ea_s = self.expected_score(sa, sb)
        self.surface_ratings[(winner, surface)] = sa + k_w * (1.0 - ea_s)
        self.surface_ratings[(loser, surface)] = sb + k_l * (ea_s - 1.0)

        # Update counters
        self.match_counts[winner] = self.match_counts.get(winner, 0) + 1
        self.match_counts[loser] = self.match_counts.get(loser, 0) + 1
        self.surface_match_counts[(winner, surface)] = (
            self.surface_match_counts.get((winner, surface), 0) + 1
        )
        self.surface_match_counts[(loser, surface)] = (
            self.surface_match_counts.get((loser, surface), 0) + 1
        )

        return prob_winner
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe tests/test_elo.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/models/elo.py tests/test_elo.py
git commit -m "feat: surface-specific Elo with temporal decay and adaptive K-factor"
```

---

## Task 5: Feature Engineering (TDD)

**Files:**
- Create: `src/features/engineering.py`
- Create: `tests/test_features.py`

`FeatureBuilder` tracks H2H wins, recent match history per player, and last match date. All state is updated AFTER getting features for a match (no lookahead).

- [ ] **Step 1: Write failing tests in `tests/test_features.py`**

```python
import pytest
from datetime import date
from src.features.engineering import FeatureBuilder


def test_new_player_returns_defaults():
    fb = FeatureBuilder()
    f = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    assert f["recent_win_rate"] == 0.5
    assert f["h2h_win_rate"] == 0.5
    assert f["h2h_matches"] == 0
    assert f["rest_days"] == 14  # default when no history


def test_recent_win_rate_after_all_wins():
    fb = FeatureBuilder()
    for _ in range(10):
        fb.update("A", "B", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 1.0


def test_recent_win_rate_after_all_losses():
    fb = FeatureBuilder()
    for _ in range(10):
        fb.update("B", "A", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 0.0


def test_recent_win_rate_uses_only_last_n():
    fb = FeatureBuilder(recent_n=5)
    # 10 losses then 5 wins — should see only the 5 wins
    for _ in range(10):
        fb.update("B", "A", "hard", date(2023, 1, 1))
    for _ in range(5):
        fb.update("A", "B", "hard", date(2023, 2, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    assert f["recent_win_rate"] == 1.0


def test_h2h_tracks_wins_per_direction():
    fb = FeatureBuilder()
    fb.update("A", "B", "hard", date(2023, 1, 1))
    fb.update("A", "B", "hard", date(2023, 2, 1))
    fb.update("B", "A", "hard", date(2023, 3, 1))

    fa = fb.get_features("A", "B", "hard", date(2023, 6, 1))
    fb_ = fb.get_features("B", "A", "hard", date(2023, 6, 1))

    assert fa["h2h_matches"] == 3
    assert abs(fa["h2h_win_rate"] - 2 / 3) < 1e-10
    assert abs(fb_["h2h_win_rate"] - 1 / 3) < 1e-10


def test_rest_days_calculated_correctly():
    fb = FeatureBuilder()
    fb.update("A", "B", "hard", date(2023, 1, 1))
    f = fb.get_features("A", "B", "hard", date(2023, 1, 8))
    assert f["rest_days"] == 7


def test_surface_form_is_surface_specific():
    fb = FeatureBuilder(surface_n=5)
    fb.update("A", "B", "clay", date(2023, 1, 1))
    fb.update("A", "B", "clay", date(2023, 1, 2))
    fb.update("B", "A", "hard", date(2023, 1, 3))
    fb.update("B", "A", "hard", date(2023, 1, 4))

    f_clay = fb.get_features("A", "B", "clay", date(2023, 6, 1))
    f_hard = fb.get_features("A", "B", "hard", date(2023, 6, 1))

    assert f_clay["recent_win_rate_surface"] == 1.0
    assert f_hard["recent_win_rate_surface"] == 0.0


def test_update_does_not_affect_features_for_current_match():
    """Features must use only pre-match information."""
    fb = FeatureBuilder()
    f_before = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    fb.update("A", "B", "hard", date(2023, 1, 1))
    f_after = fb.get_features("A", "B", "hard", date(2023, 1, 1))
    # h2h before update should be 0, after should be 1
    assert f_before["h2h_matches"] == 0
    assert f_after["h2h_matches"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe tests/test_features.py -v
```

Expected: `ModuleNotFoundError` for `src.features.engineering`.

- [ ] **Step 3: Write `src/features/engineering.py`**

```python
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Tuple


class FeatureBuilder:
    def __init__(self, recent_n: int = 10, surface_n: int = 10):
        self.recent_n = recent_n
        self.surface_n = surface_n
        # player -> [(match_date, surface, won)]
        self._history: Dict[str, List[Tuple[date, str, bool]]] = defaultdict(list)
        # (winner, loser) -> count of times winner beat loser
        self._h2h_wins: Dict[Tuple[str, str], int] = defaultdict(int)
        self._last_match_date: Dict[str, Optional[date]] = {}

    def _win_rate(self, results: List[bool]) -> float:
        if not results:
            return 0.5
        return sum(results) / len(results)

    def get_features(self, player: str, opponent: str, surface: str, match_date: date) -> dict:
        history = self._history[player]

        recent = [won for _, _, won in history[-self.recent_n :]]
        recent_surface = [won for _, s, won in history if s == surface][-self.surface_n :]

        p_wins = self._h2h_wins[(player, opponent)]
        o_wins = self._h2h_wins[(opponent, player)]
        total_h2h = p_wins + o_wins
        h2h_rate = p_wins / total_h2h if total_h2h > 0 else 0.5

        last = self._last_match_date.get(player)
        rest_days = (match_date - last).days if last is not None else 14

        return {
            "recent_win_rate": self._win_rate(recent),
            "recent_win_rate_surface": self._win_rate(recent_surface) if recent_surface else 0.5,
            "h2h_win_rate": h2h_rate,
            "h2h_matches": total_h2h,
            "rest_days": float(rest_days),
        }

    def update(self, winner: str, loser: str, surface: str, match_date: date) -> None:
        self._history[winner].append((match_date, surface, True))
        self._history[loser].append((match_date, surface, False))
        self._h2h_wins[(winner, loser)] += 1
        self._last_match_date[winner] = match_date
        self._last_match_date[loser] = match_date
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe tests/test_features.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/features/engineering.py tests/test_features.py
git commit -m "feat: feature builder for H2H, recent form, surface form, rest days"
```

---

## Task 6: Walk-Forward Backtest (TDD)

**Files:**
- Create: `src/backtest/walkforward.py`
- Create: `tests/test_backtest.py`

**Design note on balanced training:** Every match in Jeff Sackmann's dataset has `outcome=1` for the winner. To train logistic regression correctly, we mirror each match row (swapping player1/player2 with negated difference features and `outcome=0`). Mirror rows are included in training but **excluded from evaluation** (flag `is_mirror=True`).

Feature columns (all differences: winner − loser):
- `elo_diff`: effective Elo difference
- `elo_prob`: Elo-predicted win probability for winner
- `rank_diff`: `loser_rank − winner_rank` (positive = winner is better-ranked)
- `form_diff`: recent win-rate difference
- `surface_form_diff`: surface-specific form difference
- `h2h_rate`: winner's H2H win rate vs this loser
- `rest_diff`: winner's rest days − loser's rest days

- [ ] **Step 1: Write failing tests in `tests/test_backtest.py`**

```python
import numpy as np
import pandas as pd
import pytest
from src.backtest.walkforward import walk_forward_backtest


def _synthetic_features(n: int = 2000, n_years: int = 12, start_year: int = 2010) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    years = rng.integers(start_year, start_year + n_years, n)
    elo_diff = rng.normal(50, 150, n)
    elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))
    return pd.DataFrame(
        {
            "year": years,
            "elo_diff": elo_diff,
            "elo_prob": elo_prob,
            "rank_diff": rng.normal(20, 80, n),
            "form_diff": rng.uniform(-0.5, 0.5, n),
            "surface_form_diff": rng.uniform(-0.5, 0.5, n),
            "h2h_rate": rng.uniform(0.3, 0.7, n),
            "rest_diff": rng.normal(0, 5, n),
            "outcome": 1,
            "is_mirror": False,
        }
    )


def test_returns_metrics_dict_with_one_entry_per_test_year():
    df = _synthetic_features()
    results = walk_forward_backtest(df, warmup_years=5)
    assert len(results) > 0
    all_years = sorted(df["year"].unique())
    expected_test_years = all_years[5:]
    assert set(results.keys()) == set(expected_test_years)


def test_each_year_has_required_metrics():
    df = _synthetic_features()
    results = walk_forward_backtest(df, warmup_years=5)
    required = {"accuracy", "log_loss", "brier_score", "n_matches", "elo_only_accuracy", "elo_only_log_loss"}
    for year, m in results.items():
        assert required.issubset(m.keys()), f"Year {year} missing keys: {required - m.keys()}"


def test_metrics_are_in_valid_ranges():
    df = _synthetic_features()
    results = walk_forward_backtest(df, warmup_years=5)
    for year, m in results.items():
        assert 0.0 <= m["accuracy"] <= 1.0, f"Year {year}: accuracy={m['accuracy']}"
        assert m["log_loss"] >= 0.0, f"Year {year}: log_loss={m['log_loss']}"
        assert 0.0 <= m["brier_score"] <= 1.0, f"Year {year}: brier={m['brier_score']}"
        assert 0.0 <= m["elo_only_accuracy"] <= 1.0


def test_more_warmup_years_leaves_fewer_test_years():
    df = _synthetic_features()
    r3 = walk_forward_backtest(df, warmup_years=3)
    r7 = walk_forward_backtest(df, warmup_years=7)
    assert len(r3) >= len(r7)


def test_n_matches_correct_per_year():
    df = _synthetic_features()
    results = walk_forward_backtest(df, warmup_years=5)
    for year, m in results.items():
        expected = int((df["year"] == year).sum())
        assert m["n_matches"] == expected
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe tests/test_backtest.py -v
```

Expected: `ModuleNotFoundError` for `src.backtest.walkforward`.

- [ ] **Step 3: Write `src/backtest/walkforward.py`**

```python
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

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
]


def build_match_features(
    df: pd.DataFrame,
    elo_system: EloSystem,
    feature_builder: FeatureBuilder,
) -> pd.DataFrame:
    """Process all matches sequentially. Returns feature dataframe with mirror rows."""
    records = []
    for _, row in df.iterrows():
        winner = row["winner_name"]
        loser = row["loser_name"]
        surface = row["surface"]
        match_date = row["match_date"].date()
        w_rank = row.get("winner_rank", np.nan)
        l_rank = row.get("loser_rank", np.nan)

        # Pre-match Elo
        elo_w = elo_system.get_effective_rating(winner, surface)
        elo_l = elo_system.get_effective_rating(loser, surface)
        elo_prob = elo_system.expected_score(elo_w, elo_l)

        # Pre-match contextual features
        wf = feature_builder.get_features(winner, loser, surface, match_date)
        lf = feature_builder.get_features(loser, winner, surface, match_date)

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
                "outcome": 1,
                "is_mirror": False,
            }
        )

        # Post-match state update (no lookahead)
        elo_system.update(winner, loser, surface, match_date)
        feature_builder.update(winner, loser, surface, match_date)

    original = pd.DataFrame(records)

    # Mirror rows: swap perspectives so loser is player1 → outcome=0
    mirror = original.copy()
    for col in ("elo_diff", "rank_diff", "form_diff", "surface_form_diff", "rest_diff"):
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

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        elo_probs = test_df["elo_prob"].clip(1e-6, 1 - 1e-6).values

        results[test_year] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "log_loss": float(log_loss(y_test, probs)),
            "brier_score": float(brier_score_loss(y_test, probs)),
            "n_matches": int(len(test_df)),
            "elo_only_accuracy": float(accuracy_score(y_test, (elo_probs >= 0.5).astype(int))),
            "elo_only_log_loss": float(log_loss(y_test, elo_probs)),
        }

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe tests/test_backtest.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run the full test suite**

```powershell
D:\ia-tenis\tenis-env\Scripts\pytest.exe -v
```

Expected: all tests in test_elo.py, test_features.py, test_loader.py, test_backtest.py PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/backtest/walkforward.py tests/test_backtest.py
git commit -m "feat: walk-forward backtest with LR calibration and Elo baseline comparison"
```

---

## Task 7: Pipeline Integration

**Files:**
- Create: `src/pipeline.py`

Ties data loading → Elo + feature building → feature CSV export → walk-forward backtest → printed results table.

- [ ] **Step 1: Write `src/pipeline.py`**

```python
"""End-to-end pipeline: load data → features → backtest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.walkforward import build_match_features, walk_forward_backtest
from src.data.loader import load_atp_matches, load_wta_matches
from src.features.engineering import FeatureBuilder
from src.models.elo import EloSystem


def run_pipeline(
    tour: str = "atp",
    start_year: int = 1990,
    end_year: int = 2023,
    warmup_years: int = 10,
) -> None:
    print(f"Loading {tour.upper()} matches {start_year}–{end_year}…")
    loader = load_atp_matches if tour == "atp" else load_wta_matches
    df = loader(start_year, end_year)
    print(f"  Loaded {len(df):,} matches across {df['year'].nunique()} seasons")

    elo = EloSystem()
    fb = FeatureBuilder()

    print("Building features (sequential, no lookahead)…")
    match_df = build_match_features(df, elo, fb)
    original = match_df[~match_df["is_mirror"]]
    print(f"  {len(original):,} feature rows ({len(match_df):,} with mirrors)")

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{tour}_features.csv"
    original.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")

    print(f"\nWalk-forward backtest (warmup={warmup_years} years)…")
    results = walk_forward_backtest(match_df, warmup_years=warmup_years)

    header = f"{'Year':>6} | {'Acc':>7} | {'LogLoss':>8} | {'Brier':>7} | {'EloAcc':>7} | {'EloLL':>8} | {'N':>6}"
    print(f"\n{'=' * len(header)}")
    print("  BACKTEST RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for year, m in sorted(results.items()):
        print(
            f"{year:>6} | {m['accuracy']:>7.4f} | {m['log_loss']:>8.4f} | "
            f"{m['brier_score']:>7.4f} | {m['elo_only_accuracy']:>7.4f} | "
            f"{m['elo_only_log_loss']:>8.4f} | {m['n_matches']:>6}"
        )
    print("=" * len(header))

    accs = [m["accuracy"] for m in results.values()]
    elo_accs = [m["elo_only_accuracy"] for m in results.values()]
    lls = [m["log_loss"] for m in results.values()]
    elo_lls = [m["elo_only_log_loss"] for m in results.values()]
    print(f"\n  Mean Accuracy : {np.mean(accs):.4f}  (Elo-only: {np.mean(elo_accs):.4f})")
    print(f"  Mean Log-Loss : {np.mean(lls):.4f}  (Elo-only: {np.mean(elo_lls):.4f})")
    print(f"  Years tested  : {len(results)}")


if __name__ == "__main__":
    tour = sys.argv[1] if len(sys.argv) > 1 else "atp"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1990
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 2023
    run_pipeline(tour=tour, start_year=start, end_year=end)
```

- [ ] **Step 2: Run the ATP pipeline end-to-end**

```powershell
cd D:\ia-tenis
D:\ia-tenis\tenis-env\Scripts\python.exe -m src.pipeline atp 1990 2023
```

Expected output (approximate numbers):
```
Loading ATP matches 1990–2023…
  Loaded ~160,000 matches across 34 seasons
Building features (sequential, no lookahead)…
  ~160,000 feature rows (~320,000 with mirrors)
  Saved to data/processed/atp_features.csv
Walk-forward backtest (warmup=10 years)…
====================================================
  BACKTEST RESULTS
====================================================
  Year |     Acc | LogLoss |   Brier |  EloAcc |    EloLL |      N
----------------------------------------------------
  2000 |  0.6600 |  0.6400 |  0.2200 |  0.6400 |   0.6600 |   4500
  ...
  2023 |  0.6700 |  0.6100 |  0.2150 |  0.6500 |   0.6500 |   5000
====================================================
  Mean Accuracy : 0.665  (Elo-only: 0.648)
  Mean Log-Loss : 0.625  (Elo-only: 0.640)
  Years tested  : 24
```
Realistic tennis prediction accuracy is ~64–67% for strong baselines.

- [ ] **Step 3: Run the WTA pipeline**

```powershell
D:\ia-tenis\tenis-env\Scripts\python.exe -m src.pipeline wta 1990 2023
```

- [ ] **Step 4: Commit final state**

```powershell
git add src/pipeline.py data/processed/.gitkeep
git commit -m "feat: end-to-end pipeline — Elo features → walk-forward backtest with printed metrics"
```

---

## Self-Review

### Spec Coverage

| Requirement | Covered by |
|---|---|
| Folder structure: data/raw, data/processed, src/models, src/features, src/backtest, notebooks/ | Task 1 |
| Download Jeff Sackmann ATP + WTA from GitHub | Task 2 |
| requirements.txt with pandas, numpy, scikit-learn, scipy | Task 1 |
| Elo per surface (clay/hard/grass) + general Elo | Task 4 `EloSystem` |
| Temporal decay (FiveThirtyEight methodology) | Task 4 — yearly decay to mean |
| ATP/WTA ranking features | Task 6 — `rank_diff` feature |
| H2H feature | Task 5 `FeatureBuilder._h2h_wins` |
| Recent form (last N matches) | Task 5 — `recent_win_rate`, `recent_win_rate_surface` |
| Rest days between matches | Task 5 — `rest_days` |
| Walk-forward backtest | Task 6 `walk_forward_backtest` |
| Accuracy, log-loss, Brier score metrics | Task 6 |
| No API connection | Not applicable — not included |

### Placeholder Scan

No TBDs or "implement later" found — all steps include full code.

### Type Consistency

- `EloSystem.update` returns `float` — matches usage in `build_match_features` (assigned but not used directly; checked when mirroring `elo_prob`).
- `FeatureBuilder.get_features` returns `dict` with keys `recent_win_rate`, `recent_win_rate_surface`, `h2h_win_rate`, `h2h_matches`, `rest_days` — all referenced by exact name in `build_match_features`.
- `_FEATURE_COLS` in `walkforward.py` matches columns built in `build_match_features`.

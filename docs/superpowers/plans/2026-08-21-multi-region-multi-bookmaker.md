# Multi-Region / Multi-Bookmaker Odds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `regions=eu` + fixed-bookmaker (`pinnacle`) odds pipeline with a configurable multi-region, multi-bookmaker one that selects the best price per side independently, while keeping the old fixed-bookmaker behavior available as an explicit override.

**Architecture:** A single shared helper `_best_price(event, allowed_bookmakers)` in `src/odds_api.py` replaces two duplicated bookmaker-loops (`find_match_odds`, `daily_scanner._extract_h2h_odds`). Two new env vars (`ODDS_API_REGIONS`, `ALLOWED_BOOKMAKERS`) plus a `resolve_allowed_bookmakers()` helper control which bookmakers are eligible; the existing `ODDS_API_BOOKMAKER`/`--bookmaker` settings become an override that collapses the allow-list to one book. The winning bookmaker per side (`bookmaker_a`/`bookmaker_b`) is threaded through every downstream consumer: `MatchOdds`, `DiscoveredMatch`, `value_bets_log.csv`, `prediction_audit_log.csv`, the email notifier, and the settlement report.

**Tech Stack:** Python 3.14, pytest, dataclasses, csv/DictWriter, python-dotenv (via `src/config.py`).

Reference spec: `docs/superpowers/specs/2026-08-21-multi-region-multi-bookmaker-design.md`

---

## Task 1: `.env.example` config additions

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add the two new variables**

Add after the `ODDS_API_KEY=tu_api_key_aqui` line (line 1):

```
ODDS_API_KEY=tu_api_key_aqui

# Regiones de bookmakers a consultar en The Odds API (coma-separado).
# Cada region adicional aumenta el consumo de cuota del plan (regions x markets
# por request) - ver docs/superpowers/specs/2026-08-21-multi-region-multi-bookmaker-design.md.
ODDS_API_REGIONS=eu,uk,us

# Bookmakers permitidos para seleccion de mejor precio (coma-separado, ej.
# "pinnacle,bet365,williamhill"). Vacio o ausente = permitir todos los que
# devuelva la API para las regiones configuradas.
ALLOWED_BOOKMAKERS=
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "$(cat <<'EOF'
docs(config): add ODDS_API_REGIONS/ALLOWED_BOOKMAKERS to .env.example

EOF
)"
```

---

## Task 2: `_best_price` helper in `src/odds_api.py`

**Files:**
- Modify: `src/odds_api.py`
- Test: `tests/test_odds_api.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_odds_api.py` (after the `_event` helper defined around line 64, before `class TestFindMatchOdds`):

```python
def _multi_bk_event(home, away, prices_by_bookmaker):
    """prices_by_bookmaker: {bookmaker_key: {home: price, away: price}}"""
    return {
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": bk,
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": home, "price": prices[home]},
                        {"name": away, "price": prices[away]},
                    ],
                }],
            }
            for bk, prices in prices_by_bookmaker.items()
        ],
    }


class TestBestPrice:
    def test_picks_max_price_per_side_independently(self):
        from src.odds_api import _best_price
        event = _multi_bk_event("Djokovic", "Sinner", {
            "pinnacle": {"Djokovic": 1.50, "Sinner": 2.60},
            "bet365":   {"Djokovic": 1.60, "Sinner": 2.50},
        })
        odds_home, odds_away, bk_home, bk_away = _best_price(event, allowed_bookmakers=None)
        assert odds_home == 1.60 and bk_home == "bet365"
        assert odds_away == 2.60 and bk_away == "pinnacle"

    def test_allowed_bookmakers_filters_out_others(self):
        from src.odds_api import _best_price
        event = _multi_bk_event("Djokovic", "Sinner", {
            "pinnacle": {"Djokovic": 1.50, "Sinner": 2.60},
            "bet365":   {"Djokovic": 1.60, "Sinner": 2.50},
        })
        odds_home, odds_away, bk_home, bk_away = _best_price(event, allowed_bookmakers={"pinnacle"})
        assert odds_home == 1.50 and bk_home == "pinnacle"
        assert odds_away == 2.60 and bk_away == "pinnacle"

    def test_empty_allowed_bookmakers_set_means_none_allowed(self):
        from src.odds_api import _best_price
        event = _multi_bk_event("Djokovic", "Sinner", {
            "pinnacle": {"Djokovic": 1.50, "Sinner": 2.60},
        })
        result = _best_price(event, allowed_bookmakers=set())
        assert result == (None, None, None, None)

    def test_no_bookmakers_quote_event_returns_all_none(self):
        from src.odds_api import _best_price
        event = _multi_bk_event("Djokovic", "Sinner", {
            "bet365": {"Djokovic": 1.60, "Sinner": 2.50},
        })
        result = _best_price(event, allowed_bookmakers={"pinnacle"})
        assert result == (None, None, None, None)

    def test_ignores_non_h2h_markets(self):
        from src.odds_api import _best_price
        event = {
            "home_team": "Djokovic", "away_team": "Sinner",
            "bookmakers": [{
                "key": "pinnacle",
                "markets": [{"key": "totals", "outcomes": []}],
            }],
        }
        result = _best_price(event, allowed_bookmakers=None)
        assert result == (None, None, None, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_odds_api.py -k TestBestPrice -v`
Expected: FAIL with `ImportError: cannot import name '_best_price'`

- [ ] **Step 3: Implement `_best_price`**

In `src/odds_api.py`, add after the `MatchOdds` dataclass (after line 28, before `find_match_odds`):

```python
def _best_price(
    event: dict, allowed_bookmakers: Optional[set[str]]
) -> tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """Scan event["bookmakers"] for the h2h market and return the best
    (highest) price for each side independently, along with which
    bookmaker offered it.

    allowed_bookmakers=None means "allow every bookmaker in the response".
    An empty set means "allow none" (nothing qualifies). The winning
    bookmaker for the home side and the away side may differ — this is
    deliberate, since the best price for one player is not necessarily
    offered by the same book as the best price for their opponent.

    Returns (None, None, None, None) for a side with no allowed bookmaker
    quoting it (mirrors the old "bookmaker didn't quote it" None case).
    """
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    best_home: Optional[float] = None
    best_away: Optional[float] = None
    bk_home: Optional[str] = None
    bk_away: Optional[str] = None

    for bk in event.get("bookmakers", []):
        key = bk.get("key")
        if allowed_bookmakers is not None and key not in allowed_bookmakers:
            continue
        for market in bk.get("markets", []):
            if market.get("key") != "h2h":
                continue
            prices = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if home in prices and (best_home is None or prices[home] > best_home):
                best_home, bk_home = prices[home], key
            if away in prices and (best_away is None or prices[away] > best_away):
                best_away, bk_away = prices[away], key

    return best_home, best_away, bk_home, bk_away
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_odds_api.py -k TestBestPrice -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/odds_api.py tests/test_odds_api.py
git commit -m "$(cat <<'EOF'
feat(odds): add _best_price helper for multi-bookmaker selection

Selects the highest price per side independently across an allowed set
of bookmakers, tracking which book offered each side's best price.

EOF
)"
```

---

## Task 3: `resolve_allowed_bookmakers` env helper

**Files:**
- Modify: `src/odds_api.py`
- Test: `tests/test_odds_api.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_odds_api.py`:

```python
class TestResolveAllowedBookmakers:
    def test_explicit_bookmaker_collapses_to_single_item_set(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.delenv("ALLOWED_BOOKMAKERS", raising=False)
        assert resolve_allowed_bookmakers("pinnacle") == {"pinnacle"}

    def test_explicit_bookmaker_overrides_env_var(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.setenv("ALLOWED_BOOKMAKERS", "bet365,williamhill")
        assert resolve_allowed_bookmakers("pinnacle") == {"pinnacle"}

    def test_no_explicit_bookmaker_parses_env_var(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.setenv("ALLOWED_BOOKMAKERS", "bet365, williamhill ,pinnacle")
        assert resolve_allowed_bookmakers(None) == {"bet365", "williamhill", "pinnacle"}

    def test_unset_env_var_and_no_explicit_means_allow_all(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.delenv("ALLOWED_BOOKMAKERS", raising=False)
        assert resolve_allowed_bookmakers(None) is None

    def test_empty_env_var_means_allow_all(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.setenv("ALLOWED_BOOKMAKERS", "")
        assert resolve_allowed_bookmakers(None) is None

    def test_empty_explicit_string_falls_back_to_env(self, monkeypatch):
        from src.odds_api import resolve_allowed_bookmakers
        monkeypatch.setenv("ALLOWED_BOOKMAKERS", "bet365")
        assert resolve_allowed_bookmakers("") == {"bet365"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_odds_api.py -k TestResolveAllowedBookmakers -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_allowed_bookmakers'`

- [ ] **Step 3: Implement it**

In `src/odds_api.py`, add `import os` to the top-of-file imports (after `import json`, alphabetical with existing style):

```python
import json
import os
import urllib.request
```

Add the function after `_best_price` (before `def fetch_sports_index`):

```python
def resolve_allowed_bookmakers(explicit_bookmaker: Optional[str] = None) -> Optional[set[str]]:
    """Build the allow-list of bookmaker keys eligible for best-price
    selection.

    explicit_bookmaker (the ODDS_API_BOOKMAKER env var or --bookmaker CLI
    flag, when set) is a backward-compatible override: it collapses the
    allow-list to that single bookmaker, reproducing the old fixed-book
    behavior exactly, regardless of ALLOWED_BOOKMAKERS.

    Otherwise, ALLOWED_BOOKMAKERS (comma-separated env var) is parsed into
    a set. Empty or unset means "allow every bookmaker" (None).
    """
    if explicit_bookmaker:
        return {explicit_bookmaker}
    raw = os.environ.get("ALLOWED_BOOKMAKERS", "")
    allowed = {b.strip() for b in raw.split(",") if b.strip()}
    return allowed or None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_odds_api.py -k TestResolveAllowedBookmakers -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/odds_api.py tests/test_odds_api.py
git commit -m "$(cat <<'EOF'
feat(odds): add resolve_allowed_bookmakers with backward-compat override

ODDS_API_BOOKMAKER/--bookmaker, when set, still collapse to a single
fixed bookmaker; otherwise ALLOWED_BOOKMAKERS (or "allow all") applies.

EOF
)"
```

---

## Task 4: `ODDS_API_REGIONS` in the fetch functions

**Files:**
- Modify: `src/odds_api.py`
- Test: `tests/test_odds_api.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_odds_api.py`:

```python
class TestRegionsParam:
    def test_default_regions_when_env_unset(self, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.delenv("ODDS_API_REGIONS", raising=False)
        captured = {}

        class _FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"[]"

        def _fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResp()

        monkeypatch.setattr(odds_api.urllib.request, "urlopen", _fake_urlopen)
        odds_api.fetch_odds_events_by_key("tennis_atp_wimbledon", "fake-key")
        assert "regions=eu,uk,us" in captured["url"]

    def test_custom_regions_from_env(self, monkeypatch):
        import src.odds_api as odds_api
        monkeypatch.setenv("ODDS_API_REGIONS", "au,us2")
        captured = {}

        class _FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"[]"

        def _fake_urlopen(url, timeout=None):
            captured["url"] = url
            return _FakeResp()

        monkeypatch.setattr(odds_api.urllib.request, "urlopen", _fake_urlopen)
        odds_api.fetch_odds_events_by_key("tennis_atp_wimbledon", "fake-key")
        assert "regions=au,us2" in captured["url"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_odds_api.py -k TestRegionsParam -v`
Expected: FAIL — `assert "regions=eu,uk,us" in captured["url"]` fails because the URL still hardcodes `regions=eu`

- [ ] **Step 3: Add `DEFAULT_REGIONS` constant and use it in the URL**

In `src/odds_api.py`, add near `DEFAULT_BOOKMAKER`/`DEFAULT_CACHE_MINUTES` (around line 19-20):

```python
DEFAULT_BOOKMAKER = "pinnacle"
DEFAULT_REGIONS = "eu,uk,us"
DEFAULT_CACHE_MINUTES = 15
```

Replace `fetch_odds_events_by_key` (existing lines 128-135):

```python
def fetch_odds_events_by_key(sport_key: str, api_key: str) -> list[dict]:
    """Fetch raw upcoming h2h odds events for an explicit sport_key."""
    regions = os.environ.get("ODDS_API_REGIONS", DEFAULT_REGIONS)
    url = (
        f"{ODDS_API_BASE}/{sport_key}/odds/"
        f"?apiKey={api_key}&regions={regions}&markets=h2h&oddsFormat=decimal"
    )
    with urllib.request.urlopen(url, timeout=20) as resp:
        return json.loads(resp.read())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_odds_api.py -k TestRegionsParam -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/odds_api.py tests/test_odds_api.py
git commit -m "$(cat <<'EOF'
feat(odds): make regions configurable via ODDS_API_REGIONS (default eu,uk,us)

EOF
)"
```

---

## Task 5: `MatchOdds` gains `bookmaker_a`/`bookmaker_b`; rewrite `find_match_odds`

**Files:**
- Modify: `src/odds_api.py`
- Test: `tests/test_odds_api.py`

- [ ] **Step 1: Update the existing `TestFindMatchOdds` tests to expect the new fields**

Replace the whole `TestFindMatchOdds` class in `tests/test_odds_api.py` (existing lines 86-137) with:

```python
class TestFindMatchOdds:
    def test_direct_order_match(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Novak Djokovic", "Jannik Sinner")
        assert result == MatchOdds(
            odds_a=1.50, odds_b=2.60,
            matched_home="Novak Djokovic", matched_away="Jannik Sinner",
            bookmaker_a="pinnacle", bookmaker_b="pinnacle",
        )

    def test_reversed_order_match(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Jannik Sinner", "Novak Djokovic")
        assert result == MatchOdds(
            odds_a=2.60, odds_b=1.50,
            matched_home="Novak Djokovic", matched_away="Jannik Sinner",
            bookmaker_a="pinnacle", bookmaker_b="pinnacle",
        )

    def test_no_match_returns_none(self):
        events = [_event("Novak Djokovic", "Jannik Sinner")]
        result = find_match_odds(events, "Carlos Alcaraz", "Daniil Medvedev")
        assert result is None

    def test_bookmaker_absent_returns_none(self):
        events = [_event("Novak Djokovic", "Jannik Sinner", bookmaker_key="pinnacle")]
        result = find_match_odds(
            events, "Novak Djokovic", "Jannik Sinner", allowed_bookmakers={"bet365"},
        )
        assert result is None

    def test_independent_bookmaker_per_side(self):
        events = [_multi_bk_event("Novak Djokovic", "Jannik Sinner", {
            "pinnacle": {"Novak Djokovic": 1.50, "Jannik Sinner": 2.60},
            "bet365":   {"Novak Djokovic": 1.55, "Jannik Sinner": 2.55},
        })]
        result = find_match_odds(events, "Novak Djokovic", "Jannik Sinner")
        assert result.odds_a == 1.55 and result.bookmaker_a == "bet365"
        assert result.odds_b == 2.60 and result.bookmaker_b == "pinnacle"

    def test_matches_regardless_of_periods_and_accents(self):
        events = [_event("Sabalenka A.", "Swiatek I.")]
        result = find_match_odds(events, "Aryna Sabalenka", "Iga Swiatek")
        assert result is not None
        assert result.matched_home == "Sabalenka A."

    def test_rejects_surname_match_with_different_first_initial(self):
        # Zverev brothers: same surname, different first name. Querying for
        # Alexander must not silently return odds for a Mischa Zverev event —
        # surname alone was enough to false-positive-match before this check.
        events = [_event("Mischa Zverev", "Novak Djokovic")]
        result = find_match_odds(events, "Alexander Zverev", "Novak Djokovic")
        assert result is None

    def test_accepts_surname_match_when_first_initial_also_matches(self):
        events = [_event("Alexander Zverev", "Novak Djokovic")]
        result = find_match_odds(events, "Alexander Zverev", "Novak Djokovic")
        assert result is not None

    def test_first_initial_check_skipped_when_name_has_no_first_name_info(self):
        # A bare surname-only name carries no initial to compare — must fall
        # back to surname-only matching rather than always rejecting.
        events = [_event("Zverev", "Novak Djokovic")]
        result = find_match_odds(events, "Zverev", "Novak Djokovic")
        assert result is not None
```

Also update `get_match_odds`'s docstring caller usage is unaffected; no changes needed there yet (Task 5 Step 3 rewrites it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_odds_api.py -k TestFindMatchOdds -v`
Expected: FAIL — `MatchOdds() got an unexpected keyword argument 'bookmaker_a'`

- [ ] **Step 3: Update `MatchOdds`, `find_match_odds`, `get_match_odds`**

Replace the `MatchOdds` dataclass (existing lines 23-28):

```python
@dataclass
class MatchOdds:
    odds_a: float
    odds_b: float
    matched_home: str
    matched_away: str
    bookmaker_a: str
    bookmaker_b: str
```

Replace `find_match_odds` (existing lines 31-93):

```python
def find_match_odds(
    events: list[dict],
    player_a: str,
    player_b: str,
    allowed_bookmakers: Optional[set[str]] = None,
) -> Optional[MatchOdds]:
    """Find the event matching player_a/player_b (either order) and extract
    the best available h2h price for each side, independently, across
    allowed_bookmakers (None = every bookmaker in the response).

    Returns None when no event's participants match, or when no allowed
    bookmaker quoted both sides of the matched event.
    """
    surname_a = _surname(player_a)
    surname_b = _surname(player_b)
    initial_a = _first_initial(player_a)
    initial_b = _first_initial(player_b)

    def _initials_compatible(i1: str, i2: str) -> bool:
        # "" means unknown (no first-name token to compare) — don't reject
        # a surname match just because one side lacks that information.
        return not i1 or not i2 or i1 == i2

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        surname_home = _surname(home)
        surname_away = _surname(away)
        initial_home = _first_initial(home)
        initial_away = _first_initial(away)

        if (
            surname_home == surname_a and surname_away == surname_b
            and _initials_compatible(initial_home, initial_a)
            and _initials_compatible(initial_away, initial_b)
        ):
            order = (home, away)
        elif (
            surname_home == surname_b and surname_away == surname_a
            and _initials_compatible(initial_home, initial_b)
            and _initials_compatible(initial_away, initial_a)
        ):
            order = (away, home)
        else:
            continue

        odds_home, odds_away, bk_home, bk_away = _best_price(event, allowed_bookmakers)
        if odds_home is None or odds_away is None:
            return None  # event matched but no allowed bookmaker quoted both sides

        if order == (home, away):
            return MatchOdds(
                odds_a=odds_home, odds_b=odds_away,
                matched_home=home, matched_away=away,
                bookmaker_a=bk_home, bookmaker_b=bk_away,
            )
        return MatchOdds(
            odds_a=odds_away, odds_b=odds_home,
            matched_home=home, matched_away=away,
            bookmaker_a=bk_away, bookmaker_b=bk_home,
        )

    return None
```

Replace `get_match_odds` (existing lines 182-192):

```python
def get_match_odds(
    tour: str,
    player_a: str,
    player_b: str,
    api_key: str,
    allowed_bookmakers: Optional[set[str]] = None,
    cache_minutes: int = DEFAULT_CACHE_MINUTES,
) -> Optional[MatchOdds]:
    """Top-level lookup: cached/fetched events -> matched odds for this pairing."""
    events = get_events(tour, api_key, cache_minutes)
    return find_match_odds(events, player_a, player_b, allowed_bookmakers)
```

- [ ] **Step 4: Run the full odds_api test file**

Run: `pytest tests/test_odds_api.py -v`
Expected: PASS (all tests, including the ones from Tasks 2-4)

- [ ] **Step 5: Commit**

```bash
git add src/odds_api.py tests/test_odds_api.py
git commit -m "$(cat <<'EOF'
feat(odds): find_match_odds/get_match_odds select best price per side

Replaces the fixed-bookmaker lookup with _best_price + an allow-list,
recording which bookmaker won each side on MatchOdds.

EOF
)"
```

---

## Task 6: `daily_scanner._extract_h2h_odds` + `DiscoveredMatch` fields

**Files:**
- Modify: `src/daily_scanner.py`
- Test: `tests/test_daily_scanner.py`

- [ ] **Step 1: Update the failing tests**

Replace `TestExtractH2hOdds` in `tests/test_daily_scanner.py` (existing lines 62-69):

```python
class TestExtractH2hOdds:
    def test_extracts_prices_for_allowed_bookmaker(self):
        event = _event(prices={"Novak Djokovic": 1.5, "Jannik Sinner": 2.6}, bookmaker_key="bet365")
        assert _extract_h2h_odds(event, {"bet365"}) == (1.5, 2.6, "bet365", "bet365")

    def test_returns_none_when_no_allowed_bookmaker_quotes_it(self):
        event = _event(bookmaker_key="pinnacle")
        assert _extract_h2h_odds(event, {"bet365"}) is None

    def test_none_allowed_bookmakers_means_all_allowed(self):
        event = _event(prices={"Novak Djokovic": 1.5, "Jannik Sinner": 2.6}, bookmaker_key="pinnacle")
        assert _extract_h2h_odds(event, None) == (1.5, 2.6, "pinnacle", "pinnacle")
```

Update the `DiscoveredMatch` construction assertion in `test_discovers_and_matches_full_pipeline` (existing lines 90-102) — add two lines after the existing `odds` assertion:

```python
        assert m.odds_a == 1.5 and m.odds_b == 2.6
        assert m.bookmaker_a == DEFAULT_BOOKMAKER and m.bookmaker_b == DEFAULT_BOOKMAKER
        assert m.match_date == date(2026, 7, 27)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_daily_scanner.py -k "TestExtractH2hOdds or test_discovers_and_matches_full_pipeline" -v`
Expected: FAIL — `_extract_h2h_odds() takes 2 positional arguments but 2 were given` type mismatch (old signature took a bookmaker string, returned a 2-tuple) and `AttributeError: 'DiscoveredMatch' object has no attribute 'bookmaker_a'`

- [ ] **Step 3: Implement**

In `src/daily_scanner.py`, update the import block (existing lines 55-60):

```python
from src.odds_api import (
    DEFAULT_BOOKMAKER,
    _best_price,
    fetch_odds_events_by_key,
    fetch_sports_index,
    list_tennis_sport_keys,
    resolve_allowed_bookmakers,
)
```

Replace `_extract_h2h_odds` (existing lines 125-139):

```python
def _extract_h2h_odds(
    event: dict, allowed_bookmakers: Optional[set[str]]
) -> Optional[tuple[float, float, str, str]]:
    """Return (odds_home, odds_away, bookmaker_home, bookmaker_away) from the
    best-priced allowed bookmaker's h2h market, or None if no allowed
    bookmaker quoted both sides of this event."""
    odds_home, odds_away, bk_home, bk_away = _best_price(event, allowed_bookmakers)
    if odds_home is None or odds_away is None:
        return None
    return odds_home, odds_away, bk_home, bk_away
```

Add `bookmaker_a`/`bookmaker_b` fields to `DiscoveredMatch` (existing lines 76-87), with defaults so existing call sites that don't pass them (tests, `daily_workflow`'s test fixtures) keep working:

```python
@dataclass
class DiscoveredMatch:
    tour: str
    tournament: str
    surface: str
    match_date: date
    player_a: str   # canonical dataset name (post player_matcher resolution)
    player_b: str
    odds_a: float
    odds_b: float
    raw_home: str   # name as reported by The Odds API, for display/debugging
    raw_away: str
    bookmaker_a: str = ""
    bookmaker_b: str = ""
```

Update `discover_matches` (existing lines 142-195) — change the `bookmaker` parameter to `bookmaker: Optional[str] = None` (was `bookmaker: str = DEFAULT_BOOKMAKER`), resolve the allow-list once at the top, and thread the two new fields through:

```python
def discover_matches(
    api_key: str,
    canonical_names_by_tour: dict[str, list[str]],
    bookmaker: Optional[str] = None,
    days_ahead: int = DEFAULT_DAYS_AHEAD,
    tours: tuple[str, ...] = ("atp", "wta"),
) -> list[DiscoveredMatch]:
    """Discover today's/next-matchday's matches with bettable odds, matched
    to canonical dataset player names.

    `bookmaker` is a backward-compatible override (ODDS_API_BOOKMAKER env
    var or --bookmaker CLI flag) that collapses best-price selection to a
    single fixed bookmaker — see resolve_allowed_bookmakers.

    Skips (with a printed reason) any event where the tournament's surface
    can't be resolved, either player can't be matched to the dataset, or no
    allowed bookmaker quoted h2h odds for both sides — never raises on a
    per-event problem, since one bad event must not abort the whole scan.
    """
    allowed_bookmakers = resolve_allowed_bookmakers(bookmaker)
    sports_index = fetch_sports_index(api_key)
    tennis_sports = [s for s in list_tennis_sport_keys(sports_index) if s["tour"] in tours]

    matches: list[DiscoveredMatch] = []
    for sport in tennis_sports:
        surface = resolve_surface(sport["title"], sport["key"])
        if surface is None:
            continue  # resolve_surface already printed the warning

        events = fetch_odds_events_by_key(sport["key"], api_key)
        for event in events:
            commence_time = event.get("commence_time")
            if not commence_time or not _is_within_window(commence_time, days_ahead):
                continue

            odds = _extract_h2h_odds(event, allowed_bookmakers)
            if odds is None:
                continue
            odds_home, odds_away, bookmaker_home, bookmaker_away = odds

            home, away = event.get("home_team", ""), event.get("away_team", "")
            canonical = canonical_names_by_tour.get(sport["tour"], [])
            player_a = match_player_name(home, canonical)
            player_b = match_player_name(away, canonical)
            if player_a is None or player_b is None:
                unmatched = [n for n, p in ((home, player_a), (away, player_b)) if p is None]
                print(f"  Aviso: no se pudo emparejar jugador(es) {', '.join(unmatched)} "
                      f"({sport['title']}). Partido excluido.")
                continue

            match_date = to_lima(datetime.fromisoformat(commence_time.replace("Z", "+00:00"))).date()
            matches.append(DiscoveredMatch(
                tour=sport["tour"], tournament=sport["title"], surface=surface,
                match_date=match_date, player_a=player_a, player_b=player_b,
                odds_a=odds_home, odds_b=odds_away,
                raw_home=home, raw_away=away,
                bookmaker_a=bookmaker_home, bookmaker_b=bookmaker_away,
            ))

    return matches
```

Note: `run_scan`'s `bookmaker: str = DEFAULT_BOOKMAKER` parameter (existing line 262) and `main()`'s `parser.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER)` (existing line 330) are updated in Task 10 — leave them alone here to keep this task's diff focused, `discover_matches` accepting `Optional[str]` is compatible with either default in the meantime.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_daily_scanner.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/daily_scanner.py tests/test_daily_scanner.py
git commit -m "$(cat <<'EOF'
feat(scanner): thread best-price bookmaker selection through DiscoveredMatch

_extract_h2h_odds delegates to the shared _best_price helper; discover_matches
resolves an allow-list once and records bookmaker_a/bookmaker_b per match.

EOF
)"
```

---

## Task 7: `value_bets_log.csv` gains `bookmaker_a`/`bookmaker_b`

**Files:**
- Modify: `src/value_analysis.py`
- Test: `tests/test_value_analysis.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_value_analysis.py` inside `class TestLogQueryStatus` (after the existing test at line ~401, i.e. right after `test_...auto...` around line 396-401):

```python
    def test_bookmaker_a_and_b_recorded_when_provided(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10,
                  bookmaker_a="bet365", bookmaker_b="pinnacle")
        row = _read_log(log_path)[0]
        assert row["bookmaker_a"] == "bet365"
        assert row["bookmaker_b"] == "pinnacle"

    def test_bookmaker_defaults_to_empty_string(self, log_path):
        pred = _make_pred()
        log_query("atp", "Test", "hard", date(2026, 7, 1),
                  "Player A", "Player B", pred, _val(), _val(), 1.90, 2.10)
        row = _read_log(log_path)[0]
        assert row["bookmaker_a"] == ""
        assert row["bookmaker_b"] == ""
```

Update `test_migrates_old_header_and_preserves_old_row` (existing lines 403-470) — add two assertions right after the existing `assert migrated["odds_b_source"] == "manual"` line (existing line 464):

```python
        assert migrated["odds_a_source"] == "manual"
        assert migrated["odds_b_source"] == "manual"
        assert migrated["bookmaker_a"] == "pinnacle"
        assert migrated["bookmaker_b"] == "pinnacle"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_value_analysis.py -k "test_bookmaker_a_and_b_recorded_when_provided or test_bookmaker_defaults_to_empty_string or test_migrates_old_header_and_preserves_old_row" -v`
Expected: FAIL — `KeyError: 'bookmaker_a'` (field not yet in `_LOG_FIELDS`) and `TypeError: log_query() got an unexpected keyword argument 'bookmaker_a'`

- [ ] **Step 3: Implement**

In `src/value_analysis.py`, add the two new fields to `_LOG_FIELDS` (existing lines 698-715), placed right after `odds_b`/`odds_b_source`:

```python
_LOG_FIELDS = [
    "timestamp", "tour", "tournament", "surface", "match_date",
    "player_a", "player_b",
    "rank_a", "rank_a_source", "rank_b", "rank_b_source",
    "p_a_raw", "p_a_cal", "p_b_raw", "p_b_cal",
    "odds_a", "odds_a_source", "odds_b", "odds_b_source",
    "bookmaker_a", "bookmaker_b",
    "implied_a", "implied_b",
    "edge_a", "ev_a", "kelly_a",
    "edge_b", "ev_b", "kelly_b",
    "shrinkage_applied",
    "status",    # ok / invalid_missing_elo / test_never_played
    "result",    # pending / A_win / B_win / excluded (test_never_played rows)
    "profit",    # filled in later
    "elo_diff", "elo_prob", "rank_diff", "form_diff",
    "surface_form_diff", "h2h_rate", "rest_diff",
    "rolling_elo_diff", "age_multiplier_diff", "rust_factor_diff",
    "fatigue_multiplier_diff", "surface_transition_multiplier_diff", "adjusted_elo_diff",
]
```

Update `_migrate_log_header_if_needed` (existing lines 718-736) to backfill `bookmaker_a`/`bookmaker_b` with `"pinnacle"` (the fixed book every historical row actually used), alongside the existing `odds_a_source`/`odds_b_source` backfill:

```python
def _migrate_log_header_if_needed() -> None:
    """If LOG_PATH exists with an older header than _LOG_FIELDS (e.g. missing
    odds_a_source/odds_b_source or bookmaker_a/bookmaker_b), rewrite it with
    the current header so old rows stay readable instead of silently
    misaligning on the next append."""
    if not LOG_PATH.exists():
        return
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == _LOG_FIELDS:
            return
        rows = list(reader)
    for row in rows:
        row.setdefault("odds_a_source", "manual")
        row.setdefault("odds_b_source", "manual")
        row.setdefault("bookmaker_a", "pinnacle")
        row.setdefault("bookmaker_b", "pinnacle")
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
```

Update `log_query`'s signature and row dict (existing lines 739-790):

```python
def log_query(tour, tournament, surface, match_date,
              player_a, player_b,
              pred, val_a, val_b, odds_a, odds_b,
              odds_a_source: str = "manual", odds_b_source: str = "manual",
              bookmaker_a: str = "", bookmaker_b: str = "") -> None:
    """Append one match prediction + odds to the CSV log.

    Rows where either player's Elo was not found (default 1500) are saved
    with status='invalid_missing_elo' instead of 'ok', so they can be
    filtered out during analysis without contaminating the pick history.
    """
    _migrate_log_header_if_needed()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    feats  = pred["features"]
    shrink = abs(pred["p_a_cal"] - pred["p_a_raw"]) > 0.001

    elo_ok = pred.get("elo_found_a", True) and pred.get("elo_found_b", True)
    status = "ok" if elo_ok else "invalid_missing_elo"

    row = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "tour":            tour,
        "tournament":      tournament,
        "surface":         surface,
        "match_date":      match_date.isoformat(),
        "player_a":        player_a,
        "player_b":        player_b,
        "rank_a":          pred["rank_a"] if pred["rank_a"] is not None else "",
        "rank_a_source":   pred["rank_a_source"] or "none",
        "rank_b":          pred["rank_b"] if pred["rank_b"] is not None else "",
        "rank_b_source":   pred["rank_b_source"] or "none",
        "p_a_raw":         round(pred["p_a_raw"], 4),
        "p_a_cal":         round(pred["p_a_cal"], 4),
        "p_b_raw":         round(pred["p_b_raw"], 4),
        "p_b_cal":         round(pred["p_b_cal"], 4),
        "odds_a":          odds_a,
        "odds_a_source":   odds_a_source,
        "odds_b":          odds_b,
        "odds_b_source":   odds_b_source,
        "bookmaker_a":     bookmaker_a,
        "bookmaker_b":     bookmaker_b,
        "implied_a":       round(val_a["implied_prob"], 4),
        "implied_b":       round(val_b["implied_prob"], 4),
        "edge_a":          round(val_a["edge"], 4),
        "ev_a":            round(val_a["ev"], 4),
        "kelly_a":         round(val_a["kelly_fraction"], 4),
        "edge_b":          round(val_b["edge"], 4),
        "ev_b":            round(val_b["ev"], 4),
        "kelly_b":         round(val_b["kelly_fraction"], 4),
        "shrinkage_applied": shrink,
        "status":          status,
        "result":          "pending",
        "profit":          "",
        **{k: round(feats[k], 4) for k in FEATURE_COLS},
    }

    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)
```

(The last two lines of the original function, writing the row, were truncated in the earlier read — keep whatever the existing trailing lines are; only the `row = {...}` dict body and the `def log_query(...)` signature line change.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_value_analysis.py -v`
Expected: PASS (all tests — this file has ~80+ tests across many classes; confirm none regressed)

- [ ] **Step 5: Commit**

```bash
git add src/value_analysis.py tests/test_value_analysis.py
git commit -m "$(cat <<'EOF'
feat(log): add bookmaker_a/bookmaker_b to value_bets_log.csv

Old rows are migrated with bookmaker_a=bookmaker_b="pinnacle" (the fixed
book every historical row actually used), same pattern as the prior
odds_a_source/odds_b_source migration.

EOF
)"
```

---

## Task 8: `prediction_audit_log.csv` gains `bookmaker_a`/`bookmaker_b`

**Files:**
- Modify: `src/calibration_audit.py`
- Test: `tests/test_calibration_audit.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_calibration_audit.py` inside `class TestLogPredictionAudit`:

```python
    def test_bookmaker_a_and_b_recorded(self, audit_log_path):
        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
            bookmaker_a="bet365", bookmaker_b="pinnacle",
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["bookmaker_a"] == "bet365"
        assert row["bookmaker_b"] == "pinnacle"

    def test_bookmaker_defaults_to_empty_string(self, audit_log_path):
        pred = _make_pred()
        v = calculate_value(0.58, 1.90)
        calibration_audit.log_prediction_audit(
            "atp", "Test", "hard", date(2026, 7, 25),
            "Player A", "Player B",
            pred, v, v, 1.90, 1.90,
            "manual", "manual",
            decision="logged",
            model_snapshot_id=None,
            shrink_hi=0.90, shrink_lo=0.10, shrink_rate=0.60,
        )
        row = _read_audit_log(audit_log_path)[0]
        assert row["bookmaker_a"] == ""
        assert row["bookmaker_b"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calibration_audit.py -k bookmaker -v`
Expected: FAIL — `TypeError: log_prediction_audit() got an unexpected keyword argument 'bookmaker_a'`

- [ ] **Step 3: Implement**

In `src/calibration_audit.py`, add the two fields to `_AUDIT_LOG_FIELDS` (existing lines 19-31), after `odds_b_source`:

```python
_AUDIT_LOG_FIELDS = [
    "timestamp", "tour", "tournament", "surface", "match_date",
    "player_a", "player_b",
    "p_a_raw", "p_a_cal", "p_b_raw", "p_b_cal",
    "shrink_hi", "shrink_lo", "shrink_rate", "shrinkage_applied",
    "odds_a", "odds_a_source", "odds_b", "odds_b_source",
    "bookmaker_a", "bookmaker_b",
    "implied_a", "implied_b",
    "edge_a", "ev_a", "kelly_a",
    "edge_b", "ev_b", "kelly_b",
    "model_commit", "model_snapshot_id",
    "elo_found_a", "elo_found_b",
    "decision",  # logged / passed_low_edge / passed_user_declined / blocked_suspicious / invalid_missing_elo
]
```

Update `log_prediction_audit`'s signature (existing lines 73-81) and row dict (existing lines 101-132ish):

```python
def log_prediction_audit(
    tour, tournament, surface, match_date,
    player_a, player_b,
    pred: dict, val_a: dict, val_b: dict, odds_a: float, odds_b: float,
    odds_a_source: str, odds_b_source: str,
    decision: str,
    model_snapshot_id: str | None,
    shrink_hi: float, shrink_lo: float, shrink_rate: float,
    bookmaker_a: str = "", bookmaker_b: str = "",
) -> None:
```

(docstring unchanged). In the `row = {...}` dict, add the two fields right after `"odds_b_source": odds_b_source,`:

```python
        "odds_b_source":   odds_b_source,
        "bookmaker_a":     bookmaker_a,
        "bookmaker_b":     bookmaker_b,
        "implied_a":       round(val_a["implied_prob"], 4),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calibration_audit.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/calibration_audit.py tests/test_calibration_audit.py
git commit -m "$(cat <<'EOF'
feat(audit): add bookmaker_a/bookmaker_b to prediction_audit_log.csv

Mirrors the value_bets_log.csv schema change so both logs stay
consistent for calibration analysis.

EOF
)"
```

---

## Task 9: `daily_scanner.run_scan` passes bookmaker through to both logs

**Files:**
- Modify: `src/daily_scanner.py`
- Test: `tests/test_daily_scanner.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_daily_scanner.py`, a new test class:

```python
class TestRunScanBookmakerLogging:
    def test_logs_bookmaker_a_and_b_from_match(self, monkeypatch):
        import src.daily_scanner as scanner

        m = scanner.DiscoveredMatch(
            tour="atp", tournament="ATP Wimbledon", surface="grass",
            match_date=date(2026, 7, 27), player_a="Novak Djokovic", player_b="Jannik Sinner",
            odds_a=1.5, odds_b=2.6, raw_home="Novak Djokovic", raw_away="Jannik Sinner",
            bookmaker_a="bet365", bookmaker_b="pinnacle",
        )
        r = EvaluatedMatch(
            match=m, pred=_pred(), val_a=_val(has_value=True), val_b=_val(has_value=False),
            elo_ok=True, suspicious=False,
        )
        monkeypatch.setattr(scanner, "_load_models", lambda tours, retrain: {"atp": (None,)*6})
        monkeypatch.setattr(scanner, "_canonical_names", lambda models: {"atp": [], "wta": []})
        monkeypatch.setattr(scanner, "discover_matches", lambda *a, **kw: [m])
        monkeypatch.setattr(scanner, "evaluate_matches", lambda matches, models, halt_on_suspicious=True: [r])
        monkeypatch.setattr("builtins.input", lambda *a: "s")
        monkeypatch.setenv("ODDS_API_KEY", "fake-key")

        logged = []
        monkeypatch.setattr(scanner, "log_query", lambda *a, **kw: logged.append(kw))
        monkeypatch.setattr(scanner, "log_prediction_audit", lambda *a, **kw: None)

        scanner.run_scan(tours=("atp",))

        assert logged[0]["bookmaker_a"] == "bet365"
        assert logged[0]["bookmaker_b"] == "pinnacle"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_daily_scanner.py -k test_logs_bookmaker_a_and_b_from_match -v`
Expected: FAIL — `KeyError: 'bookmaker_a'` (log_query call in `run_scan` doesn't pass it yet)

- [ ] **Step 3: Implement**

In `src/daily_scanner.py`, update the `log_query` call inside `run_scan` (existing lines 297-305):

```python
    for r in results:
        logged = save_all and r.is_value_bet
        if logged:
            log_query(
                r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                r.match.player_a, r.match.player_b,
                r.pred, r.val_a, r.val_b, r.match.odds_a, r.match.odds_b,
                odds_a_source="auto", odds_b_source="auto",
                bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
            )
```

And the `log_prediction_audit` call right below it (existing lines 306-320):

```python
        try:
            decision = classify_audit_decision(
                r.low_sample, r.suspicious, r.elo_ok, logged,
                r.val_a["has_value"], r.val_b["has_value"],
            )
            log_prediction_audit(
                r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                r.match.player_a, r.match.player_b,
                r.pred, r.val_a, r.val_b, r.match.odds_a, r.match.odds_b,
                odds_a_source="auto", odds_b_source="auto",
                decision=decision, model_snapshot_id=None,
                shrink_hi=_SHRINK_HI, shrink_lo=_SHRINK_LO, shrink_rate=_SHRINK_RATE,
                bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
            )
        except Exception as e:
            print(f"  Aviso: no se pudo escribir en el audit log ({e}). Continuando.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_daily_scanner.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/daily_scanner.py tests/test_daily_scanner.py
git commit -m "$(cat <<'EOF'
feat(scanner): propagate bookmaker_a/bookmaker_b into both CSV logs

EOF
)"
```

---

## Task 10: retire the fixed-bookmaker default in `daily_scanner`'s CLI/`run_scan`

**Files:**
- Modify: `src/daily_scanner.py`
- Test: `tests/test_daily_scanner.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_daily_scanner.py`:

```python
class TestRunScanDefaultBookmaker:
    def test_default_bookmaker_arg_is_none_not_pinnacle(self):
        import argparse
        import src.daily_scanner as scanner
        parser = argparse.ArgumentParser()
        parser.add_argument("--bookmaker", default=None)
        args = parser.parse_args([])
        assert args.bookmaker is None
```

This test documents the intended CLI default rather than exercising `main()` directly (argparse's own parser isn't easily introspectable from outside `main()`); the substantive behavior (an unset `--bookmaker` no longer forces `pinnacle`-only) is already covered by `discover_matches`'s `Optional[str] = None` default from Task 6 and `resolve_allowed_bookmakers`'s tests from Task 3.

- [ ] **Step 2: Update `run_scan` and `main()`**

In `src/daily_scanner.py`, change `run_scan`'s signature (existing line 262):

```python
def run_scan(
    tours: tuple[str, ...] = ("atp", "wta"),
    bookmaker: Optional[str] = None,
    days_ahead: int = DEFAULT_DAYS_AHEAD,
    halt_on_suspicious: bool = True,
    retrain: bool = False,
) -> None:
```

Change `main()`'s argparse default (existing line 330):

```python
    parser.add_argument(
        "--bookmaker", default=None,
        help="Restringe la seleccion a un unico bookmaker (por defecto: mejor precio "
             "entre todos los permitidos por ALLOWED_BOOKMAKERS).",
    )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_daily_scanner.py -v`
Expected: PASS (all tests)

- [ ] **Step 4: Commit**

```bash
git add src/daily_scanner.py tests/test_daily_scanner.py
git commit -m "$(cat <<'EOF'
feat(scanner): --bookmaker/run_scan default to best-price across all books

Previously defaulted to DEFAULT_BOOKMAKER ("pinnacle"), silently
restricting every scan to one book unless overridden. Now defaults to
None, letting resolve_allowed_bookmakers apply ALLOWED_BOOKMAKERS (or
"allow all") unless --bookmaker is explicitly passed.

EOF
)"
```

---

## Task 11: `scripts/daily_workflow.py` propagates bookmaker

**Files:**
- Modify: `scripts/daily_workflow.py`
- Test: `tests/test_daily_workflow.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_daily_workflow.py`, inside `class TestRun`:

```python
    def test_selected_picks_include_bookmaker(self, monkeypatch):
        m = DiscoveredMatch(
            tour="atp", tournament="ATP Test Open", surface="hard",
            match_date=date(2026, 8, 19), player_a="A Player", player_b="B Player",
            odds_a=2.0, odds_b=1.8, raw_home="A Player", raw_away="B Player",
            bookmaker_a="bet365", bookmaker_b="pinnacle",
        )
        r = _result(match=m, val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [r])

        selected = workflow.run()

        assert selected[0]["bookmaker"] == "bet365"

    def test_logs_query_with_bookmaker_from_match(self, monkeypatch):
        m = DiscoveredMatch(
            tour="atp", tournament="ATP Test Open", surface="hard",
            match_date=date(2026, 8, 19), player_a="A Player", player_b="B Player",
            odds_a=2.0, odds_b=1.8, raw_home="A Player", raw_away="B Player",
            bookmaker_a="bet365", bookmaker_b="pinnacle",
        )
        r = _result(match=m, val_a=_val(edge=0.05, kelly=0.04, has_value=True))
        self._wire_common(monkeypatch, [r])

        workflow.run()

        _, kwargs = self.logged_queries[0]
        assert kwargs["bookmaker_a"] == "bet365"
        assert kwargs["bookmaker_b"] == "pinnacle"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_daily_workflow.py -k "bookmaker" -v`
Expected: FAIL — `KeyError: 'bookmaker'` and `KeyError: 'bookmaker_a'`

- [ ] **Step 3: Implement**

In `scripts/daily_workflow.py`, update the `selected.append({...})` block inside `run()` (existing lines 110-119):

```python
        match_label = f"{r.match.player_a} vs {r.match.player_b}"
        for qualifies, val, odds, name, bookmaker in (
            (qualifies_a, r.val_a, r.match.odds_a, r.match.player_a, r.match.bookmaker_a),
            (qualifies_b, r.val_b, r.match.odds_b, r.match.player_b, r.match.bookmaker_b),
        ):
            if qualifies:
                selected.append({
                    "match": match_label, "pick": name, "odds": odds,
                    "edge": val["edge"], "kelly": val["kelly_fraction"],
                    "bookmaker": bookmaker,
                })
```

Update the `log_query` call right below it (existing lines 121-126):

```python
        if not dry_run:
            log_query(
                r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                r.match.player_a, r.match.player_b, r.pred, r.val_a, r.val_b,
                r.match.odds_a, r.match.odds_b, odds_a_source="auto", odds_b_source="auto",
                bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
            )
```

Update the `log_prediction_audit` call (existing lines 135-141):

```python
                log_prediction_audit(
                    r.match.tour, r.match.tournament, r.match.surface, r.match.match_date,
                    r.match.player_a, r.match.player_b, r.pred, r.val_a, r.val_b,
                    r.match.odds_a, r.match.odds_b, odds_a_source="auto", odds_b_source="auto",
                    decision=decision, model_snapshot_id=None,
                    shrink_hi=_SHRINK_HI, shrink_lo=_SHRINK_LO, shrink_rate=_SHRINK_RATE,
                    bookmaker_a=r.match.bookmaker_a, bookmaker_b=r.match.bookmaker_b,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_daily_workflow.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/daily_workflow.py tests/test_daily_workflow.py
git commit -m "$(cat <<'EOF'
feat(workflow): propagate bookmaker into daily_workflow picks and logs

EOF
)"
```

---

## Task 12: `notifier.py` email table shows the bookmaker

**Files:**
- Modify: `src/utils/notifier.py`
- Test: `tests/test_notifier.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_notifier.py`:

```python
SAMPLE_PICKS_WITH_BOOKMAKER = [
    {"match": "Tirante vs Mensik", "pick": "Tirante", "odds": 2.75, "edge": 0.0322,
     "kelly": 0.0092, "bookmaker": "bet365"},
]


class TestPicksTableBookmakerColumn:
    def test_shows_bookmaker_when_present(self):
        html = notifier.build_email_html(SAMPLE_PICKS_WITH_BOOKMAKER, SAMPLE_VPN_OK)
        assert "bet365" in html

    def test_blank_when_bookmaker_absent(self):
        # SAMPLE_PICKS (module-level) has no "bookmaker" key — must not raise.
        html = notifier.build_email_html(SAMPLE_PICKS, SAMPLE_VPN_OK)
        assert "Tirante vs Mensik" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notifier.py -k Bookmaker -v`
Expected: FAIL — `test_shows_bookmaker_when_present` fails because "bet365" isn't in the HTML yet; `test_blank_when_bookmaker_absent` currently passes already (no column exists) but is included to pin down the no-KeyError requirement once the column is added

- [ ] **Step 3: Implement**

In `src/utils/notifier.py`, update `_picks_table_html` (existing lines 22-48):

```python
def _picks_table_html(selected_picks: list[dict]) -> str:
    if not selected_picks:
        return "<p>No se seleccionaron picks en esta jornada.</p>"
    rows = "\n".join(
        f'<tr><td style="padding:6px;border-bottom:1px solid #e0e0e0;">{p["match"]}</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;">{p["pick"]}</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;">{p.get("bookmaker", "")}</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;text-align:right;">{p["odds"]:.2f}</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;text-align:right;">{p["edge"] * 100:+.1f}%</td>'
        f'<td style="padding:6px;border-bottom:1px solid #e0e0e0;text-align:right;">{p["kelly"] * 100:.2f}%</td></tr>'
        for p in selected_picks
    )
    return f"""
    <table style="border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:14px;">
      <thead>
        <tr style="background:#1f3a5f;color:#fff;">
          <th style="padding:8px;text-align:left;">Partido</th>
          <th style="padding:8px;text-align:left;">Pick</th>
          <th style="padding:8px;text-align:left;">Bookmaker</th>
          <th style="padding:8px;text-align:right;">Cuota</th>
          <th style="padding:8px;text-align:right;">Edge</th>
          <th style="padding:8px;text-align:right;">Stake (1/2 Kelly)</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_notifier.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils/notifier.py tests/test_notifier.py
git commit -m "$(cat <<'EOF'
feat(notifier): show bookmaker column in the daily picks email

Falls back to a blank cell when a pick dict has no "bookmaker" key, so
callers that don't set it (or older test fixtures) don't KeyError.

EOF
)"
```

---

## Task 13: `settle_workflow.py` report shows the bookmaker

**Files:**
- Modify: `scripts/settle_workflow.py`
- Test: `tests/test_settle_workflow.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_settle_workflow.py`. First extend the module-level `_row` helper to optionally carry bookmaker fields (it must stay backward-compatible — existing calls don't pass them):

```python
def _row(match_date="2026-08-19", player_a="A Player", player_b="B Player",
         odds_a="2.0", odds_b="1.8", kelly_a="0.02", kelly_b="0.0",
         status="ok", result="pending", profit="",
         bookmaker_a="", bookmaker_b=""):
    return {
        "timestamp": "2026-08-19T10:00:00", "tour": "atp", "tournament": "ATP Test Open",
        "surface": "hard", "match_date": match_date, "player_a": player_a, "player_b": player_b,
        "odds_a": odds_a, "odds_b": odds_b, "ev_a": "0.1", "ev_b": "0.1",
        "kelly_a": kelly_a, "kelly_b": kelly_b,
        "status": status, "result": result, "profit": profit,
        "bookmaker_a": bookmaker_a, "bookmaker_b": bookmaker_b,
    }
```

Add a new test class:

```python
class TestBuildReportBookmakerColumn:
    def test_shows_bookmaker_for_settled_side(self, tmp_path, monkeypatch):
        path = tmp_path / "value_bets_log.csv"
        _write_csv(path, [
            _row(match_date="2026-08-19", odds_a="2.0", kelly_a="0.05", bookmaker_a="bet365"),
        ])
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        report_path = tmp_path / "2026-08-19-jornada-cincinnati.md"
        content = report_path.read_text(encoding="utf-8")
        assert "bet365" in content

    def test_blank_when_bookmaker_field_absent(self, tmp_path, monkeypatch):
        # Simulates a pre-migration CSV on disk (written before this task
        # added bookmaker_a/bookmaker_b to HEADER) — the row dicts read back
        # via csv.DictReader genuinely have no bookmaker_a/bookmaker_b keys
        # at all, so _build_report's row.get(...) fallback must not raise.
        OLD_HEADER = [h for h in HEADER if h not in ("bookmaker_a", "bookmaker_b")]
        path = tmp_path / "value_bets_log.csv"
        old_row = {k: v for k, v in _row(match_date="2026-08-19", odds_a="2.0", kelly_a="0.05").items()
                   if k in OLD_HEADER}
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OLD_HEADER)
            writer.writeheader()
            writer.writerow(old_row)
        monkeypatch.setattr(settle, "_ask_winner", lambda row: "a")
        monkeypatch.setattr(settle, "commit_and_push", lambda *a, **kw: True)

        settle.run(date(2026, 8, 19), log_path=str(path), report_dir=str(tmp_path))

        report_path = tmp_path / "2026-08-19-jornada-cincinnati.md"
        assert report_path.exists()
```

Note: since `_row` now always includes `bookmaker_a`/`bookmaker_b` keys, and `_write_csv` writes using the module-level `HEADER` list (which does NOT include those two keys), `csv.DictWriter` would raise `ValueError: dict contains fields not in fieldnames` for every existing test in this file that uses `_write_csv(path, [_row(...)])`. Add `"bookmaker_a", "bookmaker_b"` to `HEADER` (existing lines 10-13) to keep all existing `_write_csv` calls working:

```python
HEADER = [
    "timestamp", "tour", "tournament", "surface", "match_date", "player_a", "player_b",
    "odds_a", "odds_b", "ev_a", "ev_b", "kelly_a", "kelly_b", "status", "result", "profit",
    "bookmaker_a", "bookmaker_b",
]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_settle_workflow.py -v`
Expected: FAIL — `test_shows_bookmaker_for_settled_side` fails because `_build_report`'s table has no bookmaker column yet; all other existing tests should still PASS after the `HEADER`/`_row` change in Step 1 (verify this — if any existing test breaks here, the `HEADER` edit needs to happen before re-running)

- [ ] **Step 3: Implement**

In `scripts/settle_workflow.py`, update `_build_report` (existing lines 90-111):

```python
def _build_report(target: str, settled: list[dict], log_path: str, bankroll: float = DEFAULT_BANKROLL) -> str:
    lines = [f"# Jornada {target} — liquidacion\n"]
    lines.append(f"{len(settled)} apuesta(s) liquidada(s) en `data/value_bets_log.csv`.\n")
    lines.append("| Partido | Pick | Bookmaker | Odds | Kelly | Resultado | Profit |")
    lines.append("|---|---|---|---|---|---|---|")

    stake_total = profit_total = 0.0
    wins = 0
    for row in settled:
        side = _bet_side(row)
        odds = float(row[f"odds_{side}"])
        kelly = float(row[f"kelly_{side}"])
        pick = row["player_a"] if side == "a" else row["player_b"]
        opponent = row["player_b"] if side == "a" else row["player_a"]
        bookmaker = row.get(f"bookmaker_{side}", "")
        profit = float(row["profit"])
        stake_total += kelly
        profit_total += profit
        wins += 1 if profit > 0 else 0
        lines.append(
            f"| {pick} vs {opponent} | {pick} | {bookmaker} | {odds:.2f} | {kelly * 100:.2f}% | "
            f"{row['result']} | {profit:+.4f} |"
        )
```

(the rest of the function — summary lines and accumulated metrics — is unchanged)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_settle_workflow.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/settle_workflow.py tests/test_settle_workflow.py
git commit -m "$(cat <<'EOF'
feat(settle): show bookmaker column in the jornada settlement report

Falls back to a blank cell via row.get() so rows without the new field
(pre-migration CSVs) don't raise a KeyError.

EOF
)"
```

---

## Task 14: interactive CLI (`value_analysis.py`) shows and logs the real bookmaker

**Files:**
- Modify: `src/value_analysis.py`

**No new automated test** — `interactive_cli()` is a manual REPL loop with no existing unit test harness in this codebase (confirmed: no `test_interactive_cli` tests exist in `tests/test_value_analysis.py`). This task is verified manually per Step 3 below, consistent with how this function's other behavior changes have been handled historically.

- [ ] **Step 1: Update `try_auto_odds`'s bookmaker resolution**

In `src/value_analysis.py`, replace the `try_auto_odds` function body (existing lines 909-930):

```python
def try_auto_odds(tour: str, player_a: str, player_b: str) -> Optional[MatchOdds]:
    """Best-effort odds auto-fill via The Odds API. Never raises, never blocks.

    Returns None silently when ODDS_API_KEY isn't set or no match/bookmaker
    quote was found. Prints a one-time-per-session warning on genuine
    failures (network error, bad key, rate limit) so the user knows why
    it's not filling in — the manual flow always still works.
    """
    global _odds_warned
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        return None
    try:
        allowed_bookmakers = resolve_allowed_bookmakers(os.environ.get("ODDS_API_BOOKMAKER"))
        cache_minutes = int(os.environ.get("ODDS_API_CACHE_MINUTES", DEFAULT_CACHE_MINUTES))
        return get_match_odds(tour, player_a, player_b, api_key, allowed_bookmakers, cache_minutes)
    except Exception as e:
        if not _odds_warned:
            print(f"\n  Aviso: no se pudieron obtener cuotas automaticas ({e}). "
                  f"Se pediran las cuotas a mano el resto de la sesion.")
            _odds_warned = True
        return None
```

Update the import line (existing line 47):

```python
from src.odds_api import DEFAULT_CACHE_MINUTES, MatchOdds, get_match_odds, resolve_allowed_bookmakers
```

(`DEFAULT_BOOKMAKER` is dropped from this import since it's no longer referenced anywhere in this file after this task's edits — verify with a project-wide grep before removing: `grep -n DEFAULT_BOOKMAKER src/value_analysis.py` should show zero remaining hits after Step 2 below.)

- [ ] **Step 2: Update the confirmation print line and thread bookmaker into the log calls**

Replace the confirmation block (existing lines 1052-1057):

```python
            # Cuotas automaticas (best-effort, nunca bloquea el flujo)
            auto_odds = try_auto_odds(tour, player_a, player_b)
            if auto_odds is not None:
                print(f"\n  Cuotas encontradas (A: {auto_odds.bookmaker_a}, "
                      f"B: {auto_odds.bookmaker_b}): "
                      f"{auto_odds.matched_home} vs {auto_odds.matched_away}")
```

After the odds-input block (existing lines 1104-1116, right after `val_b = calculate_value(...)`), compute the two bookmaker values to thread into logging — insert immediately after the existing `val_b = calculate_value(pred["p_b_cal"], odds_b)` line:

```python
        val_a = calculate_value(pred["p_a_cal"], odds_a)
        val_b = calculate_value(pred["p_b_cal"], odds_b)
        bookmaker_a = auto_odds.bookmaker_a if (auto_odds and odds_a_source == "auto") else ""
        bookmaker_b = auto_odds.bookmaker_b if (auto_odds and odds_b_source == "auto") else ""
```

Update the `log_query` call (existing lines 1147-1152):

```python
        if logged:
            log_query(
                tour, tournament, surface, match_date,
                player_a, player_b,
                pred, val_a, val_b, odds_a, odds_b,
                odds_a_source, odds_b_source,
                bookmaker_a=bookmaker_a, bookmaker_b=bookmaker_b,
            )
```

Update the `log_prediction_audit` call (existing lines 1167-1175):

```python
            log_prediction_audit(
                tour, tournament, surface, match_date,
                player_a, player_b,
                pred, val_a, val_b, odds_a, odds_b,
                odds_a_source, odds_b_source,
                decision=decision,
                model_snapshot_id=snapshot,
                shrink_hi=_SHRINK_HI, shrink_lo=_SHRINK_LO, shrink_rate=_SHRINK_RATE,
                bookmaker_a=bookmaker_a, bookmaker_b=bookmaker_b,
            )
```

Update the module docstring (existing lines 15-20) to drop the now-inaccurate "bookmaker fijo" description:

```python
Cuotas: si ODDS_API_KEY esta configurada en el entorno, se intenta
autocompletar la cuota de cada jugador via The Odds API — mejor precio
disponible entre los bookmakers permitidos (ALLOWED_BOOKMAKERS, o todos
si no esta configurada; ODDS_API_BOOKMAKER sigue forzando uno solo si se
define), cache de eventos configurable con ODDS_API_CACHE_MINUTES. Si no
hay key, no hay match, o falla la llamada, se pide la cuota a mano igual
que antes — el auto-fetch nunca bloquea el flujo.
```

- [ ] **Step 3: Manual verification**

Run: `pytest tests/test_value_analysis.py -v`
Expected: PASS (all tests — this task touches no test-covered surface directly, so this run is a regression check for the import/signature changes)

Then run: `python -c "import src.value_analysis"` to confirm the module still imports cleanly (catches the `DEFAULT_BOOKMAKER` import removal if anything else still referenced it).

- [ ] **Step 4: Commit**

```bash
git add src/value_analysis.py
git commit -m "$(cat <<'EOF'
feat(cli): interactive CLI shows and logs the real per-side bookmaker

try_auto_odds resolves an allow-list (ODDS_API_BOOKMAKER still forces a
single book when set) instead of a fixed default; the confirmation
message and both CSV logs now reflect the actual bookmaker_a/bookmaker_b
MatchOdds returned, which may differ between sides.

EOF
)"
```

---

## Task 15: full suite + final review

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -v`
Expected: All tests pass (509 pre-existing + ~25 new from Tasks 2-14). If any pre-existing test outside the files touched above fails, it means something else imports `DEFAULT_BOOKMAKER`, `find_match_odds`, `_extract_h2h_odds`, `log_query`, or `log_prediction_audit` with an assumption this plan didn't account for — grep for the failing symbol project-wide and fix the call site before proceeding.

- [ ] **Step 2: Grep for any remaining references to the old single-bookmaker default outside test files**

Run: `grep -rn "DEFAULT_BOOKMAKER" src/ scripts/`
Expected: Only the definition in `src/odds_api.py` and its use as the `resolve_allowed_bookmakers(explicit_bookmaker)` override value where `daily_scanner`/`value_analysis` pass an explicit `ODDS_API_BOOKMAKER`/`--bookmaker` value remain — no code path should read `DEFAULT_BOOKMAKER` as an unconditional fallback bookmaker anymore.

- [ ] **Step 3: Verify quota-cost risk (manual, not code)**

With a real `ODDS_API_KEY` set, run `python -m src.daily_scanner --days-ahead 1` once and check the response headers The Odds API returns (`x-requests-remaining`, `x-requests-used`) to confirm actual quota cost per multi-region call, per the open risk noted in the design spec. Record the finding as a follow-up note (not a code change) if quota pressure looks likely — out of scope for this plan.

- [ ] **Step 4: Final commit (if Step 3 turned up follow-up notes worth recording)**

Only if Step 3 produced something worth persisting — e.g. update the design spec's "Open risk" section with the measured quota cost:

```bash
git add docs/superpowers/specs/2026-08-21-multi-region-multi-bookmaker-design.md
git commit -m "$(cat <<'EOF'
docs(specs): record measured quota cost for multi-region odds requests

EOF
)"
```

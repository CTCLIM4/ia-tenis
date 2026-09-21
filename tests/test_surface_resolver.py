"""Tests for src/surface_resolver.py — surface detection from Odds API metadata."""
from __future__ import annotations

from src.surface_resolver import resolve_surface


class TestKeywordDetection:
    def test_keyword_in_sport_title(self):
        assert resolve_surface("Wimbledon - Grass Court Championships", "tennis_atp_wimbledon") == "grass"

    def test_keyword_in_sport_key(self):
        assert resolve_surface("Some Challenger Event", "tennis_atp_clay_challenger") == "clay"

    def test_keyword_case_insensitive(self):
        assert resolve_surface("WIMBLEDON GRASS CHAMPIONSHIPS", "tennis_atp_wimbledon") == "grass"

    def test_keyword_takes_priority_over_dictionary(self):
        # "madrid" resolves to clay via the dictionary, but an explicit "hard"
        # keyword in the title (e.g. a one-off indoor exhibition) must win.
        assert resolve_surface("Madrid Open - Hard Court Exhibition", "tennis_atp_madrid") == "hard"


class TestTournamentDictionaryFallback:
    def test_wimbledon_by_name_alone(self):
        assert resolve_surface("ATP Wimbledon", "tennis_atp_wimbledon") == "grass"

    def test_roland_garros(self):
        assert resolve_surface("ATP Roland Garros", "tennis_atp_french_open") == "clay"

    def test_kitzbuhel(self):
        assert resolve_surface("ATP Kitzbuhel", "tennis_atp_kitzbuhel") == "clay"

    def test_estoril(self):
        assert resolve_surface("ATP Estoril", "tennis_atp_estoril") == "clay"

    def test_hamburg(self):
        assert resolve_surface("WTA Hamburg", "tennis_wta_hamburg") == "clay"

    def test_prague(self):
        assert resolve_surface("WTA Prague", "tennis_wta_prague") == "hard"

    def test_washington_open(self):
        assert resolve_surface("ATP Washington Open", "tennis_atp_washington_open") == "hard"
        assert resolve_surface("WTA Washington Open", "tennis_wta_washington_open") == "hard"

    def test_guadalajara_open(self):
        # WTA 500, outdoor hard courts, Panamerican Tennis Center, Zapopan --
        # confirmed via wtatennis.com and Wikipedia 2026-09-18. Showed up
        # unresolved in the live Odds API feed (sport_key
        # "tennis_wta_guadalajara_open") during the 2026-09-18 WTA Guadalajara
        # Open, excluding it from the automated scan entirely.
        assert resolve_surface("WTA Guadalajara Open", "tennis_wta_guadalajara_open") == "hard"

    def test_singapore_open(self):
        # WTA 250, indoor hard courts, OCBC Arena -- confirmed via
        # wtatennis.com and Wikipedia 2026-09-21. Showed up unresolved in
        # the live Odds API feed ("WTA Singapore Open") during the
        # 2026-09-21 scan, excluding it from the automated scan entirely.
        assert resolve_surface("WTA Singapore Open", "tennis_wta_singapore_open") == "hard"

    def test_seoul_korea_open(self):
        # WTA 250, outdoor hard courts, Olympic Park Tennis Center --
        # confirmed via wtatennis.com and Wikipedia 2026-09-21 (Sept 21-27
        # 2026). Officially "Korea Open" but commonly referred to as
        # "Seoul" -- mapped under both names since it's unknown which one
        # a given Odds API feed uses for sport_title.
        assert resolve_surface("WTA Seoul", "tennis_wta_seoul") == "hard"
        assert resolve_surface("WTA Korea Open", "tennis_wta_korea_open") == "hard"


class TestFallSwingCoverage:
    """Proactive coverage for the Sept-Oct 2026 Asian hard-court swing --
    none of these were live in the Odds API feed yet as of 2026-09-18 (only
    Guadalajara was), but they start the following week (Sept 22-29) and
    would hit the exact same silent-exclusion gap Guadalajara did. Surfaces
    confirmed via atptour.com's official 2026 calendar PDF + Wikipedia
    (2026-09-18): Chengdu and Hangzhou (ATP 250, outdoor hard, Sept 23-29),
    Tokyo/Japan Open (ATP 500, outdoor hard, Sept 29-Oct 6), Beijing/China
    Open (ATP 500 + WTA 1000, outdoor hard, Sept 29-Oct 12), Wuhan (WTA
    1000, outdoor hard, Oct 12-18)."""

    def test_surface_resolver_covers_guadalajara(self):
        assert resolve_surface("WTA Guadalajara Open", "tennis_wta_guadalajara_open") == "hard"

    def test_chengdu(self):
        assert resolve_surface("ATP Chengdu", "tennis_atp_chengdu") == "hard"

    def test_hangzhou(self):
        assert resolve_surface("ATP Hangzhou", "tennis_atp_hangzhou") == "hard"

    def test_tokyo_japan_open(self):
        assert resolve_surface("ATP Tokyo", "tennis_atp_tokyo") == "hard"
        assert resolve_surface("ATP Japan Open", "tennis_atp_japan_open") == "hard"

    def test_beijing_china_open(self):
        assert resolve_surface("ATP Beijing", "tennis_atp_beijing") == "hard"
        assert resolve_surface("WTA China Open", "tennis_wta_china_open") == "hard"

    def test_wuhan(self):
        assert resolve_surface("WTA Wuhan Open", "tennis_wta_wuhan_open") == "hard"


class TestKnownAtpTournamentCoverage:
    def test_surface_resolver_covers_known_atp_tournaments(self):
        # Every entry currently in TOURNAMENT_SURFACES must actually resolve
        # via its own name -- a regression guard against a future edit that
        # adds a key but typos the lookup, or renames resolve_surface's
        # matching logic without re-checking the whole table.
        import src.surface_resolver as sr

        for tournament, expected_surface in sr.TOURNAMENT_SURFACES.items():
            title = f"ATP {tournament.title()}"
            assert resolve_surface(title, "tennis_atp_generic") == expected_surface, (
                f"{tournament!r} did not resolve to {expected_surface!r} via title {title!r}"
            )


class TestAllCurrentLiveTournaments:
    def test_surface_resolver_all_current_tournaments(self):
        # Re-checked live against The Odds API 2026-09-18 (Fase 3 of the
        # scanner-improvements session): only "WTA Guadalajara Open" was
        # active, and it already resolves. This is a portable, deterministic
        # regression pin of that live check (not a network call itself) --
        # re-run the live check in TestFallSwingCoverage's docstring
        # scenario whenever new tournaments start showing up unresolved.
        live_sports = [
            {"key": "tennis_wta_guadalajara_open", "title": "WTA Guadalajara Open"},
            {"key": "tennis_wta_singapore_open", "title": "WTA Singapore Open"},
            {"key": "tennis_wta_seoul", "title": "WTA Seoul"},
        ]
        unresolved = [s for s in live_sports if resolve_surface(s["title"], s["key"]) is None]
        assert unresolved == []


class TestUnknownSurface:
    def test_unknown_tournament_returns_none(self):
        assert resolve_surface("ATP Fictional Open 3000", "tennis_atp_fictional") is None

    def test_unknown_tournament_prints_warning(self, capsys):
        resolve_surface("ATP Fictional Open 3000", "tennis_atp_fictional")
        captured = capsys.readouterr()
        assert "Fictional Open 3000" in captured.out
        assert "superficie" in captured.out.lower()

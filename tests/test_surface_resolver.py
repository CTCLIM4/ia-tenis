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


class TestUnknownSurface:
    def test_unknown_tournament_returns_none(self):
        assert resolve_surface("ATP Fictional Open 3000", "tennis_atp_fictional") is None

    def test_unknown_tournament_prints_warning(self, capsys):
        resolve_surface("ATP Fictional Open 3000", "tennis_atp_fictional")
        captured = capsys.readouterr()
        assert "Fictional Open 3000" in captured.out
        assert "superficie" in captured.out.lower()

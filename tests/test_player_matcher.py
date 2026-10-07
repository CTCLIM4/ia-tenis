"""Tests for src/player_matcher.py — matching Odds API player names to the
canonical names known to the Elo/features dataset."""
from __future__ import annotations

from src.player_matcher import match_player_name


class TestExactMatch:
    def test_exact_name_returns_itself(self):
        canonical = ["Alexander Zverev", "Novak Djokovic"]
        assert match_player_name("Novak Djokovic", canonical) == "Novak Djokovic"


class TestAbbreviatedFormats:
    def test_first_initial_first_format(self):
        canonical = ["Alexander Zverev", "Novak Djokovic"]
        assert match_player_name("A. Zverev", canonical) == "Alexander Zverev"

    def test_surname_first_format(self):
        canonical = ["Aryna Sabalenka", "Iga Swiatek"]
        assert match_player_name("Sabalenka A.", canonical) == "Aryna Sabalenka"

    def test_accented_name_variant(self):
        canonical = ["Carlos Alcaraz Garfia"]
        assert match_player_name("Carlos Alcaraz Garfía", canonical) == "Carlos Alcaraz Garfia"


class TestDisambiguation:
    def test_same_surname_different_initial_picks_correct_sibling(self):
        # Zverev brothers: surname alone must not be enough.
        canonical = ["Alexander Zverev", "Mischa Zverev"]
        assert match_player_name("A. Zverev", canonical) == "Alexander Zverev"
        assert match_player_name("M. Zverev", canonical) == "Mischa Zverev"

    def test_same_surname_same_initial_is_ambiguous_returns_none(self):
        # Pliskova sisters: both first names start with "K" — a single
        # initial can't disambiguate, and guessing wrong is worse than
        # returning no match.
        canonical = ["Karolina Pliskova", "Kristyna Pliskova"]
        assert match_player_name("K. Pliskova", canonical) is None


class TestTruncatedFirstNameCollisions:
    """Real bug found 2026-09-21 scanning WTA Singapore Open: the dataset
    disambiguates same-initial collisions with a longer truncated first
    name ("Wang Xin.", "Wang Xiy.") instead of a bare single letter.
    normalize_name() strips the period before surname()/first_initial()
    ever see it, so a >1-char truncated token was indistinguishable from a
    real "Firstname Surname" full name -- "Wang Xin." parsed as
    surname="xin", initial="w", so "Xinyu Wang" (surname=wang, initial=x)
    matched NEITHER "Wang Xin." nor "Wang Xiy." on surname, and instead
    silently matched the unrelated, near-empty-sample "Wang X." (7
    matches) through the single-candidate path -- feeding the model a
    near-fresh player's Elo for a 233-match veteran and producing a
    bogus 27% edge that only got caught by the suspicious-edge halt."""

    def test_truncated_first_name_no_longer_misparsed_as_surname(self):
        canonical = ["Wang X.", "Wang Xin.", "Wang Xiy."]
        # Ambiguous by design once surname is parsed correctly (three
        # same-surname, same-initial candidates) -- refusing to guess is
        # correct here, matching the Pliskova-sisters precedent, and far
        # safer than the old silent wrong-match.
        assert match_player_name("Xinyu Wang", canonical) is None

    def test_truncated_first_name_still_matches_when_unambiguous(self):
        canonical = ["Wang Xin.", "Alcaraz Garfia"]
        assert match_player_name("Xinyu Wang", canonical) == "Wang Xin."


class TestNoMatch:
    def test_unknown_player_returns_none(self):
        canonical = ["Novak Djokovic", "Jannik Sinner"]
        assert match_player_name("Carlos Alcaraz", canonical) is None

    def test_empty_canonical_list_returns_none(self):
        assert match_player_name("Novak Djokovic", []) is None


class TestFullNameVariants:
    def test_extra_given_names_match_unique_complete_suffix(self):
        canonical = ["Coleman Wong", "Chun Hun Wong", "Wayne Wong"]
        assert match_player_name("Chak Lam Coleman Wong", canonical) == "Coleman Wong"

    def test_extra_given_names_refuse_overlapping_suffixes(self):
        canonical = ["Coleman Wong", "Lam Coleman Wong"]
        assert match_player_name("Chak Lam Coleman Wong", canonical) is None

    def test_surname_first_full_name_matches_unique_reverse(self):
        canonical = ["Yibing Wu", "Di Wu", "Tung-Lin Wu"]
        assert match_player_name("Wu Yibing", canonical) == "Yibing Wu"

    def test_surname_first_reverse_refuses_normalized_duplicates(self):
        canonical = ["Yibing Wu", "Yíbíng Wu"]
        assert match_player_name("Wu Yibing", canonical) is None

    def test_reverse_does_not_guess_from_shared_surname(self):
        assert match_player_name("Wu Yibing", ["Di Wu", "Tung-Lin Wu"]) is None


class TestKnownProblematicCases:
    """Verified against real behavior before writing these (see
    project_scanner_gaps_2026-09-18 memory) -- two of the four originally
    proposed cases already worked with zero changes needed."""

    def test_surname_initial_variants(self):
        # Already covered in spirit by TestAbbreviatedFormats -- pinned
        # again with the exact name from the task brief for traceability.
        canonical = ["Novak Djokovic", "Jannik Sinner"]
        assert match_player_name("N. Djokovic", canonical) == "Novak Djokovic"

    def test_accented_names(self):
        # normalize_name() (src/name_matching.py) already NFKD-strips
        # accents for both surname() and first_initial() -- already covered
        # in spirit by TestAbbreviatedFormats.test_accented_name_variant,
        # pinned again with the exact name from the task brief.
        canonical = ["Cristian Garin", "Novak Djokovic"]
        assert match_player_name("Cristian Garín", canonical) == "Cristian Garin"

    def test_davis_cup_names_use_the_same_matcher_no_special_case_needed(self):
        # Davis Cup ties are embedded in the same stats.tennismylife.org ATP
        # feed (tourney_level == "D") as regular tour matches -- same
        # winner_name/loser_name schema, same "First Last" format, verified
        # directly against real loaded data 2026-09-18 (e.g. "Holger Rune").
        # There is no separate Davis Cup naming convention to handle.
        canonical = ["Holger Rune", "Alexandar Lazarov", "Dimitar Kuzmanov"]
        assert match_player_name("H. Rune", canonical) == "Holger Rune"
        assert match_player_name("Kuzmanov D.", canonical) == "Dimitar Kuzmanov"

    def test_bare_nickname_does_not_match_by_design(self):
        # "Iga" -> "Iga Swiatek": deliberately NOT implemented. The Odds API
        # never actually sends bare first-name nicknames (verified: it uses
        # structured "First Last" / "Last F." formats, same as every other
        # case in this file) -- and matching on a first-name-only token
        # would mean guessing among every player who happens to share it,
        # which is exactly the class of wrong-guess risk
        # match_player_name's docstring says is worse than not matching.
        canonical = ["Iga Swiatek", "Novak Djokovic"]
        assert match_player_name("Iga", canonical) is None

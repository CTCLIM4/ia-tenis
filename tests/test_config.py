"""Tests for src/config.py — .env discovery and loading."""
from __future__ import annotations

import os

from dotenv import dotenv_values

import src.config as config


class TestDotenvPath:
    def test_resolves_to_repo_root(self):
        assert config.DOTENV_PATH == config._ROOT / ".env"

    def test_repo_root_matches_requirements_txt_location(self):
        # requirements.txt lives at the repo root .env should also live at.
        assert (config._ROOT / "requirements.txt").exists()


class TestThresholdConstants:
    def test_min_matches_threshold_is_25(self):
        assert config.MIN_MATCHES_THRESHOLD == 25

    def test_max_suspicious_edge_is_0_10(self):
        assert config.MAX_SUSPICIOUS_EDGE == 0.10


class TestDotenvLoading:
    def test_env_example_is_parseable_and_has_odds_api_key(self):
        example_path = config._ROOT / ".env.example"
        values = dotenv_values(example_path)
        assert "ODDS_API_KEY" in values

    def test_load_dotenv_sets_process_env(self, tmp_path, monkeypatch):
        # Exercises the same dotenv.load_dotenv call src.config makes at
        # import time, against a throwaway .env — proves the mechanism
        # works without depending on import-time side effects or mutating
        # the real process environment for other tests.
        from dotenv import load_dotenv

        monkeypatch.delenv("SMOKE_TEST_VAR", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text("SMOKE_TEST_VAR=hello\n", encoding="utf-8")

        load_dotenv(env_file)

        assert os.environ["SMOKE_TEST_VAR"] == "hello"
        monkeypatch.delenv("SMOKE_TEST_VAR", raising=False)

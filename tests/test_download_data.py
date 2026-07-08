from __future__ import annotations

import datetime

import pandas as pd

from scripts import download_data as dd


class TestFilterAtpManifest:
    def test_keeps_year_files_and_ongoing(self):
        files = [
            {"name": "1990.csv", "url": "u", "size": 100, "mtime": "t"},
            {"name": "2026.csv", "url": "u", "size": 200, "mtime": "t"},
            {"name": "ongoing_tourneys.csv", "url": "u", "size": 300, "mtime": "t"},
        ]
        result = dd._filter_atp_manifest(files)
        assert [f["name"] for f in result] == [
            "1990.csv", "2026.csv", "ongoing_tourneys.csv",
        ]

    def test_drops_challenger_quali_and_aggregate_files(self):
        files = [
            {"name": "2026_challenger.csv", "url": "u", "size": 1, "mtime": "t"},
            {"name": "atp_quali/2026_atp_quali.csv", "url": "u", "size": 1, "mtime": "t"},
            {"name": "challenger_ongoing_tourneys.csv", "url": "u", "size": 1, "mtime": "t"},
            {"name": "ATP_Database.csv", "url": "u", "size": 1, "mtime": "t"},
        ]
        assert dd._filter_atp_manifest(files) == []

    def test_schema_constant_has_fifty_columns(self):
        assert len(dd.ATP_SCHEMA_COLUMNS) == 50
        assert dd.ATP_SCHEMA_COLUMNS[0] == "tourney_id"
        assert dd.ATP_SCHEMA_COLUMNS[-1] == "l_bpFaced"


class TestShouldDownload:
    def test_missing_local_file(self, tmp_path):
        assert dd._should_download(tmp_path / "missing.csv", 100) is True

    def test_size_mismatch_triggers_download(self, tmp_path):
        f = tmp_path / "2020.csv"
        f.write_bytes(b"a" * 50)
        assert dd._should_download(f, 100) is True

    def test_size_match_skips_download(self, tmp_path):
        f = tmp_path / "2020.csv"
        f.write_bytes(b"a" * 100)
        assert dd._should_download(f, 100) is False

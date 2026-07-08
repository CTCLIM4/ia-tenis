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


class TestValidateAtpCsv:
    def test_valid_header_passes(self):
        header = ",".join(dd.ATP_SCHEMA_COLUMNS).encode("utf-8")
        data = header + b"\n2026-001,Test Open,Hard,32,A,O,20260101,1,,,,\n"
        assert dd._validate_atp_csv(data) is True

    def test_html_error_page_fails(self):
        data = b"<html><body>502 Bad Gateway</body></html>"
        assert dd._validate_atp_csv(data) is False

    def test_wrong_columns_fail(self):
        data = b"foo,bar,baz\n1,2,3\n"
        assert dd._validate_atp_csv(data) is False

    def test_empty_bytes_fail(self):
        assert dd._validate_atp_csv(b"") is False


class _FakeResponse:
    def __init__(self, content: bytes = b""):
        self.content = content

    def raise_for_status(self):
        pass


class TestDownloadFile:
    def _valid_body(self) -> bytes:
        header = ",".join(dd.ATP_SCHEMA_COLUMNS)
        return (header + "\n2026-001,Test,Hard,32,A,O,20260101,1\n").encode("utf-8")

    def test_successful_download_writes_dest_with_no_tmp_left_behind(self, tmp_path, monkeypatch):
        dest = tmp_path / "2020.csv"
        body = self._valid_body()
        monkeypatch.setattr(dd.requests, "get", lambda url, timeout=None: _FakeResponse(content=body))

        assert dd._download_file("http://x/2020.csv", dest) is True
        assert dest.read_bytes() == body
        assert not (tmp_path / "2020.csv.tmp").exists()

    def test_invalid_schema_leaves_existing_file_untouched(self, tmp_path, monkeypatch):
        dest = tmp_path / "2020.csv"
        dest.write_bytes(b"good,old,data\n1,2,3\n")
        monkeypatch.setattr(
            dd.requests, "get",
            lambda url, timeout=None: _FakeResponse(content=b"<html>error page</html>"),
        )

        assert dd._download_file("http://x/2020.csv", dest) is False
        assert dest.read_bytes() == b"good,old,data\n1,2,3\n"
        assert not (tmp_path / "2020.csv.tmp").exists()

    def test_network_error_leaves_existing_file_untouched(self, tmp_path, monkeypatch):
        dest = tmp_path / "2020.csv"
        dest.write_bytes(b"good,old,data\n1,2,3\n")

        def boom(url, timeout=None):
            raise ConnectionError("network down")

        monkeypatch.setattr(dd.requests, "get", boom)

        assert dd._download_file("http://x/2020.csv", dest) is False
        assert dest.read_bytes() == b"good,old,data\n1,2,3\n"

    def test_write_failure_after_validation_returns_false_and_cleans_up_tmp(self, tmp_path, monkeypatch):
        dest = tmp_path / "2020.csv"
        dest.write_bytes(b"good,old,data\n1,2,3\n")
        body = self._valid_body()
        monkeypatch.setattr(dd.requests, "get", lambda url, timeout=None: _FakeResponse(content=body))

        def boom(self, data):
            raise OSError("disk full")

        monkeypatch.setattr(dd.Path, "write_bytes", boom)

        assert dd._download_file("http://x/2020.csv", dest) is False
        assert dest.read_bytes() == b"good,old,data\n1,2,3\n"
        assert not (tmp_path / "2020.csv.tmp").exists()

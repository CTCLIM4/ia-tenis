from __future__ import annotations

import datetime
import stat

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


def _atp_row(**overrides) -> dict:
    row = {col: "" for col in dd.ATP_SCHEMA_COLUMNS}
    row.update(overrides)
    return row


class TestMergeOngoingIntoYear:
    def test_no_overlap_keeps_all_rows(self):
        year_df = pd.DataFrame([
            _atp_row(tourney_id="2026-580", tourney_name="Wimbledon", round="F",
                     match_num=1, winner_name="A"),
        ])
        ongoing_df = pd.DataFrame([
            _atp_row(tourney_id="2026-540", tourney_name="Wimbledon Q", round="R32",
                     match_num=5, winner_name="B"),
        ])
        merged = dd._merge_ongoing_into_year(year_df, ongoing_df)
        assert len(merged) == 2

    def test_conflicting_row_year_file_wins(self):
        year_df = pd.DataFrame([
            _atp_row(tourney_id="2026-540", tourney_name="Wimbledon", round="F",
                     match_num=1, winner_name="ARCHIVED_WINNER"),
        ])
        ongoing_df = pd.DataFrame([
            _atp_row(tourney_id="2026-540", tourney_name="Wimbledon", round="F",
                     match_num=1, winner_name="LIVE_WINNER"),
        ])
        merged = dd._merge_ongoing_into_year(year_df, ongoing_df)
        assert len(merged) == 1
        assert merged.iloc[0]["winner_name"] == "ARCHIVED_WINNER"

    def test_tourney_id_collision_across_different_tournaments_not_collapsed(self):
        # Real bug seen live in the API's 2026.csv: tourney_id "2026-416" is
        # reused by both Munich and Rome Masters, both with match_num=1.
        # (tourney_id, match_num) alone would wrongly collapse these into one
        # row; the composite key must not.
        year_df = pd.DataFrame([
            _atp_row(tourney_id="2026-416", tourney_name="Munich", round="R32",
                     match_num=1, winner_name="MUNICH_WINNER"),
        ])
        ongoing_df = pd.DataFrame([
            _atp_row(tourney_id="2026-416", tourney_name="Rome Masters", round="R128",
                     match_num=1, winner_name="ROME_WINNER"),
        ])
        merged = dd._merge_ongoing_into_year(year_df, ongoing_df)
        assert len(merged) == 2

    def test_result_column_order_matches_schema(self):
        year_df = pd.DataFrame([_atp_row(tourney_id="A", tourney_name="X", round="F", match_num=1)])
        ongoing_df = pd.DataFrame([_atp_row(tourney_id="B", tourney_name="Y", round="F", match_num=1)])
        merged = dd._merge_ongoing_into_year(year_df, ongoing_df)
        assert list(merged.columns) == dd.ATP_SCHEMA_COLUMNS


class TestCleanupLegacyGitClone:
    def test_removes_git_clone_artifacts_but_keeps_year_files(self, tmp_path):
        dest = tmp_path / "tennis_atp_tml"
        dest.mkdir()
        (dest / ".git").mkdir()
        (dest / ".git" / "HEAD").write_text("ref: refs/heads/master")
        (dest / ".github").mkdir()
        (dest / "README.md").write_text("readme")
        (dest / "logo.jpg").write_bytes(b"\xff\xd8")
        (dest / "ATP_Database.csv").write_text("old,aggregate")
        (dest / "ongoing_tourneys.csv").write_text("stale,ongoing")
        (dest / "2025.csv").write_text("keep,me")

        dd._cleanup_legacy_git_clone(dest)

        assert not (dest / ".git").exists()
        assert not (dest / ".github").exists()
        assert not (dest / "README.md").exists()
        assert not (dest / "logo.jpg").exists()
        assert not (dest / "ATP_Database.csv").exists()
        assert not (dest / "ongoing_tourneys.csv").exists()
        assert (dest / "2025.csv").exists()

    def test_noop_when_not_a_git_clone(self, tmp_path):
        dest = tmp_path / "tennis_atp_tml"
        dest.mkdir()
        (dest / "2025.csv").write_text("keep,me")

        dd._cleanup_legacy_git_clone(dest)

        assert (dest / "2025.csv").exists()

    def test_removes_readonly_git_objects_on_windows(self, tmp_path):
        # Git marks packed/loose objects read-only on Windows. Plain
        # shutil.rmtree() crashes with PermissionError on these; the
        # cleanup must tolerate and clear that bit.
        dest = tmp_path / "tennis_atp_tml"
        dest.mkdir()
        objects_dir = dest / ".git" / "objects" / "info"
        objects_dir.mkdir(parents=True)
        readonly_file = objects_dir / "commit-graph-chain"
        readonly_file.write_text("deadbeef")
        readonly_file.chmod(stat.S_IREAD)
        (dest / "2025.csv").write_text("keep,me")

        try:
            dd._cleanup_legacy_git_clone(dest)
        finally:
            # Safety net: if the test fails partway, don't leave a
            # read-only file behind to break tmp_path cleanup.
            if readonly_file.exists():
                readonly_file.chmod(stat.S_IWRITE)

        assert not (dest / ".git").exists()
        assert (dest / "2025.csv").exists()

    def test_idempotent_when_called_twice(self, tmp_path):
        dest = tmp_path / "tennis_atp_tml"
        dest.mkdir()
        (dest / ".git").mkdir()
        (dest / ".git" / "HEAD").write_text("ref: refs/heads/master")
        (dest / "2025.csv").write_text("keep,me")

        dd._cleanup_legacy_git_clone(dest)
        dd._cleanup_legacy_git_clone(dest)

        assert not (dest / ".git").exists()
        assert (dest / "2025.csv").exists()

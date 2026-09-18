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


class TestDownloadAtpOrchestration:
    def test_full_flow_downloads_merges_and_skips_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "ATP_DIR", tmp_path / "tennis_atp_tml")
        current_year = datetime.date.today().year
        current_year_name = f"{current_year}.csv"

        header = ",".join(dd.ATP_SCHEMA_COLUMNS)

        def make_row(**overrides):
            vals = {col: "" for col in dd.ATP_SCHEMA_COLUMNS}
            vals.update(overrides)
            return ",".join(str(vals[c]) for c in dd.ATP_SCHEMA_COLUMNS)

        year_csv = (header + "\n" + make_row(
            tourney_id="Y-1", tourney_name="Old Open", round="F",
            match_num=1, winner_name="OLD_WINNER") + "\n").encode("utf-8")
        ongoing_csv = (header + "\n" + make_row(
            tourney_id="Y-2", tourney_name="Live Open", round="R16",
            match_num=1, winner_name="LIVE_WINNER") + "\n").encode("utf-8")

        manifest_json = {
            "count": 3,
            "files": [
                {"name": "1990.csv", "url": "http://x/1990.csv", "size": 999999, "mtime": "t"},
                {"name": current_year_name, "url": "http://x/year.csv", "size": len(year_csv), "mtime": "t"},
                {"name": "ongoing_tourneys.csv", "url": "http://x/ongoing.csv", "size": len(ongoing_csv), "mtime": "t"},
            ],
        }

        class FakeManifestResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return manifest_json

        class FakeFileResponse:
            def __init__(self, content):
                self.content = content

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=None):
            if url == dd.ATP_MANIFEST_URL:
                return FakeManifestResponse()
            if url == "http://x/year.csv":
                return FakeFileResponse(year_csv)
            if url == "http://x/ongoing.csv":
                return FakeFileResponse(ongoing_csv)
            if url == "http://x/1990.csv":
                raise AssertionError("1990.csv should have been skipped: size already matches")
            raise AssertionError(f"unexpected url requested: {url}")

        monkeypatch.setattr(dd.requests, "get", fake_get)

        dd.ATP_DIR.mkdir(parents=True)
        (dd.ATP_DIR / "1990.csv").write_bytes(b"x" * 999999)

        dd.download_atp()

        result = pd.read_csv(dd.ATP_DIR / current_year_name)
        assert set(result["winner_name"]) == {"OLD_WINNER", "LIVE_WINNER"}
        assert (dd.ATP_DIR / "ongoing_tourneys.csv").exists()
        assert (dd.ATP_DIR / "1990.csv").stat().st_size == 999999

    def test_removes_legacy_git_clone_before_downloading(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "ATP_DIR", tmp_path / "tennis_atp_tml")
        dd.ATP_DIR.mkdir(parents=True)
        (dd.ATP_DIR / ".git").mkdir()
        (dd.ATP_DIR / "README.md").write_text("readme")

        def fake_get(url, timeout=None):
            class Empty:
                def raise_for_status(self_inner):
                    pass

                def json(self_inner):
                    return {"count": 0, "files": []}

            return Empty()

        monkeypatch.setattr(dd.requests, "get", fake_get)

        dd.download_atp()

        assert not (dd.ATP_DIR / ".git").exists()
        assert not (dd.ATP_DIR / "README.md").exists()


class TestDownloadAtpFailureIsolation:
    def test_wta_still_runs_when_atp_manifest_fetch_fails(self, tmp_path, monkeypatch):
        # The single most important property of download_atp(): an ATP-side
        # failure must never prevent the WTA section of download() from
        # running afterward.
        monkeypatch.setattr(dd, "DATA_RAW", tmp_path / "raw")
        monkeypatch.setattr(dd, "ATP_DIR", tmp_path / "raw" / "tennis_atp_tml")
        monkeypatch.setattr(dd, "WTA_DIR", tmp_path / "raw" / "tennis_wta_tduk")

        def boom(url, timeout=None):
            raise ConnectionError("network down")

        monkeypatch.setattr(dd.requests, "get", boom)
        monkeypatch.setattr(dd, "_fetch_wta_links", lambda: {})

        wta_years_called = []

        def fake_wta_year(year, discovered_url=None):
            wta_years_called.append(year)
            return True

        monkeypatch.setattr(dd, "_download_wta_year", fake_wta_year)

        dd.download()

        assert wta_years_called, "WTA section never ran after the ATP manifest fetch failed"
        assert wta_years_called[0] == dd.WTA_START_YEAR
        assert wta_years_called[-1] == datetime.date.today().year

    def test_download_failure_counted_as_failed_not_skipped(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(dd, "ATP_DIR", tmp_path / "tennis_atp_tml")

        manifest_json = {
            "count": 1,
            "files": [
                {"name": "2020.csv", "url": "http://x/2020.csv", "size": 999, "mtime": "t"},
            ],
        }

        class FakeManifestResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return manifest_json

        class FakeBadFileResponse:
            content = b"<html>502 Bad Gateway</html>"

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=None):
            if url == dd.ATP_MANIFEST_URL:
                return FakeManifestResponse()
            if url == "http://x/2020.csv":
                return FakeBadFileResponse()
            raise AssertionError(f"unexpected url requested: {url}")

        monkeypatch.setattr(dd.requests, "get", fake_get)

        dd.download_atp()

        assert not (dd.ATP_DIR / "2020.csv").exists()
        captured = capsys.readouterr()
        assert "0 downloaded, 0 already current or unchanged, 1 failed." in captured.out

    def test_merge_failure_is_caught_and_does_not_propagate(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(dd, "ATP_DIR", tmp_path / "tennis_atp_tml")
        dd.ATP_DIR.mkdir(parents=True)
        current_year_name = f"{datetime.date.today().year}.csv"

        # Both on-disk files are malformed (neither has any ATP_SCHEMA_COLUMNS),
        # simulating a stale legacy-clone file whose size coincidentally
        # matches the manifest so the download loop skips re-fetching it.
        year_bytes = b"malformed,data\n1,2\n"
        ongoing_bytes = b"also,malformed\n3,4\n"
        (dd.ATP_DIR / current_year_name).write_bytes(year_bytes)
        (dd.ATP_DIR / "ongoing_tourneys.csv").write_bytes(ongoing_bytes)

        manifest_json = {
            "count": 2,
            "files": [
                {"name": current_year_name, "url": "http://x/year.csv",
                 "size": len(year_bytes), "mtime": "t"},
                {"name": "ongoing_tourneys.csv", "url": "http://x/ongoing.csv",
                 "size": len(ongoing_bytes), "mtime": "t"},
            ],
        }

        class FakeManifestResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return manifest_json

        def fake_get(url, timeout=None):
            if url == dd.ATP_MANIFEST_URL:
                return FakeManifestResponse()
            raise AssertionError(
                f"unexpected download for {url}: both files should be skipped as unchanged"
            )

        monkeypatch.setattr(dd.requests, "get", fake_get)

        dd.download_atp()  # must not raise

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "merge" in captured.out.lower()

    def test_should_download_exception_is_isolated_and_loop_continues(self, tmp_path, monkeypatch, capsys):
        # A transient filesystem hiccup (antivirus lock, permissions blip -
        # the same bug class fixed in Task 3's atomic write and Task 5's
        # read-only .git cleanup) raised mid-loop must not abort the whole
        # download_atp() call, and entries after the failing one must still
        # be processed.
        monkeypatch.setattr(dd, "ATP_DIR", tmp_path / "tennis_atp_tml")

        manifest_json = {
            "count": 2,
            "files": [
                {"name": "1990.csv", "url": "http://x/1990.csv", "size": 100, "mtime": "t"},
                {"name": "1991.csv", "url": "http://x/1991.csv", "size": 200, "mtime": "t"},
            ],
        }

        class FakeManifestResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return manifest_json

        header = ",".join(dd.ATP_SCHEMA_COLUMNS)
        good_body = (header + "\n").encode("utf-8")

        class FakeFileResponse:
            def __init__(self, content):
                self.content = content

            def raise_for_status(self):
                pass

        def fake_get(url, timeout=None):
            if url == dd.ATP_MANIFEST_URL:
                return FakeManifestResponse()
            if url == "http://x/1991.csv":
                return FakeFileResponse(good_body)
            raise AssertionError(f"unexpected url requested: {url}")

        monkeypatch.setattr(dd.requests, "get", fake_get)

        real_should_download = dd._should_download

        def flaky_should_download(local_path, remote_size):
            if local_path.name == "1990.csv":
                raise PermissionError("locked by antivirus")
            return real_should_download(local_path, remote_size)

        monkeypatch.setattr(dd, "_should_download", flaky_should_download)

        dd.download_atp()  # must not raise

        # The failing entry (1990.csv) must not have been downloaded, but the
        # loop must have continued and successfully processed 1991.csv after it.
        assert not (dd.ATP_DIR / "1990.csv").exists()
        assert (dd.ATP_DIR / "1991.csv").exists()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "  ATP: 1 downloaded, 0 already current or unchanged, 1 failed." in captured.out


class _FakeHeadResponse:
    def __init__(self, content_length):
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeGetResponse:
    def __init__(self, data: bytes):
        self._data = data
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


def _wta_xlsx_body() -> bytes:
    return dd._XLSX_MAGIC + b"fake xlsx payload"


class TestWtaRemoteSize:
    def test_returns_content_length_when_present(self, monkeypatch):
        def fake_urlopen(req, timeout=None):
            assert req.get_method() == "HEAD"
            return _FakeHeadResponse(content_length=12345)

        monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)
        assert dd._wta_remote_size("http://x/2026w.xlsx") == 12345

    def test_returns_none_when_header_missing(self, monkeypatch):
        monkeypatch.setattr(
            dd.urllib.request, "urlopen",
            lambda req, timeout=None: _FakeHeadResponse(content_length=None),
        )
        assert dd._wta_remote_size("http://x/2026w.xlsx") is None

    def test_returns_none_on_network_error(self, monkeypatch):
        def boom(req, timeout=None):
            raise ConnectionError("network down")

        monkeypatch.setattr(dd.urllib.request, "urlopen", boom)
        assert dd._wta_remote_size("http://x/2026w.xlsx") is None


class TestFetchWtaLinks:
    """tennis-data.co.uk moved its WTA files under an opaque path prefix
    (seen live 2026-09-18: /hrjk-85HytOjkhth76j_ygh4jf7/{year}w/{year}.xlsx)
    that the old hardcoded {year}w/{year}w.xls-style patterns don't match.
    _fetch_wta_links() scrapes the live data page instead of guessing, so a
    future path change doesn't require another code fix.
    """

    _SAMPLE_HTML = """
    <a HREF="hrjk-85HytOjkhth76j_ygh4jf7/2025/2025.xlsx">2025 men</a>
    <a HREF="hrjk-85HytOjkhth76j_ygh4jf7/2025w/2025.xlsx">2025 women</a>
    <a HREF="hrjk-85HytOjkhth76j_ygh4jf7/2026/2026.xlsx">2026 men</a>
    <a HREF="hrjk-85HytOjkhth76j_ygh4jf7/2026w/2026.xlsx">2026 women</a>
    """

    def test_parses_women_links_and_ignores_men(self, monkeypatch):
        monkeypatch.setattr(
            dd.urllib.request, "urlopen",
            lambda req, timeout=None: _FakeGetResponse(self._SAMPLE_HTML.encode("utf-8")),
        )
        links = dd._fetch_wta_links()
        assert links == {
            2025: "https://www.tennis-data.co.uk/hrjk-85HytOjkhth76j_ygh4jf7/2025w/2025.xlsx",
            2026: "https://www.tennis-data.co.uk/hrjk-85HytOjkhth76j_ygh4jf7/2026w/2026.xlsx",
        }

    def test_returns_empty_dict_on_network_error(self, monkeypatch):
        def boom(req, timeout=None):
            raise ConnectionError("network down")

        monkeypatch.setattr(dd.urllib.request, "urlopen", boom)
        assert dd._fetch_wta_links() == {}

    def test_absolute_href_passed_through_unchanged(self, monkeypatch):
        html = '<a href="https://cdn.example.com/2026w/2026.xlsx">2026 women</a>'
        monkeypatch.setattr(
            dd.urllib.request, "urlopen",
            lambda req, timeout=None: _FakeGetResponse(html.encode("utf-8")),
        )
        links = dd._fetch_wta_links()
        assert links == {2026: "https://cdn.example.com/2026w/2026.xlsx"}


class TestDownloadWtaYearSizeComparison:
    """The known bug (memory: project-wta-download-size-bug): _download_wta_year
    skipped any year whose local file already existed, with no comparison
    against the remote — unlike the ATP path's _should_download. These tests
    pin the fixed behavior: mirror _should_download so the current year's
    file is re-fetched whenever its remote size changes, and only skipped
    when the size still matches.
    """

    def test_redownloads_when_remote_size_differs_from_local(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "WTA_DIR", tmp_path)
        local = tmp_path / "2026w.xlsx"
        local.write_bytes(dd._XLSX_MAGIC + b"stale old content")

        new_body = _wta_xlsx_body()

        def fake_urlopen(req, timeout=None):
            if req.get_method() == "HEAD":
                # Remote size differs from the stale local file's size.
                return _FakeHeadResponse(content_length=len(new_body) + 999)
            return _FakeGetResponse(new_body)

        monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

        assert dd._download_wta_year(2026) is True
        assert local.read_bytes() == new_body

    def test_skips_when_remote_size_matches_local(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "WTA_DIR", tmp_path)
        local = tmp_path / "2026w.xlsx"
        body = dd._XLSX_MAGIC + b"already current content"
        local.write_bytes(body)

        def fake_urlopen(req, timeout=None):
            if req.get_method() == "HEAD":
                return _FakeHeadResponse(content_length=len(body))
            raise AssertionError(
                "GET should never be called: remote size matches local, must skip"
            )

        monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

        assert dd._download_wta_year(2026) is True
        assert local.read_bytes() == body  # untouched

    def test_parity_with_atp_missing_file_always_downloads(self, tmp_path, monkeypatch):
        # Mirrors TestShouldDownload.test_missing_local_file: no local file
        # means no skip decision to make at all, same as the ATP path.
        monkeypatch.setattr(dd, "WTA_DIR", tmp_path)
        new_body = _wta_xlsx_body()

        head_calls = []

        def fake_urlopen(req, timeout=None):
            if req.get_method() == "HEAD":
                head_calls.append(req.full_url)
                return _FakeHeadResponse(content_length=len(new_body))
            return _FakeGetResponse(new_body)

        monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

        assert dd._download_wta_year(2026) is True
        assert (tmp_path / "2026w.xlsx").read_bytes() == new_body
        # No pre-existing file, so there is nothing to size-compare against.
        assert head_calls == []


class TestDownloadWtaYearDiscoveredUrl:
    """A discovered_url (from _fetch_wta_links) is the real, current URL
    scraped off the live site -- it must be tried before any of the
    hardcoded guessed patterns, which go stale whenever the site
    restructures its paths (as it did 2026-09-18)."""

    def test_discovered_url_used_before_guessed_patterns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "WTA_DIR", tmp_path)
        new_body = _wta_xlsx_body()
        discovered = "https://www.tennis-data.co.uk/hrjk-xyz/2026w/2026.xlsx"

        def fake_urlopen(req, timeout=None):
            if req.full_url != discovered:
                raise AssertionError(f"should not hit guessed pattern: {req.full_url}")
            return _FakeGetResponse(new_body)

        monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

        assert dd._download_wta_year(2026, discovered_url=discovered) is True
        assert (tmp_path / "2026w.xlsx").read_bytes() == new_body

    def test_falls_back_to_guessed_patterns_when_discovered_url_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "WTA_DIR", tmp_path)
        new_body = _wta_xlsx_body()
        discovered = "https://www.tennis-data.co.uk/hrjk-xyz/2026w/2026.xlsx"
        fallback = "http://www.tennis-data.co.uk/2026w/2026w.xls"

        def fake_urlopen(req, timeout=None):
            if req.full_url == discovered:
                raise ConnectionError("stale discovered link, site moved again")
            if req.full_url == fallback:
                return _FakeGetResponse(new_body)
            raise AssertionError(f"unexpected url: {req.full_url}")

        monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

        assert dd._download_wta_year(2026, discovered_url=discovered) is True
        # Destination filename is derived from the downloaded content's
        # detected format, not the guessed pattern's URL suffix.
        assert (tmp_path / "2026w.xlsx").read_bytes() == new_body

    def test_no_discovered_url_falls_back_to_guessed_patterns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dd, "WTA_DIR", tmp_path)
        new_body = _wta_xlsx_body()

        def fake_urlopen(req, timeout=None):
            return _FakeGetResponse(new_body)

        monkeypatch.setattr(dd.urllib.request, "urlopen", fake_urlopen)

        assert dd._download_wta_year(2026, discovered_url=None) is True

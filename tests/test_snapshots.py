"""Tests for src/data/snapshots.py — dataset snapshot creation, listing,
metadata, and integrity verification. No test touches the real
data/raw/, data/processed/, or data/snapshots/ — everything works against
tmp_path fixtures via monkeypatched module constants."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import src.data.snapshots as snapshots


class TestSha256File:
    def test_matches_hashlib_reference(self, tmp_path):
        path = tmp_path / "sample.txt"
        path.write_bytes(b"hello world" * 1000)
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert snapshots._sha256_file(path) == expected

    def test_different_content_gives_different_hash(self, tmp_path):
        path_a = tmp_path / "a.txt"
        path_b = tmp_path / "b.txt"
        path_a.write_bytes(b"content A")
        path_b.write_bytes(b"content B")
        assert snapshots._sha256_file(path_a) != snapshots._sha256_file(path_b)

    def test_handles_file_larger_than_chunk_size(self, tmp_path):
        path = tmp_path / "large.bin"
        path.write_bytes(b"x" * (snapshots._HASH_CHUNK_SIZE * 3 + 12345))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert snapshots._sha256_file(path) == expected


class TestFileRecord:
    def test_returns_sha256_and_bytes(self, tmp_path):
        path = tmp_path / "sample.txt"
        content = b"some content"
        path.write_bytes(content)
        record = snapshots._file_record(path)
        assert record == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }


@pytest.fixture()
def fake_data_dirs(tmp_path, monkeypatch):
    """Redirect snapshots.py's data-source and snapshot-root constants to an
    isolated tmp_path tree, and populate minimal ATP+WTA processed/raw
    fixtures."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    snapshot_root = tmp_path / "snapshots"
    monkeypatch.setattr(snapshots, "_RAW_DIR", raw_dir)
    monkeypatch.setattr(snapshots, "_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(snapshots, "SNAPSHOT_ROOT", snapshot_root)

    processed_dir.mkdir(parents=True)
    for tour in ("atp", "wta"):
        df = pd.DataFrame({
            "match_date": ["2026-07-10", "2026-07-15", "2026-07-20"],
            "winner": ["A", "B", "A"],
            "loser":  ["B", "A", "C"],
        })
        df.to_csv(processed_dir / f"{tour}_features.csv", index=False)

    (raw_dir / "tennis_atp_tml").mkdir(parents=True)
    (raw_dir / "tennis_atp_tml" / "2026.csv").write_text("winner_name,loser_name\nA,B\n")
    (raw_dir / "tennis_wta_tduk").mkdir(parents=True)
    (raw_dir / "tennis_wta_tduk" / "2026w.csv").write_text("Winner,Loser\nA,B\n")

    return {"raw_dir": raw_dir, "processed_dir": processed_dir, "snapshot_root": snapshot_root}


class TestCreateSnapshot:
    def test_creates_expected_directory_structure(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        assert snapshot_dir == fake_data_dirs["snapshot_root"] / "2026-07-24"
        assert (snapshot_dir / "metadata.json").exists()
        assert (snapshot_dir / "processed" / "atp_features.csv").exists()
        assert (snapshot_dir / "processed" / "wta_features.csv").exists()
        assert (snapshot_dir / "raw" / "tennis_atp_tml" / "2026.csv").exists()
        assert (snapshot_dir / "raw" / "tennis_wta_tduk" / "2026w.csv").exists()

    def test_defaults_snapshot_id_to_today(self, fake_data_dirs):
        from datetime import date
        snapshot_dir = snapshots.create_snapshot()
        assert snapshot_dir.name == date.today().isoformat()

    def test_metadata_has_correct_row_counts_and_last_match_date(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        with open(snapshot_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["snapshot_id"] == "2026-07-24"
        assert meta["tours"]["atp"]["n_rows"] == 3
        assert meta["tours"]["atp"]["last_match_date"] == "2026-07-20"
        assert meta["tours"]["wta"]["n_rows"] == 3

    def test_metadata_has_file_hashes(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        with open(snapshot_dir / "metadata.json") as f:
            meta = json.load(f)
        atp_files = meta["tours"]["atp"]["files"]
        assert "processed/atp_features.csv" in atp_files
        entry = atp_files["processed/atp_features.csv"]
        assert "sha256" in entry and len(entry["sha256"]) == 64
        assert entry["bytes"] == (snapshot_dir / "processed" / "atp_features.csv").stat().st_size

    def test_metadata_has_created_at_and_git_commit_keys(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        with open(snapshot_dir / "metadata.json") as f:
            meta = json.load(f)
        assert "created_at" in meta
        assert "git_commit" in meta  # may be None outside a git checkout, key must exist regardless

    def test_raises_on_duplicate_snapshot_id(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        with pytest.raises(snapshots.SnapshotExistsError):
            snapshots.create_snapshot(snapshot_id="2026-07-24")

    def test_raises_clean_error_when_processed_csv_missing(self, fake_data_dirs):
        (fake_data_dirs["processed_dir"] / "wta_features.csv").unlink()
        with pytest.raises(FileNotFoundError):
            snapshots.create_snapshot(snapshot_id="2026-07-24")
        assert not (fake_data_dirs["snapshot_root"] / "2026-07-24").exists()

    @pytest.mark.parametrize("bad_id", ["../escape", "..\\escape", "..", ".", "a/b", "a\\b"])
    def test_rejects_path_traversal_snapshot_ids(self, fake_data_dirs, bad_id):
        """snapshot_id flows straight into a directory path that's later
        shutil.rmtree'd on failure — must reject anything that could escape
        SNAPSHOT_ROOT, not just rely on the caller behaving."""
        with pytest.raises(ValueError):
            snapshots.create_snapshot(snapshot_id=bad_id)
        assert not fake_data_dirs["snapshot_root"].exists()

    def test_can_snapshot_a_single_tour(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24", tours=("atp",))
        assert (snapshot_dir / "processed" / "atp_features.csv").exists()
        assert not (snapshot_dir / "processed" / "wta_features.csv").exists()

    def test_skips_dotfiles_in_raw_directory(self, fake_data_dirs):
        """Regression: a stray .gitignore left over from the old git-clone
        ATP download method (pre stats.tennismylife.org API migration) got
        copied into a real snapshot and hashed into metadata.json alongside
        the real {year}.csv files — found 2026-07-25."""
        (fake_data_dirs["raw_dir"] / "tennis_atp_tml" / ".gitignore").write_text(
            "atp_matches_amateur.csv\nplayers_preopen.csv"
        )
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24", tours=("atp",))

        assert not (snapshot_dir / "raw" / "tennis_atp_tml" / ".gitignore").exists()
        meta = json.loads((snapshot_dir / "metadata.json").read_text())
        assert not any(".gitignore" in key for key in meta["tours"]["atp"]["files"])

    def test_skips_ds_store_and_unexpected_extensions_keeps_real_data_files(self, fake_data_dirs):
        (fake_data_dirs["raw_dir"] / "tennis_atp_tml" / ".DS_Store").write_bytes(b"\x00\x01\x02")
        (fake_data_dirs["raw_dir"] / "tennis_atp_tml" / "notes.txt").write_text("scratch notes")

        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24", tours=("atp",))

        raw_files = {p.name for p in (snapshot_dir / "raw" / "tennis_atp_tml").iterdir()}
        assert ".DS_Store" not in raw_files
        assert "notes.txt" not in raw_files
        assert "2026.csv" in raw_files  # the real fixture data file must still be copied


class TestDavisCupRawTourDir:
    """Davis Cup has no separate raw source -- tennis-data.co.uk doesn't
    have Davis Cup data at all (verified 2026-09-18), and Davis Cup ties are
    already embedded in the same stats.tennismylife.org ATP feed. So
    "davis" maps to the *same* raw directory as "atp", not a new one."""

    def test_davis_is_in_raw_tour_dirs(self):
        assert "davis" in snapshots._RAW_TOUR_DIRS

    def test_davis_maps_to_same_raw_dir_as_atp(self):
        assert snapshots._RAW_TOUR_DIRS["davis"] == snapshots._RAW_TOUR_DIRS["atp"]

    def test_can_snapshot_davis_alone(self, fake_data_dirs):
        (fake_data_dirs["processed_dir"] / "davis_features.csv").write_text(
            "match_date,winner,loser\n2026-07-10,A,B\n"
        )

        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24", tours=("davis",))

        assert (snapshot_dir / "processed" / "davis_features.csv").exists()
        assert (snapshot_dir / "raw" / "tennis_atp_tml" / "2026.csv").exists()
        meta = json.loads((snapshot_dir / "metadata.json").read_text())
        assert meta["tours"]["davis"]["n_rows"] == 1

    def test_can_snapshot_atp_and_davis_together_despite_shared_raw_dir(self, fake_data_dirs):
        # Regression: both tours resolve to the same _RAW_TOUR_DIRS entry
        # ("tennis_atp_tml"), so the raw destination directory gets created
        # (and populated) twice in one create_snapshot() call -- the second
        # mkdir must not raise FileExistsError.
        (fake_data_dirs["processed_dir"] / "davis_features.csv").write_text(
            "match_date,winner,loser\n2026-07-10,A,B\n"
        )

        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24", tours=("atp", "davis"))

        assert (snapshot_dir / "processed" / "atp_features.csv").exists()
        assert (snapshot_dir / "processed" / "davis_features.csv").exists()
        assert (snapshot_dir / "raw" / "tennis_atp_tml" / "2026.csv").exists()


class TestListSnapshots:
    def test_empty_when_no_snapshots_dir(self, fake_data_dirs):
        assert snapshots.list_snapshots() == []

    def test_lists_created_snapshots_sorted(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-20")
        snapshots.create_snapshot(snapshot_id="2026-07-10")
        assert snapshots.list_snapshots() == ["2026-07-10", "2026-07-20"]

    def test_ignores_directories_without_metadata_json(self, fake_data_dirs):
        fake_data_dirs["snapshot_root"].mkdir(parents=True)
        (fake_data_dirs["snapshot_root"] / "not-a-snapshot").mkdir()
        assert snapshots.list_snapshots() == []


class TestLoadSnapshotMetadata:
    def test_loads_written_metadata(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        meta = snapshots.load_snapshot_metadata("2026-07-24")
        assert meta["snapshot_id"] == "2026-07-24"

    def test_raises_for_unknown_snapshot(self, fake_data_dirs):
        with pytest.raises(snapshots.SnapshotNotFoundError):
            snapshots.load_snapshot_metadata("does-not-exist")


class TestResolveSnapshotPath:
    def test_processed_path(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        path = snapshots.resolve_snapshot_path("2026-07-24", "atp", "processed")
        assert path == fake_data_dirs["snapshot_root"] / "2026-07-24" / "processed" / "atp_features.csv"
        assert path.exists()

    def test_raw_path(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        path = snapshots.resolve_snapshot_path("2026-07-24", "wta", "raw")
        assert path == fake_data_dirs["snapshot_root"] / "2026-07-24" / "raw" / "tennis_wta_tduk"
        assert path.is_dir()

    def test_raises_for_unknown_snapshot(self, fake_data_dirs):
        with pytest.raises(snapshots.SnapshotNotFoundError):
            snapshots.resolve_snapshot_path("does-not-exist", "atp", "processed")

    def test_raises_for_unknown_kind(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        with pytest.raises(ValueError):
            snapshots.resolve_snapshot_path("2026-07-24", "atp", "bogus")


class TestVerifySnapshotIntegrity:
    def test_true_for_untouched_snapshot(self, fake_data_dirs):
        snapshots.create_snapshot(snapshot_id="2026-07-24")
        assert snapshots.verify_snapshot_integrity("2026-07-24") is True

    def test_false_when_file_content_tampered(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        target = snapshot_dir / "processed" / "atp_features.csv"
        target.write_text(target.read_text() + "\nTAMPERED,ROW,HERE")
        assert snapshots.verify_snapshot_integrity("2026-07-24") is False

    def test_false_when_file_missing(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24")
        (snapshot_dir / "processed" / "wta_features.csv").unlink()
        assert snapshots.verify_snapshot_integrity("2026-07-24") is False

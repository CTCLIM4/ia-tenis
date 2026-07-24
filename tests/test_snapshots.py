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

    def test_can_snapshot_a_single_tour(self, fake_data_dirs):
        snapshot_dir = snapshots.create_snapshot(snapshot_id="2026-07-24", tours=("atp",))
        assert (snapshot_dir / "processed" / "atp_features.csv").exists()
        assert not (snapshot_dir / "processed" / "wta_features.csv").exists()


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

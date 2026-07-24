"""Tests for src/data/snapshots.py — dataset snapshot creation, listing,
metadata, and integrity verification. No test touches the real
data/raw/, data/processed/, or data/snapshots/ — everything works against
tmp_path fixtures via monkeypatched module constants."""
from __future__ import annotations

import hashlib

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

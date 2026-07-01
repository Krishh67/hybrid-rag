"""Tests for content-based duplicate document detection.

Acceptance criteria covered:
    1 — Same file ingested twice → detected as duplicate on second run.
    2 — Same content with different filename → detected as duplicate.
    3 — Different content with same filename → NOT a duplicate.
    4 — Existing KB restart (manifest reload) still detects duplicates.
    5 — Manifest correctly stores and retrieves content hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from indexing.manifest import (
    _default_manifest,
    is_duplicate,
    load_manifest,
    save_manifest,
    update_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_fake_kb(
    kb_path: Path,
    doc_id: str,
    content_hash: str,
    filename: str,
    n_chunks: int = 3,
) -> None:
    """Write minimal parsed + embedding files for a fake document."""
    # parsed/
    parsed_dir = kb_path / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_data = {
        "doc_id": doc_id,
        "source_path": f"inputs/{filename}",
        "original_filename": filename,
        "file_type": "pdf",
        "content_hash": content_hash,
        "full_text": "Sample text",
        "structure_tier": 3,
        "headings": [],
        "title": None,
        "title_source": "none",
        "page_count": 1,
        "language": "en",
        "ocr_used": False,
        "parser_used": "test",
        "tables": [],
        "extraction_warnings": [],
        "extraction_errors": [],
        "metadata": {},
    }
    (parsed_dir / f"{doc_id}.json").write_text(
        json.dumps(parsed_data), encoding="utf-8"
    )

    # embeddings/
    emb_dir = kb_path / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    dim = 1024
    vecs = np.random.RandomState(42).randn(n_chunks, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(str(emb_dir / f"{doc_id}_dense.npy"), vecs)
    meta = [
        {
            "chunk_id": f"chunk-{i}",
            "dense_vector": vecs[i].tolist(),
            "sparse_vector": {"word": 0.5},
            "embedding_model": "BAAI/bge-m3",
            "embedding_model_version": "test",
            "embed_text_hash": f"hash-{i}",
            "dim": dim,
            "generated_at": "2024-01-01T00:00:00",
        }
        for i in range(n_chunks)
    ]
    (emb_dir / f"{doc_id}_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # chunks/
    chunks_dir = kb_path / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = [
        {
            "chunk_id": f"chunk-{i}",
            "parent_window_id": "pw-0",
            "segment_id": "seg-0",
            "chunk_text": f"Chunk {i} of {filename}",
            "embed_text": f"Title: Test | Chunk {i}",
            "char_start": i * 50,
            "char_end": i * 50 + 50,
            "token_count": 10,
            "metadata": {
                "doc_id": doc_id,
                "original_filename": filename,
                "title": "Test",
                "heading_path": [],
                "chunk_type": "text",
            },
        }
        for i in range(n_chunks)
    ]
    (chunks_dir / f"{doc_id}_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


# ---------------------------------------------------------------------------
# Criterion 5 — Manifest stores and retrieves content hashes correctly
# ---------------------------------------------------------------------------


class TestManifestHashStorage:
    def test_default_manifest_has_empty_hashes(self):
        m = _default_manifest("kb_test")
        assert "indexed_content_hashes" in m
        assert m["indexed_content_hashes"] == {}

    def test_update_manifest_records_content_hashes(self):
        m = _default_manifest("kb_test")
        hashes = {
            "aabbcc": {"doc_id": "doc-001", "original_filename": "paper.pdf"},
        }
        update_manifest(m, new_doc_ids=["doc-001"], new_chunk_count=5, new_content_hashes=hashes)

        assert "aabbcc" in m["indexed_content_hashes"]
        assert m["indexed_content_hashes"]["aabbcc"]["doc_id"] == "doc-001"

    def test_update_manifest_does_not_overwrite_existing_hash(self):
        m = _default_manifest("kb_test")
        original = {"aabbcc": {"doc_id": "doc-001", "original_filename": "paper.pdf"}}
        update_manifest(m, new_doc_ids=["doc-001"], new_chunk_count=5, new_content_hashes=original)

        # Try to overwrite with a different doc_id (should be ignored)
        overwrite_attempt = {"aabbcc": {"doc_id": "doc-999", "original_filename": "other.pdf"}}
        update_manifest(m, new_doc_ids=["doc-999"], new_chunk_count=2, new_content_hashes=overwrite_attempt)

        assert m["indexed_content_hashes"]["aabbcc"]["doc_id"] == "doc-001", (
            "Existing content hash must not be overwritten by a re-index"
        )

    def test_load_manifest_backfills_missing_hashes_key(self, tmp_path):
        """Older manifests without indexed_content_hashes are upgraded on load."""
        old_manifest = {
            "kb_id": "kb_old",
            "total_documents": 1,
            "total_chunks": 5,
            "embedding_model": "BAAI/bge-m3",
            "indexed_doc_ids": ["doc-001"],
        }
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        (index_dir / "manifest.json").write_text(
            json.dumps(old_manifest), encoding="utf-8"
        )
        m = load_manifest(index_dir, "kb_old")
        assert "indexed_content_hashes" in m
        assert m["indexed_content_hashes"] == {}


# ---------------------------------------------------------------------------
# is_duplicate helper
# ---------------------------------------------------------------------------


class TestIsDuplicate:
    def test_returns_false_for_empty_manifest(self):
        m = _default_manifest("kb_test")
        assert not is_duplicate(m, "somehash")

    def test_returns_true_for_known_hash(self):
        m = _default_manifest("kb_test")
        m["indexed_content_hashes"]["aabbcc"] = {"doc_id": "doc-001"}
        assert is_duplicate(m, "aabbcc")

    def test_returns_false_for_unknown_hash(self):
        m = _default_manifest("kb_test")
        m["indexed_content_hashes"]["aabbcc"] = {"doc_id": "doc-001"}
        assert not is_duplicate(m, "ddeeff")


# ---------------------------------------------------------------------------
# Criterion 1 — Same file indexed twice
# ---------------------------------------------------------------------------


class TestSameFileIndexedTwice:
    def test_second_run_adds_zero_chunks(self, tmp_path):
        """Criterion 1: re-indexing the same file produces 0 new chunks."""
        from indexing.pipeline import index_kb

        content_hash = _sha256(b"content of paper.pdf")
        _write_fake_kb(tmp_path, "doc-001", content_hash, "paper.pdf", n_chunks=5)

        # First run — fully indexes
        r1 = index_kb(str(tmp_path))
        assert r1.total_chunks == 5

        # Second run — same doc_id already in indexed_doc_ids, so 0 new
        r2 = index_kb(str(tmp_path))
        assert r2.new_chunks_added == 0
        assert r2.total_chunks == 5


# ---------------------------------------------------------------------------
# Criterion 2 — Same content, different filename
# ---------------------------------------------------------------------------


class TestSameContentDifferentFilename:
    def test_duplicate_detected_via_content_hash(self, tmp_path):
        """Criterion 2: same bytes under a different filename must be detected."""
        from indexing.pipeline import index_kb

        shared_bytes = b"identical content for both files"
        content_hash = _sha256(shared_bytes)

        # First document: paper.pdf
        _write_fake_kb(tmp_path, "doc-001", content_hash, "paper.pdf", n_chunks=4)
        r1 = index_kb(str(tmp_path))
        assert r1.total_chunks == 4

        # Load the manifest as main.py would
        index_dir = tmp_path / "index"
        manifest = load_manifest(index_dir, "kb_test")

        # Second document: paper_copy.pdf — same content_hash, different filename + doc_id
        assert is_duplicate(manifest, content_hash), (
            "Content with the same hash must be detected as a duplicate "
            "even with a different filename"
        )


# ---------------------------------------------------------------------------
# Criterion 3 — Different content, same filename
# ---------------------------------------------------------------------------


class TestDifferentContentSameFilename:
    def test_not_a_duplicate_when_content_differs(self, tmp_path):
        """Criterion 3: same filename but different bytes is NOT a duplicate."""
        from indexing.pipeline import index_kb

        hash_v1 = _sha256(b"version 1 content")
        hash_v2 = _sha256(b"version 2 content - different bytes")

        _write_fake_kb(tmp_path, "doc-001", hash_v1, "report.pdf", n_chunks=3)
        index_kb(str(tmp_path))

        index_dir = tmp_path / "index"
        manifest = load_manifest(index_dir, "kb_test")

        # Different content hash under the same "filename" — must NOT be a duplicate
        assert not is_duplicate(manifest, hash_v2), (
            "A different content_hash must never be treated as a duplicate, "
            "even if the filename matches a previously indexed file"
        )


# ---------------------------------------------------------------------------
# Criterion 4 — Manifest reload (simulated restart)
# ---------------------------------------------------------------------------


class TestManifestReloadAfterRestart:
    def test_duplicate_detected_after_simulated_restart(self, tmp_path):
        """Criterion 4: content hashes survive serialization → dedup works after restart."""
        from indexing.pipeline import index_kb

        content_hash = _sha256(b"persistent document content")
        _write_fake_kb(tmp_path, "doc-001", content_hash, "doc.pdf", n_chunks=3)
        index_kb(str(tmp_path))

        # Simulate process restart: load manifest fresh from disk
        index_dir = tmp_path / "index"
        manifest_after_restart = load_manifest(index_dir, "kb_test")

        assert is_duplicate(manifest_after_restart, content_hash), (
            "Content hash must survive a manifest save/load cycle so dedup "
            "works correctly after a process restart"
        )

    def test_manifest_hash_survives_roundtrip(self, tmp_path):
        """Content hashes written to manifest.json are identical when re-read."""
        index_dir = tmp_path / "index"
        m = _default_manifest("kb_test")
        m["indexed_content_hashes"]["deadbeef"] = {
            "doc_id": "doc-abc",
            "original_filename": "file.pdf",
        }
        save_manifest(m, index_dir)

        reloaded = load_manifest(index_dir, "kb_test")
        assert reloaded["indexed_content_hashes"].get("deadbeef") == {
            "doc_id": "doc-abc",
            "original_filename": "file.pdf",
        }

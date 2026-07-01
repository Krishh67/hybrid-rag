"""Tests for Stage 5 — Index & Storage.

Acceptance criteria covered:
    1  — Single document indexes successfully.
    2  — Multiple documents merge into one KB index.
    3  — FAISS row count equals metadata count.
    4  — Manifest counts are correct.
    5  — Incremental update adds only new vectors.
    6  — Index reload works after restart (FAISS persisted correctly).
    7  — BM25 reload works after restart.
    8  — Empty KB handled gracefully (no crash).
    9  — Corrupted embedding files fail with clear errors (logged, not crash).
    10 — All tests pass.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pytest

from indexing.bm25_builder import load_bm25, save_bm25, build_bm25
from indexing.faiss_builder import build_index, load_index, save_index
from indexing.manifest import load_manifest, save_manifest
from indexing.metadata_builder import load_metadata, save_metadata
from indexing.pipeline import IndexResult, index_kb
from indexing.schema import RetrievalRecord


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

DIM = 1024


def _rand_vecs(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, DIM).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_emb_meta(n: int, doc_id: str, prefix: str = "chunk") -> List[dict]:
    return [
        {
            "chunk_id": f"{prefix}-{i}",
            "dense_vector": [0.0] * DIM,  # not used by indexer
            "sparse_vector": {"word": 0.5},
            "embedding_model": "BAAI/bge-m3",
            "embedding_model_version": "test",
            "embed_text_hash": f"hash-{prefix}-{i}",
            "dim": DIM,
            "generated_at": "2024-01-01T00:00:00",
        }
        for i in range(n)
    ]


def _make_chunks_json(n: int, doc_id: str, prefix: str = "chunk") -> List[dict]:
    return [
        {
            "chunk_id": f"{prefix}-{i}",
            "parent_window_id": "pw-0",
            "segment_id": "seg-0",
            "chunk_text": f"This is chunk {prefix}-{i} text for document {doc_id}.",
            "embed_text": f"Title: Doc | Section | This is chunk {prefix}-{i}.",
            "char_start": i * 100,
            "char_end": i * 100 + 50,
            "token_count": 15,
            "metadata": {
                "doc_id": doc_id,
                "original_filename": f"{doc_id}.pdf",
                "title": "Test Document",
                "heading_path": ["Section One"],
                "chunk_type": "text",
            },
        }
        for i in range(n)
    ]


def _write_doc(kb_path: Path, doc_id: str, n: int, seed: int = 42) -> None:
    """Write fake embedding + chunk files for a single document."""
    prefix = doc_id[:8]
    vecs = _rand_vecs(n, seed=seed)
    meta = _make_emb_meta(n, doc_id, prefix=prefix)
    chunks = _make_chunks_json(n, doc_id, prefix=prefix)

    emb_dir = kb_path / "embeddings"
    chunks_dir = kb_path / "chunks"
    emb_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(emb_dir / f"{doc_id}_dense.npy"), vecs)
    with (emb_dir / f"{doc_id}_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    with (chunks_dir / f"{doc_id}_chunks.json").open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh)


# ---------------------------------------------------------------------------
# Criterion 1 — Single document
# ---------------------------------------------------------------------------


class TestSingleDocument:
    def test_indexes_successfully(self, tmp_path):
        """Criterion 1: single document indexes without errors."""
        _write_doc(tmp_path, "doc-001", n=10)
        result = index_kb(str(tmp_path))

        assert isinstance(result, IndexResult)
        assert result.total_chunks == 10
        assert result.total_documents == 1
        assert result.new_chunks_added == 10
        assert result.new_documents_added == 1
        assert not result.warnings

    def test_index_files_created(self, tmp_path):
        """After indexing, all four output files must exist."""
        _write_doc(tmp_path, "doc-001", n=5)
        result = index_kb(str(tmp_path))

        index_dir = result.index_dir
        assert (index_dir / "faiss.index").exists()
        assert (index_dir / "bm25.pkl").exists()
        assert (index_dir / "metadata.pkl").exists()
        assert (index_dir / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Criterion 2 — Multiple documents
# ---------------------------------------------------------------------------


class TestMultipleDocuments:
    def test_merges_into_one_index(self, tmp_path):
        """Criterion 2: two documents produce one KB-wide index."""
        _write_doc(tmp_path, "doc-001", n=8, seed=1)
        _write_doc(tmp_path, "doc-002", n=6, seed=2)
        result = index_kb(str(tmp_path))

        assert result.total_chunks == 14
        assert result.total_documents == 2


# ---------------------------------------------------------------------------
# Criterion 3 — FAISS row count == metadata count
# ---------------------------------------------------------------------------


class TestAlignmentInvariant:
    def test_faiss_equals_metadata_single_doc(self, tmp_path):
        """Criterion 3: FAISS.ntotal == len(metadata) after single doc."""
        _write_doc(tmp_path, "doc-001", n=12)
        result = index_kb(str(tmp_path))

        idx = load_index(result.index_dir)
        meta = load_metadata(result.index_dir)
        assert idx.ntotal == len(meta) == 12

    def test_faiss_equals_metadata_multi_doc(self, tmp_path):
        """Criterion 3: FAISS.ntotal == len(metadata) after multiple docs."""
        _write_doc(tmp_path, "doc-001", n=7, seed=1)
        _write_doc(tmp_path, "doc-002", n=9, seed=2)
        result = index_kb(str(tmp_path))

        idx = load_index(result.index_dir)
        meta = load_metadata(result.index_dir)
        assert idx.ntotal == len(meta) == 16


# ---------------------------------------------------------------------------
# Criterion 4 — Manifest counts
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_counts_are_correct(self, tmp_path):
        """Criterion 4: manifest.total_chunks and total_documents match reality."""
        _write_doc(tmp_path, "doc-001", n=5, seed=1)
        _write_doc(tmp_path, "doc-002", n=3, seed=2)
        result = index_kb(str(tmp_path))

        manifest_path = result.index_dir / "manifest.json"
        with manifest_path.open(encoding="utf-8") as fh:
            manifest = json.load(fh)

        assert manifest["total_chunks"] == 8
        assert manifest["total_documents"] == 2
        assert set(manifest["indexed_doc_ids"]) == {"doc-001", "doc-002"}


# ---------------------------------------------------------------------------
# Criterion 5 — Incremental update
# ---------------------------------------------------------------------------


class TestIncrementalUpdate:
    def test_incremental_adds_only_new_vectors(self, tmp_path):
        """Criterion 5: re-running after adding a new doc only indexes the new doc."""
        _write_doc(tmp_path, "doc-001", n=5, seed=1)
        result1 = index_kb(str(tmp_path))

        assert result1.new_chunks_added == 5

        # Add second document
        _write_doc(tmp_path, "doc-002", n=3, seed=2)
        result2 = index_kb(str(tmp_path))

        assert result2.new_chunks_added == 3
        assert result2.new_documents_added == 1
        assert result2.total_chunks == 8
        assert result2.total_documents == 2

    def test_rerun_without_new_docs_is_noop(self, tmp_path):
        """Criterion 5: re-running with no new docs reports 0 new chunks."""
        _write_doc(tmp_path, "doc-001", n=5, seed=1)
        index_kb(str(tmp_path))

        result2 = index_kb(str(tmp_path))
        assert result2.new_chunks_added == 0
        assert result2.new_documents_added == 0


# ---------------------------------------------------------------------------
# Criterion 6 — FAISS reload
# ---------------------------------------------------------------------------


class TestFaissReload:
    def test_index_reload_after_restart(self, tmp_path):
        """Criterion 6: loaded index produces same ntotal and correct dim."""
        _write_doc(tmp_path, "doc-001", n=10)
        result = index_kb(str(tmp_path))

        # Simulate a process restart by loading from disk
        loaded = load_index(result.index_dir)
        assert loaded.ntotal == 10
        assert loaded.d == DIM


# ---------------------------------------------------------------------------
# Criterion 7 — BM25 reload
# ---------------------------------------------------------------------------


class TestBm25Reload:
    def test_bm25_reload_after_restart(self, tmp_path):
        """Criterion 7: loaded BM25 state has same document count."""
        _write_doc(tmp_path, "doc-001", n=7)
        result = index_kb(str(tmp_path))

        loaded = load_bm25(result.index_dir)
        assert len(loaded.tokenized_corpus) == 7

    def test_bm25_scores_non_zero_for_matching_query(self, tmp_path):
        """BM25 scores are non-zero for a term that appears in some but not all docs."""
        _write_doc(tmp_path, "doc-001", n=5)
        result = index_kb(str(tmp_path))

        loaded = load_bm25(result.index_dir)
        # chunk texts contain "chunk-0", "chunk-1", ..., "chunk-4"
        # Query for "2" (from "chunk-2 text ...") — appears in only one doc
        # so IDF > 0 and BM25 score > 0 for that document.
        scores = loaded.model.get_scores(["2"])
        assert any(s > 0 for s in scores), (
            "Expected non-zero BM25 score for term '2' which appears in only one chunk"
        )


# ---------------------------------------------------------------------------
# Criterion 8 — Empty KB
# ---------------------------------------------------------------------------


class TestEmptyKB:
    def test_empty_kb_handled_gracefully(self, tmp_path):
        """Criterion 8: no embedding files → no crash, 0 chunks reported."""
        (tmp_path / "embeddings").mkdir(parents=True)
        result = index_kb(str(tmp_path))

        assert result.total_chunks == 0
        assert result.total_documents == 0
        assert result.new_chunks_added == 0


# ---------------------------------------------------------------------------
# Criterion 9 — Corrupted embedding files
# ---------------------------------------------------------------------------


class TestCorruptedFiles:
    def test_corrupted_meta_json_skipped_with_error_log(self, tmp_path, caplog):
        """Criterion 9: corrupt meta.json → doc skipped with logged error, no crash."""
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()

        vecs = _rand_vecs(3)
        np.save(str(emb_dir / "doc-bad_dense.npy"), vecs)
        (emb_dir / "doc-bad_meta.json").write_text("NOT VALID JSON", encoding="utf-8")

        result = index_kb(str(tmp_path))

        # Should not crash
        assert result.total_chunks == 0
        assert any("doc-bad" in w for w in result.warnings)

    def test_missing_npy_skipped_with_error_log(self, tmp_path, caplog):
        """Criterion 9: if meta.json exists but npy is missing → skipped gracefully."""
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()

        meta = _make_emb_meta(3, "doc-missing", prefix="m")
        (emb_dir / "doc-missing_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        # No corresponding _dense.npy written

        result = index_kb(str(tmp_path))
        assert result.total_chunks == 0

    def test_wrong_shape_npy_skipped(self, tmp_path):
        """Criterion 9: npy with wrong shape fails clearly."""
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()

        bad_vecs = np.ones((3, 512), dtype=np.float32)  # wrong dim
        np.save(str(emb_dir / "doc-wrongdim_dense.npy"), bad_vecs)
        meta = _make_emb_meta(3, "doc-wrongdim", prefix="wd")
        (emb_dir / "doc-wrongdim_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        result = index_kb(str(tmp_path))
        assert result.total_chunks == 0
        assert any("doc-wrongdim" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Global chunk ID alignment
# ---------------------------------------------------------------------------


class TestGlobalIdAlignment:
    def test_global_chunk_ids_are_sequential(self, tmp_path):
        """global_chunk_id must be 0, 1, 2, ... matching FAISS row order."""
        _write_doc(tmp_path, "doc-001", n=4, seed=1)
        _write_doc(tmp_path, "doc-002", n=3, seed=2)
        index_kb(str(tmp_path))

        meta = load_metadata(tmp_path / "index")
        for i, rec in enumerate(meta):
            assert rec.global_chunk_id == i, (
                f"Record at position {i} has global_chunk_id={rec.global_chunk_id}"
            )

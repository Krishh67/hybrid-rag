"""Tests for the full embedding pipeline — all mocked (no real BGE-M3).

Acceptance criteria covered:
    #1  — single chunk → EmbeddedChunk with dim=1024 and len(dense_vector)==1024
    #4  — two chunks with identical embed_text → cosine similarity > 0.99
    #5  — N chunks → exactly N EmbeddedChunks, .npy rows match meta.json
    #6  — table chunk and text chunk both succeed through same path
    #7  — empty/whitespace embed_text skipped with logged warning, no crash
    #10 — live-model integration test (skipped unless RUN_LIVE_MODEL_TESTS=1)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from chunking.schema import Chunk
from embedding.pipeline import embed_document
from embedding.schema import EmbeddedChunk


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "sample_chunks.json"


def load_fixture_chunks() -> List[Chunk]:
    raw = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return [Chunk.model_validate(r) for r in raw]


def _deterministic_encode(texts: List[str]) -> List[Tuple[List[float], Dict[str, float]]]:
    """Deterministic mock encoder: identical texts produce identical vectors."""
    results = []
    for text in texts:
        # Hash the text into a seed for reproducibility
        seed = sum(ord(c) for c in text) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(1024).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        vec = (vec / norm).tolist()
        sparse = {"token_a": 0.9, "token_b": 0.5}
        results.append((vec, sparse))
    return results


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Unit tests — criteria #1, #4, #5, #6, #7
# ---------------------------------------------------------------------------


class TestEmbedDocumentMocked:
    @pytest.fixture(autouse=True)
    def patch_encode(self, monkeypatch, tmp_path):
        """Inject the deterministic mock encoder into the pipeline module."""
        import embedding.pipeline as pipeline_mod

        monkeypatch.setattr(pipeline_mod, "encode_batch", _deterministic_encode)
        monkeypatch.setattr(pipeline_mod, "get_model_version", lambda: "mock-rev")
        self.kb_dir = str(tmp_path)

    # --- Criterion #1 ---
    def test_single_chunk_produces_embedded_chunk_with_correct_dim(self):
        chunks = load_fixture_chunks()[:1]
        results = embed_document(chunks, self.kb_dir, "test-doc")

        assert len(results) == 1
        ec = results[0]
        assert isinstance(ec, EmbeddedChunk)
        assert ec.dim == 1024
        assert len(ec.dense_vector) == 1024
        assert isinstance(ec.sparse_vector, dict)

    # --- Criterion #4 ---
    def test_identical_embed_text_produces_similar_vectors(self):
        chunks = load_fixture_chunks()
        # Make chunk 0 and chunk 1 have the same embed_text
        c0 = chunks[0]
        c1 = chunks[1].model_copy(
            update={"embed_text": c0.embed_text, "chunk_id": "dup-chunk"}
        )
        results = embed_document([c0, c1], self.kb_dir, "test-doc")

        assert len(results) == 2
        sim = _cosine_sim(results[0].dense_vector, results[1].dense_vector)
        assert sim > 0.99, f"Expected cosine sim > 0.99 for identical texts, got {sim:.4f}"

    # --- Criterion #5 ---
    def test_n_chunks_produce_n_outputs_with_aligned_npy(self):
        chunks = load_fixture_chunks()
        doc_id = "test-doc-align"
        results = embed_document(chunks, self.kb_dir, doc_id)

        assert len(results) == len(chunks)

        # Check .npy shape and alignment with meta.json
        npy_path = Path(self.kb_dir) / "embeddings" / f"{doc_id}_dense.npy"
        meta_path = Path(self.kb_dir) / "embeddings" / f"{doc_id}_meta.json"

        arr = np.load(str(npy_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        assert arr.shape == (len(chunks), 1024)
        assert len(meta) == len(chunks)

        for i, entry in enumerate(meta):
            np.testing.assert_allclose(
                arr[i], entry["dense_vector"], rtol=1e-5,
                err_msg=f"Row {i} of .npy does not match meta.json entry {i}"
            )

    # --- Criterion #6 ---
    def test_table_and_text_chunks_both_succeed(self):
        chunks = load_fixture_chunks()
        # Fixture has chunk_type=text and chunk_type=table — both must succeed
        types = {c.metadata.chunk_type for c in chunks}
        assert "table" in types, "Fixture must include a table chunk for this test"
        assert "text" in types, "Fixture must include a text chunk for this test"

        results = embed_document(chunks, self.kb_dir, "test-doc")
        result_ids = {r.chunk_id for r in results}

        for c in chunks:
            assert c.chunk_id in result_ids, (
                f"Chunk '{c.chunk_id}' (type={c.metadata.chunk_type}) missing from output"
            )

    # --- Criterion #7 ---
    def test_empty_embed_text_skipped_with_warning(self, caplog):
        chunks = load_fixture_chunks()
        # Inject an empty-embed_text chunk
        empty_chunk = chunks[0].model_copy(
            update={"embed_text": "   ", "chunk_id": "empty-chunk"}
        )
        all_chunks = [empty_chunk] + chunks[1:]

        results = embed_document(all_chunks, self.kb_dir, "test-doc")

        result_ids = {r.chunk_id for r in results}
        assert "empty-chunk" not in result_ids, "Empty embed_text chunk must be skipped"
        assert any("empty" in r.message.lower() or "whitespace" in r.message.lower()
                   for r in caplog.records), "Expected a logged warning about empty embed_text"

    def test_empty_chunk_list_returns_empty(self):
        results = embed_document([], self.kb_dir, "test-doc-empty")
        assert results == []


# ---------------------------------------------------------------------------
# Live-model integration test — Criterion #10
# Skipped unless RUN_LIVE_MODEL_TESTS=1 in environment
# ---------------------------------------------------------------------------

LIVE = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MODEL_TESTS", "0") != "1",
    reason="Live model tests are skipped by default. Set RUN_LIVE_MODEL_TESTS=1 to run.",
)


@LIVE
class TestLiveModel:
    """Criterion #10: real BGE-M3 inference sanity check.

    Loads the real model once.  Two near-duplicate texts should have a higher
    dense-cosine-similarity than two unrelated texts.  A distinctive rare token
    should produce a non-trivial sparse weight.
    """

    @pytest.fixture(scope="class", autouse=True)
    def load_model_once(self):
        from embedding.model_wrapper import get_model, reset_singleton_for_testing

        reset_singleton_for_testing()
        get_model()  # force load

    def test_near_duplicate_similarity_higher_than_unrelated(self, tmp_path):
        from embedding.model_wrapper import encode_batch

        near_dup_a = "Mobile learning helps language acquisition in modern classrooms."
        near_dup_b = "Language learning benefits from mobile technology in classrooms."
        unrelated = "The stock market fell sharply due to inflation concerns last quarter."

        results = encode_batch([near_dup_a, near_dup_b, unrelated])
        assert all(r is not None for r in results)

        d_a, _ = results[0]
        d_b, _ = results[1]
        d_u, _ = results[2]

        sim_near = _cosine_sim(d_a, d_b)
        sim_unrelt = _cosine_sim(d_a, d_u)

        assert sim_near > sim_unrelt, (
            f"Near-duplicate similarity ({sim_near:.4f}) should exceed "
            f"unrelated similarity ({sim_unrelt:.4f})"
        )

    def test_rare_token_has_nontrivial_sparse_weight(self, tmp_path):
        from embedding.model_wrapper import encode_batch

        # 'photosynthesis' is distinctive and should appear in sparse output
        text = "Photosynthesis converts light energy into chemical energy in plants."
        results = encode_batch([text])
        assert results[0] is not None

        _, sparse = results[0]
        assert len(sparse) > 0, "Sparse vector should not be empty for real text"

        # At least one token with weight > 0.1 must be present
        significant = {k: v for k, v in sparse.items() if v > 0.1}
        assert significant, "Expected at least one token with non-trivial sparse weight"

    def test_single_chunk_full_pipeline_live(self, tmp_path):
        """Full pipeline path with a real model — single chunk round-trip."""
        from embedding.pipeline import embed_document

        chunks = load_fixture_chunks()[:1]
        results = embed_document(chunks, str(tmp_path), "live-test-doc")

        assert len(results) == 1
        ec = results[0]
        assert ec.dim == 1024
        assert len(ec.dense_vector) == 1024
        assert len(ec.sparse_vector) > 0

"""Tests for embedding.cache — load/merge/persist logic.

Acceptance criteria covered:
    #2 — full cache hit → zero encode_batch calls
    #3 — partial miss → only changed chunk re-embedded
    #9 — round-trip .npy + meta.json preserves order and data
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from embedding.cache import load_cache, merge_and_save
from embedding.schema import EmbeddedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedded_chunk(chunk_id: str, h: str, dim: int = 1024) -> EmbeddedChunk:
    vec = [float(i % dim) / dim for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    vec = [x / norm for x in vec]
    return EmbeddedChunk(
        chunk_id=chunk_id,
        dense_vector=vec,
        sparse_vector={"hello": 0.8, "world": 0.4},
        embedding_model="BAAI/bge-m3",
        embedding_model_version="test-rev",
        embed_text_hash=h,
        dim=dim,
        generated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# load_cache
# ---------------------------------------------------------------------------


class TestLoadCache:
    def test_returns_empty_dict_if_file_missing(self, tmp_path):
        cache = load_cache(tmp_path / "nonexistent_meta.json")
        assert cache == {}

    def test_loads_existing_meta(self, tmp_path):
        ec = _make_embedded_chunk("cid1", "hash_aaa")
        meta_path = tmp_path / "doc_meta.json"
        meta_path.write_text(
            json.dumps([ec.model_dump(mode="json")], default=str),
            encoding="utf-8",
        )

        cache = load_cache(meta_path)
        assert "hash_aaa" in cache
        assert cache["hash_aaa"].chunk_id == "cid1"

    def test_returns_empty_on_corrupt_json(self, tmp_path, caplog):
        meta_path = tmp_path / "corrupt_meta.json"
        meta_path.write_text("NOT VALID JSON", encoding="utf-8")

        cache = load_cache(meta_path)
        assert cache == {}
        assert any("Could not parse" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# merge_and_save
# ---------------------------------------------------------------------------


class TestMergeAndSave:
    """Criterion #9 and alignment guarantees."""

    def test_writes_npy_and_meta(self, tmp_path):
        ec1 = _make_embedded_chunk("c1", "h1")
        ec2 = _make_embedded_chunk("c2", "h2")

        ordered = merge_and_save(
            chunk_ids=["c1", "c2"],
            chunk_hashes=["h1", "h2"],
            new_results={"h1": ec1, "h2": ec2},
            cached={},
            out_dir=tmp_path,
            doc_id="doc-test",
        )

        npy_path = tmp_path / "doc-test_dense.npy"
        meta_path = tmp_path / "doc-test_meta.json"

        assert npy_path.exists()
        assert meta_path.exists()
        assert len(ordered) == 2

    def test_npy_row_order_matches_meta_order(self, tmp_path):
        """Criterion #9: .npy rows and meta.json entries are strictly aligned."""
        ec1 = _make_embedded_chunk("c1", "h1")
        ec2 = _make_embedded_chunk("c2", "h2")

        merge_and_save(
            chunk_ids=["c1", "c2"],
            chunk_hashes=["h1", "h2"],
            new_results={"h1": ec1, "h2": ec2},
            cached={},
            out_dir=tmp_path,
            doc_id="doc-test",
        )

        arr = np.load(str(tmp_path / "doc-test_dense.npy"))
        with open(tmp_path / "doc-test_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        assert arr.shape == (2, 1024)
        for i, entry in enumerate(meta):
            # Each dense row must match the meta entry's dense_vector
            np.testing.assert_allclose(arr[i], entry["dense_vector"], rtol=1e-5)

    def test_cached_and_new_merged_in_order(self, tmp_path):
        """Mixing cached + new results preserves original chunk order."""
        ec1 = _make_embedded_chunk("c1", "h1")
        ec2 = _make_embedded_chunk("c2", "h2")
        ec3 = _make_embedded_chunk("c3", "h3")

        # c2 is cached from a previous run; c1 and c3 are freshly encoded
        ordered = merge_and_save(
            chunk_ids=["c1", "c2", "c3"],
            chunk_hashes=["h1", "h2", "h3"],
            new_results={"h1": ec1, "h3": ec3},
            cached={"h2": ec2},
            out_dir=tmp_path,
            doc_id="doc-test",
        )

        assert [ec.chunk_id for ec in ordered] == ["c1", "c2", "c3"]

    def test_chunk_with_no_embedding_omitted(self, tmp_path):
        """If a chunk has no entry in new or cached, it is omitted (not zero-padded)."""
        ec1 = _make_embedded_chunk("c1", "h1")
        # c2 never got encoded and is not in cache

        ordered = merge_and_save(
            chunk_ids=["c1", "c2"],
            chunk_hashes=["h1", "h_missing"],
            new_results={"h1": ec1},
            cached={},
            out_dir=tmp_path,
            doc_id="doc-test",
        )

        assert len(ordered) == 1
        assert ordered[0].chunk_id == "c1"


# ---------------------------------------------------------------------------
# Criterion #2 and #3: cache hit/miss behaviour via pipeline integration
# These are integration-level cache tests using a mocked encode_batch.
# ---------------------------------------------------------------------------


class TestCacheHitMiss:
    """Criterion #2 (full hit) and #3 (partial miss) tested via pipeline.embed_document."""

    def _load_sample_chunks(self) -> list:
        import json
        from pathlib import Path
        from chunking.schema import Chunk

        fixtures = Path(__file__).parent / "fixtures" / "sample_chunks.json"
        raw = json.loads(fixtures.read_text(encoding="utf-8"))
        return [Chunk.model_validate(r) for r in raw]

    def _make_encode_batch_mock(self, dim: int = 1024):
        """Return a mock encode_batch that produces deterministic fake embeddings."""
        import numpy as np
        from unittest.mock import MagicMock

        call_count = 0

        def fake_encode(texts):
            nonlocal call_count
            call_count += 1
            results = []
            for text in texts:
                vec = np.zeros(dim, dtype=np.float32)
                vec[0] = 1.0  # already normalized
                sparse = {"sample": 0.9, "token": 0.5}
                results.append((vec.tolist(), sparse))
            return results

        return fake_encode, lambda: call_count

    def test_full_cache_hit_zero_encode_calls(self, tmp_path, monkeypatch):
        """Criterion #2: unchanged chunks → encode_batch never called."""
        import embedding.pipeline as pipeline_mod

        chunks = self._load_sample_chunks()
        fake_encode, get_count = self._make_encode_batch_mock()

        monkeypatch.setattr(pipeline_mod, "encode_batch", fake_encode)
        monkeypatch.setattr(pipeline_mod, "get_model_version", lambda: "test-rev")

        # First run — populates cache
        pipeline_mod.embed_document(chunks, str(tmp_path), "test-doc-001")
        calls_after_first = get_count()

        # Second run — everything should be cached
        pipeline_mod.embed_document(chunks, str(tmp_path), "test-doc-001")
        calls_after_second = get_count()

        assert calls_after_second == calls_after_first, (
            "No new encode_batch calls expected on a full-cache-hit re-run"
        )

    def test_partial_cache_miss_only_changed_chunk_reencoded(self, tmp_path, monkeypatch):
        """Criterion #3: changing one chunk's embed_text triggers re-embedding for that chunk only."""
        import copy
        import embedding.pipeline as pipeline_mod

        chunks = self._load_sample_chunks()
        fake_encode, get_count = self._make_encode_batch_mock()

        monkeypatch.setattr(pipeline_mod, "encode_batch", fake_encode)
        monkeypatch.setattr(pipeline_mod, "get_model_version", lambda: "test-rev")

        # First run
        pipeline_mod.embed_document(chunks, str(tmp_path), "test-doc-001")
        calls_after_first = get_count()

        # Modify only the first chunk's embed_text
        modified_chunks = [c.model_copy(deep=True) for c in chunks]
        modified_chunks[0] = modified_chunks[0].model_copy(
            update={"embed_text": modified_chunks[0].embed_text + " MODIFIED"}
        )

        pipeline_mod.embed_document(modified_chunks, str(tmp_path), "test-doc-001")
        calls_after_second = get_count()

        # Exactly one more batch call (for the single changed chunk)
        assert calls_after_second == calls_after_first + 1, (
            "Expected exactly one more encode_batch call for the one changed chunk"
        )

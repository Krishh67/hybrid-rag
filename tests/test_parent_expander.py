import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from retrieval.schema import RetrievalCandidate, FinalContext, RetrievalResult
from retrieval.parent_expander import ParentExpander, ContextAssembler
import retrieval.config as config


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_candidate(
    chunk_id: str,
    parent_window_id: str,
    doc_id: str = "doc-1",
    rerank_score: float = -3.0,
    rrf_score: float = 0.05,
    matched_queries=None,
    retrieval_sources=None,
):
    return RetrievalCandidate(
        chunk_id=chunk_id,
        doc_id=doc_id,
        parent_window_id=parent_window_id,
        segment_id="seg-0",
        chunk_text=f"Text for {chunk_id}",
        dense_rank=1,
        bm25_rank=1,
        rrf_score=rrf_score,
        matched_queries=matched_queries or ["q1"],
        retrieval_sources=retrieval_sources or ["dense"],
        rerank_score=rerank_score,
        rerank_rank=1,
    )


@pytest.fixture
def mock_resolver():
    resolver = MagicMock()
    # Default: returns parent text for any id except "none"
    resolver.resolve.side_effect = lambda pid: (
        None if pid == "none" else f"Parent text for {pid}"
    )
    return resolver


# ─── ParentExpander Tests ─────────────────────────────────────────────────────

def test_deduplication_by_parent_window_id(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    # chunk-a and chunk-b share parent-X
    candidates = [
        make_candidate("chunk-a", "parent-X", rerank_score=-1.0, rrf_score=0.10),
        make_candidate("chunk-b", "parent-X", rerank_score=-2.0, rrf_score=0.08),
        make_candidate("chunk-c", "parent-Y", rerank_score=-3.0, rrf_score=0.06),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    assert len(results) == 2
    assert {r.parent_window_id for r in results} == {"parent-X", "parent-Y"}


def test_merges_source_chunks(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    candidates = [
        make_candidate("chunk-a", "parent-X"),
        make_candidate("chunk-b", "parent-X"),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    assert len(results) == 1
    ctx = results[0]
    assert "chunk-a" in ctx.source_chunks
    assert "chunk-b" in ctx.source_chunks


def test_keeps_best_rerank_score(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    candidates = [
        make_candidate("chunk-a", "parent-X", rerank_score=-1.0),
        make_candidate("chunk-b", "parent-X", rerank_score=-5.0),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    assert results[0].best_rerank_score == -1.0


def test_keeps_best_rrf_score(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    candidates = [
        make_candidate("chunk-a", "parent-X", rrf_score=0.15),
        make_candidate("chunk-b", "parent-X", rrf_score=0.05),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    assert results[0].best_rrf_score == 0.15


def test_merges_matched_queries_and_sources(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    candidates = [
        make_candidate("chunk-a", "parent-X", matched_queries=["q1"], retrieval_sources=["dense"]),
        make_candidate("chunk-b", "parent-X", matched_queries=["q2"], retrieval_sources=["bm25"]),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    assert set(results[0].matched_queries) == {"q1", "q2"}
    assert set(results[0].retrieval_sources) == {"dense", "bm25"}


def test_sorted_by_best_rerank_score_desc(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    candidates = [
        make_candidate("chunk-a", "parent-X", rerank_score=-5.0),
        make_candidate("chunk-b", "parent-Y", rerank_score=-1.0),
        make_candidate("chunk-c", "parent-Z", rerank_score=-3.0),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    scores = [r.best_rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].parent_window_id == "parent-Y"


def test_max_contexts_limit(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 2)

    candidates = [
        make_candidate("chunk-a", "parent-X", rerank_score=-1.0),
        make_candidate("chunk-b", "parent-Y", rerank_score=-2.0),
        make_candidate("chunk-c", "parent-Z", rerank_score=-3.0),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates, max_contexts=2)

    assert len(results) == 2
    assert results[0].parent_window_id == "parent-X"
    assert results[1].parent_window_id == "parent-Y"


def test_skips_unresolvable_parents(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    # resolver returns None for parent-X
    mock_resolver.resolve.side_effect = lambda pid: (
        None if pid == "parent-X" else f"text for {pid}"
    )

    candidates = [
        make_candidate("chunk-a", "parent-X", rerank_score=-1.0),
        make_candidate("chunk-b", "parent-Y", rerank_score=-2.0),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    assert len(results) == 1
    assert results[0].parent_window_id == "parent-Y"


def test_ranks_are_contiguous_after_skips(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    # Skip parent-X
    mock_resolver.resolve.side_effect = lambda pid: (
        None if pid == "parent-X" else f"text for {pid}"
    )

    candidates = [
        make_candidate("chunk-a", "parent-X", rerank_score=-1.0),
        make_candidate("chunk-b", "parent-Y", rerank_score=-2.0),
        make_candidate("chunk-c", "parent-Z", rerank_score=-3.0),
    ]

    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    ranks = [r.rank for r in results]
    assert ranks == list(range(1, len(ranks) + 1))


def test_parent_text_is_resolved(mock_resolver, monkeypatch):
    monkeypatch.setattr(config, "MAX_CONTEXTS", 10)

    candidates = [make_candidate("chunk-a", "parent-X")]
    expander = ParentExpander(mock_resolver)
    results = expander.expand(candidates)

    assert results[0].parent_text == "Parent text for parent-X"


# ─── ContextAssembler Integration Test ───────────────────────────────────────

def test_context_assembler_returns_retrieval_result(tmp_path, monkeypatch):
    """Integration test: ContextAssembler builds a valid RetrievalResult."""

    # Create the minimal KB structure
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()

    # Write a fake parent_lookup.pkl
    import pickle
    lookup = {
        "parent-X": {"doc_id": "doc-1", "char_start": 0, "char_end": 50},
    }
    with open(index_dir / "parent_lookup.pkl", "wb") as f:
        pickle.dump(lookup, f)

    # Write a fake parsed document
    doc = {"full_text": "A" * 100}
    with open(parsed_dir / "doc-1.json", "w") as f:
        json.dump(doc, f)

    candidates = [
        make_candidate("chunk-a", "parent-X", rerank_score=-1.5, rrf_score=0.07),
    ]

    assembler = ContextAssembler(str(tmp_path))
    result = assembler.assemble(
        query="test query",
        top_child_chunks=candidates,
        max_contexts=5,
    )

    assert isinstance(result, RetrievalResult)
    assert result.query == "test query"
    assert len(result.top_child_chunks) == 1
    assert len(result.final_contexts) == 1

    ctx = result.final_contexts[0]
    assert ctx.rank == 1
    assert ctx.parent_window_id == "parent-X"
    assert ctx.doc_id == "doc-1"
    assert ctx.parent_text == "A" * 50   # char_start=0, char_end=50
    assert ctx.best_rerank_score == -1.5
    assert ctx.best_rrf_score == 0.07
    assert "chunk-a" in ctx.source_chunks

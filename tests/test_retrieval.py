import pytest
import numpy as np
from unittest.mock import MagicMock
import sys

# Mock FlagEmbedding before anything imports it
sys.modules["FlagEmbedding"] = MagicMock()

from retrieval.schema import RetrievalCandidate, RerankResult
from retrieval.hybrid_retriever import HybridRetriever
from indexing.schema import RetrievalRecord

@pytest.fixture
def mock_retriever(monkeypatch):
    # Mock the index loading
    monkeypatch.setattr("retrieval.hybrid_retriever.load_index", lambda x: MagicMock())
    monkeypatch.setattr("retrieval.hybrid_retriever.load_bm25", lambda x: MagicMock())
    
    # Create fake metadata
    fake_metadata = [
        RetrievalRecord(
            global_chunk_id=i,
            chunk_id=f"chunk-{i}",
            doc_id="doc-1",
            parent_window_id="none",
            segment_id="none",
            title="Test",
            heading_path=[],
            original_filename="test.pdf",
            chunk_type="text",
            chunk_text=f"This is chunk {i}",
            embed_text_hash=f"hash-{i}"
        ) for i in range(10)
    ]
    monkeypatch.setattr("retrieval.hybrid_retriever.load_metadata", lambda x: fake_metadata)
    monkeypatch.setattr("retrieval.hybrid_retriever.get_model", lambda: None)
    
    # Mock QueryRewriter to just return the query (no expansion) to keep tests simple
    monkeypatch.setattr("retrieval.query_rewriter.QueryRewriter.rewrite", lambda self, q: [q])
    
    # Mock CrossEncoderReranker so it doesn't load the heavy model in retrieval tests
    def mock_rerank(self, q, candidates):
        return RerankResult(top_k_candidates=candidates[:5], all_reranked_candidates=candidates)
    monkeypatch.setattr("retrieval.reranker.CrossEncoderReranker.rerank", mock_rerank)
    
    retriever = HybridRetriever("dummy_dir")
    
    return retriever

def test_hybrid_retriever_rrf(mock_retriever, monkeypatch):
    def mock_encode_batch(texts):
        return [([0.1]*1024, {})]
    monkeypatch.setattr("retrieval.hybrid_retriever.encode_batch", mock_encode_batch)
    
    mock_faiss = mock_retriever.faiss_index
    mock_faiss.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[2, 1, 0]]))
    
    mock_bm25 = mock_retriever.bm25_state
    mock_bm25.model.get_scores.return_value = np.array([5.0, 3.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0])
    
    candidates = mock_retriever.retrieve("query", top_k_dense=3, top_k_bm25=3, final_k=5).top_k_candidates
    
    assert len(candidates) <= 5
    assert [c.chunk_id for c in candidates] == ["chunk-0", "chunk-1", "chunk-2", "chunk-3"]
    assert candidates[0].dense_rank == 3
    assert candidates[0].bm25_rank == 1
    assert candidates[0].rrf_score > candidates[1].rrf_score
    assert "query" in candidates[0].matched_queries
    assert "dense" in candidates[0].retrieval_sources

def test_hybrid_retriever_multi_query_additive_rrf(mock_retriever, monkeypatch):
    # Mock rewriter to return 2 queries
    mock_retriever.rewriter.rewrite = lambda q: ["q1", "q2"]
    
    def mock_encode_batch(texts):
        return [([0.1]*1024, {})]
    monkeypatch.setattr("retrieval.hybrid_retriever.encode_batch", mock_encode_batch)
    
    mock_faiss = mock_retriever.faiss_index
    # Query 1: chunk 1 is rank 1. Query 2: chunk 1 is rank 2.
    # Return different mock results based on call count or just same for both
    mock_faiss.search.return_value = (np.array([[0.9, 0.8]]), np.array([[1, 2]]))
    
    mock_bm25 = mock_retriever.bm25_state
    mock_bm25.model.get_scores.return_value = np.array([0, 5.0, 3.0, 0, 0, 0, 0, 0, 0, 0])
    
    candidates = mock_retriever.retrieve("query", top_k_dense=2, top_k_bm25=2, final_k=5).top_k_candidates
    
    # Chunk 1 got matched twice by dense, twice by sparse
    assert candidates[0].chunk_id == "chunk-1"
    assert "q1" in candidates[0].matched_queries
    assert "q2" in candidates[0].matched_queries

def test_hybrid_retriever_empty_dense(mock_retriever, monkeypatch):
    monkeypatch.setattr("retrieval.hybrid_retriever.encode_batch", lambda x: [None])
    
    mock_bm25 = mock_retriever.bm25_state
    mock_bm25.model.get_scores.return_value = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    candidates = mock_retriever.retrieve("query", top_k_dense=3, top_k_bm25=3, final_k=5).top_k_candidates
    assert len(candidates) == 1
    assert candidates[0].chunk_id == "chunk-0"
    assert candidates[0].dense_rank is None
    assert candidates[0].bm25_rank == 1

def test_hybrid_retriever_empty_sparse(mock_retriever, monkeypatch):
    monkeypatch.setattr("retrieval.hybrid_retriever.encode_batch", lambda x: [([0.1]*1024, {})])
    mock_faiss = mock_retriever.faiss_index
    mock_faiss.search.return_value = (np.array([[0.9]]), np.array([[5]]))
    
    mock_bm25 = mock_retriever.bm25_state
    mock_bm25.model.get_scores.return_value = np.array([0]*10)
    
    candidates = mock_retriever.retrieve("query", top_k_dense=3, top_k_bm25=3, final_k=5).top_k_candidates
    assert len(candidates) == 1
    assert candidates[0].chunk_id == "chunk-5"
    assert candidates[0].dense_rank == 1
    assert candidates[0].bm25_rank is None

def test_hybrid_retriever_final_k_limit(mock_retriever, monkeypatch):
    monkeypatch.setattr("retrieval.hybrid_retriever.encode_batch", lambda x: [([0.1]*1024, {})])
    mock_faiss = mock_retriever.faiss_index
    mock_faiss.search.return_value = (np.ones((1, 10)), np.arange(10).reshape(1, 10))
    
    mock_bm25 = mock_retriever.bm25_state
    mock_bm25.model.get_scores.return_value = np.ones(10)
    
    candidates = mock_retriever.retrieve("query", top_k_dense=10, top_k_bm25=10, final_k=4).top_k_candidates
    assert len(candidates) == 4

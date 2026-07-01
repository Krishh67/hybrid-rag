import pytest
import sys
from unittest.mock import MagicMock, patch

# Must mock FlagEmbedding before any module that imports it is collected
sys.modules.setdefault("FlagEmbedding", MagicMock())

from retrieval.schema import RetrievalCandidate
from retrieval.reranker import CrossEncoderReranker
import retrieval.config as config

@pytest.fixture
def mock_candidates():
    candidates = []
    for i in range(10):
        candidates.append(
            RetrievalCandidate(
                chunk_id=f"chunk-{i}",
                doc_id="doc-1",
                parent_window_id="none",
                segment_id="none",
                chunk_text=f"Text {i}",
                dense_rank=1,
                bm25_rank=1,
                rrf_score=0.1,
                matched_queries=[],
                retrieval_sources=[]
            )
        )
    return candidates

@patch("retrieval.reranker.FlagReranker")
def test_reranker_scores_and_sorts(mock_flag_reranker, mock_candidates, monkeypatch):
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(config, "TOP_K_RERANK", 5)
    
    mock_instance = MagicMock()
    # Return scores backwards so chunk 9 is best, chunk 0 is worst
    mock_instance.compute_score.return_value = [float(i) for i in range(10)]
    mock_flag_reranker.return_value = mock_instance
    
    # We must clear the singleton if it was set in previous tests
    import retrieval.reranker as reranker_module
    reranker_module._RERANKER_INSTANCE = None
    
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", mock_candidates).top_k_candidates
    
    # It should have truncated to TOP_K_RERANK (5)
    assert len(results) == 5
    
    # Best should be chunk-9 with score 9.0
    assert results[0].chunk_id == "chunk-9"
    assert results[0].rerank_score == 9.0
    assert results[0].rerank_rank == 1
    
    # 5th best should be chunk-5 with score 5.0
    assert results[4].chunk_id == "chunk-5"
    assert results[4].rerank_score == 5.0
    assert results[4].rerank_rank == 5

@patch("retrieval.reranker.FlagReranker")
def test_reranker_disabled(mock_flag_reranker, mock_candidates, monkeypatch):
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    monkeypatch.setattr(config, "TOP_K_RERANK", 3)
    
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", mock_candidates).top_k_candidates
    
    # Should not load model
    mock_flag_reranker.assert_not_called()
    
    # Should truncate to 3, but leave scores as None
    assert len(results) == 3
    assert results[0].chunk_id == "chunk-0"
    assert results[0].rerank_score is None
    assert results[0].rerank_rank is None

@patch("retrieval.reranker.FlagReranker")
def test_reranker_single_candidate(mock_flag_reranker, mock_candidates, monkeypatch):
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(config, "TOP_K_RERANK", 5)
    
    mock_instance = MagicMock()
    # compute_score might return a single float if len(pairs) == 1
    mock_instance.compute_score.return_value = 8.5
    mock_flag_reranker.return_value = mock_instance
    
    import retrieval.reranker as reranker_module
    reranker_module._RERANKER_INSTANCE = None
    
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", [mock_candidates[0]]).top_k_candidates
    
    assert len(results) == 1
    assert results[0].rerank_score == 8.5
    assert results[0].rerank_rank == 1

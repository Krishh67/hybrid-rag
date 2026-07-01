import pytest
from unittest.mock import Mock, patch

from retrieval.schema import FinalContext
from generation.schema import SourceReference, GenerationResult
from generation.context_builder import ContextBuilder
from generation.prompt_builder import PromptBuilder
from generation.pipeline import GenerationPipeline

def test_context_builder_basic():
    builder = ContextBuilder(max_tokens=100) # 400 chars
    
    contexts = [
        FinalContext(
            rank=1,
            parent_window_id="pw1",
            doc_id="doc1",
            parent_text="This is the first context.",
            source_chunks=["c1"],
            best_rerank_score=0.9,
            best_rrf_score=0.8,
            matched_queries=["q1"],
            retrieval_sources=["dense"]
        ),
        FinalContext(
            rank=2,
            parent_window_id="pw2",
            doc_id="doc2",
            parent_text="This is the second context.",
            source_chunks=["c2"],
            best_rerank_score=0.8,
            best_rrf_score=0.7,
            matched_queries=["q1"],
            retrieval_sources=["dense"]
        )
    ]
    
    text, sources = builder.build(contexts)
    
    assert "Document 1" in text
    assert "doc1" in text
    assert "This is the first context." in text
    
    assert "Document 2" in text
    assert "doc2" in text
    assert "This is the second context." in text
    
    assert len(sources) == 2
    assert sources[0].doc_id == "doc1"
    assert sources[0].parent_window_id == "pw1"

def test_context_builder_token_limit():
    builder = ContextBuilder(max_tokens=15) # 60 chars max limit
    
    contexts = [
        FinalContext(
            rank=1,
            parent_window_id="pw1",
            doc_id="doc1",
            parent_text="Short text.", # formatting adds ~ 40 chars
            source_chunks=["c1"],
            best_rerank_score=0.9,
            best_rrf_score=0.8,
            matched_queries=["q1"],
            retrieval_sources=["dense"]
        ),
        FinalContext(
            rank=2,
            parent_window_id="pw2",
            doc_id="doc2",
            parent_text="This is very long and should be skipped because of token limits.",
            source_chunks=["c2"],
            best_rerank_score=0.8,
            best_rrf_score=0.7,
            matched_queries=["q1"],
            retrieval_sources=["dense"]
        )
    ]
    
    text, sources = builder.build(contexts)
    
    assert "Document 1" in text
    assert "Document 2" not in text
    assert len(sources) == 1

def test_prompt_builder():
    builder = PromptBuilder()
    sys = builder.build_system_prompt("MY_CONTEXT_123")
    assert "MY_CONTEXT_123" in sys
    assert "CRITICAL INSTRUCTIONS" in sys
    
    user = builder.build_user_prompt("What is X?")
    assert "What is X?" in user

@patch('generation.pipeline.GeminiProvider')
def test_generation_pipeline(MockProvider):
    mock_provider = Mock()
    mock_provider.generate.return_value = ("The answer is X.", "mock-model", {"total_tokens": 10})
    
    pipeline = GenerationPipeline(provider=mock_provider)
    
    contexts = [
        FinalContext(
            rank=1,
            parent_window_id="pw1",
            doc_id="doc1",
            parent_text="Context X.",
            source_chunks=["c1"],
            best_rerank_score=0.9,
            best_rrf_score=0.8,
            matched_queries=["q1"],
            retrieval_sources=["dense"]
        )
    ]
    
    result = pipeline.generate("What is X?", contexts)
    
    assert result.answer == "The answer is X."
    assert result.model_used == "mock-model"
    assert len(result.sources) == 1
    assert result.sources[0].doc_id == "doc1"
    
    mock_provider.generate.assert_called_once()

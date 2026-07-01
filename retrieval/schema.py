from typing import Optional, List
from pydantic import BaseModel

class RetrievalCandidate(BaseModel):
    chunk_id: str
    doc_id: str
    parent_window_id: str
    segment_id: str
    chunk_text: str
    dense_rank: Optional[int]
    bm25_rank: Optional[int]
    rrf_score: float
    matched_queries: List[str]
    retrieval_sources: List[str]
    rerank_score: Optional[float] = None
    rerank_rank: Optional[int] = None

class RerankResult(BaseModel):
    top_k_candidates: List[RetrievalCandidate]
    all_reranked_candidates: List[RetrievalCandidate]
    query_variants: List[str] = []

class FinalContext(BaseModel):
    rank: int
    parent_window_id: str
    doc_id: str
    parent_text: str
    source_chunks: List[str]
    best_rerank_score: float
    best_rrf_score: float
    matched_queries: List[str]
    retrieval_sources: List[str]

class RetrievalResult(BaseModel):
    query: str
    top_child_chunks: List[RetrievalCandidate]
    final_contexts: List[FinalContext]

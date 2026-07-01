import logging
from pathlib import Path
from typing import List, Dict, Any
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np

from indexing.faiss_builder import load_index
from indexing.bm25_builder import load_bm25, tokenize
from indexing.metadata_builder import load_metadata
from embedding.model_wrapper import get_model, encode_batch
from .schema import RetrievalCandidate, RerankResult
from .query_rewriter import QueryRewriter
from .reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Stage X: Multi-Query Hybrid Retrieval combining Dense and Sparse via additive RRF."""
    
    def __init__(self, kb_dir: str):
        self.kb_dir = Path(kb_dir)
        index_dir = self.kb_dir / "index"
        
        logger.info("Initializing HybridRetriever for KB: %s", self.kb_dir.name)
        with open("trace.log", "a") as f:
            f.write("TRACE: Entering HybridRetriever load_index\n")


        self.faiss_index = load_index(index_dir)
        self.bm25_state = load_bm25(index_dir)
        self.metadata = load_metadata(index_dir)
        self.rewriter = QueryRewriter()
        logger.info("HybridRetriever initialized successfully.")

    def retrieve(
        self, 
        query: str, 
        top_k_dense: int = 20, 
        top_k_bm25: int = 20, 
        final_k: int = 50,
        progress_callback = None
    ) -> RerankResult:
        """
        Executes Multi-Query Dense and Sparse retrieval, fuses scores using additive RRF (k=60), 
        and returns the top `final_k` candidates.
        """
        if progress_callback: progress_callback("Loading AI Models...")
        
        # Ensure PyTorch thread pools are fully initialized BEFORE any gRPC calls are made
        get_model()
        CrossEncoderReranker().load_model()
        
        if progress_callback: progress_callback("Generating query variants with LLM...")
        queries = self.rewriter.rewrite(query)
        logger.info("Expanded into %d queries: %s", len(queries), queries)
        
        # Tracking dictionary: global_chunk_id -> state dict
        # State dict holds: rrf_score, min_dense_rank, min_bm25_rank, matched_queries, retrieval_sources
        chunk_scores: Dict[int, dict] = {}
        RRF_K = 60
        
        if progress_callback: progress_callback("Running Dense & Sparse Search...")
        
        for q in queries:
            # 1. Dense Retrieval (FAISS)
            encode_results = encode_batch([q])
            if encode_results and encode_results[0] is not None:
                query_dense = np.array([encode_results[0][0]], dtype=np.float32)
                D, I = self.faiss_index.search(query_dense, top_k_dense)
                dense_indices = I[0]
                
                rank = 1
                for idx in dense_indices:
                    if idx == -1:
                        break
                    idx = int(idx)
                    
                    if idx not in chunk_scores:
                        chunk_scores[idx] = {
                            "rrf_score": 0.0,
                            "min_dense_rank": None,
                            "min_bm25_rank": None,
                            "matched_queries": set(),
                            "retrieval_sources": set()
                        }
                        
                    state = chunk_scores[idx]
                    state["rrf_score"] += 1.0 / (RRF_K + rank)
                    if state["min_dense_rank"] is None or rank < state["min_dense_rank"]:
                        state["min_dense_rank"] = rank
                    state["matched_queries"].add(q)
                    state["retrieval_sources"].add("dense")
                    
                    rank += 1
                    
            # 2. Sparse Retrieval (BM25)
            tokenized_query = tokenize(q)
            if tokenized_query:
                scores = self.bm25_state.model.get_scores(tokenized_query)
                k = min(top_k_bm25, len(scores))
                if k > 0:
                    top_k_indices = np.argsort(scores)[::-1][:k]
                    rank = 1
                    for idx in top_k_indices:
                        if scores[idx] <= 0:
                            break
                        idx = int(idx)
                        
                        if idx not in chunk_scores:
                            chunk_scores[idx] = {
                                "rrf_score": 0.0,
                                "min_dense_rank": None,
                                "min_bm25_rank": None,
                                "matched_queries": set(),
                                "retrieval_sources": set()
                            }
                            
                        state = chunk_scores[idx]
                        state["rrf_score"] += 1.0 / (RRF_K + rank)
                        if state["min_bm25_rank"] is None or rank < state["min_bm25_rank"]:
                            state["min_bm25_rank"] = rank
                        state["matched_queries"].add(q)
                        state["retrieval_sources"].add("bm25")
                        
                        rank += 1

        # 3. Sort by RRF score descending
        candidates_scores = []
        for idx, state in chunk_scores.items():
            candidates_scores.append((idx, state))
            
        candidates_scores.sort(key=lambda x: x[1]["rrf_score"], reverse=True)
        
        # 4. Hydration: Construct RetrievalCandidate from metadata.pkl
        results = []
        for idx, state in candidates_scores[:final_k]:
            record = self.metadata[idx]
            
            candidate = RetrievalCandidate(
                chunk_id=record.chunk_id,
                doc_id=record.doc_id,
                parent_window_id=record.parent_window_id,
                segment_id=record.segment_id,
                chunk_text=record.chunk_text,
                dense_rank=state["min_dense_rank"],
                bm25_rank=state["min_bm25_rank"],
                rrf_score=state["rrf_score"],
                matched_queries=list(state["matched_queries"]),
                retrieval_sources=list(state["retrieval_sources"])
            )
            results.append(candidate)
            
        # 5. Cross Encoder Reranking
        # NOTE: Do NOT call free_model() here. Freeing PyTorch models between
        # pipeline stages and then calling gc.collect() corrupts shared C++ state
        # (OpenMP/MKL thread pools) that gRPC and other libraries depend on.
        # Both models stay as permanent singletons for the process lifetime.
        reranker = CrossEncoderReranker()
        
        if progress_callback: progress_callback("Reranking candidates...")
        final_results = reranker.rerank(query, results)
        final_results.query_variants = queries
        
        return final_results

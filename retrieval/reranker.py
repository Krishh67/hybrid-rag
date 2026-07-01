import logging
from typing import List, Optional
from FlagEmbedding import FlagReranker

from .schema import RetrievalCandidate, RerankResult
from . import config

logger = logging.getLogger(__name__)

# Module-level singleton
_RERANKER_INSTANCE: Optional[FlagReranker] = None

class CrossEncoderReranker:
    def __init__(self):
        self.enabled = config.RERANK_ENABLED
        self.model_name = config.RERANK_MODEL
        self.top_k = config.TOP_K_RERANK
        self.batch_size = config.RERANK_BATCH_SIZE
        
    def load_model(self):
        """Loads the FlagReranker model into memory if not already loaded."""
        if not self.enabled:
            return
            
        global _RERANKER_INSTANCE
        if _RERANKER_INSTANCE is None:
            logger.info("Loading FlagReranker model: %s", self.model_name)
            # FlagReranker automatically handles device placement (GPU if available)
            # use_fp16=True halves memory usage, preventing mmap OOM issues (Error 1455) on CPU
            _RERANKER_INSTANCE = FlagReranker(self.model_name, use_fp16=True)
            logger.info("FlagReranker model loaded successfully.")
            
    def free_model(self):
        """Frees the reranker model from RAM to prevent OOM."""
        global _RERANKER_INSTANCE
        if _RERANKER_INSTANCE is not None:
            logger.info("Freeing FlagReranker model from memory...")
            _RERANKER_INSTANCE = None
            import gc
            gc.collect()
            
    def rerank(self, query: str, candidates: List[RetrievalCandidate]) -> RerankResult:
        """
        Takes the original query and a list of candidates, scores the (query, chunk) pairs,
        and returns a RerankResult containing both the top-k and the full sorted list.
        """
        if not self.enabled or not candidates:
            return RerankResult(
                top_k_candidates=candidates[:self.top_k],
                all_reranked_candidates=candidates
            )
            
        self.load_model()
        global _RERANKER_INSTANCE
        
        # Prepare pairs for batch scoring
        pairs = [(query, c.chunk_text) for c in candidates]
        
        # Batch compute scores using configured batch size
        scores = _RERANKER_INSTANCE.compute_score(pairs, batch_size=self.batch_size)
        
        if isinstance(scores, float):
            scores = [scores]
            
        # Update candidates with scores
        for c, score in zip(candidates, scores):
            c.rerank_score = score
            
        # Sort candidates descending by score
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Assign ranks
        for rank, c in enumerate(candidates, start=1):
            c.rerank_rank = rank
            
        top_k_results = candidates[:self.top_k]
            
        return RerankResult(
            top_k_candidates=top_k_results,
            all_reranked_candidates=candidates
        )

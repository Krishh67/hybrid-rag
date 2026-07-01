import logging
from typing import List, Dict, Optional

from .schema import RetrievalCandidate, FinalContext, RetrievalResult
from .parent_resolver import ParentWindowResolver
from . import config

logger = logging.getLogger(__name__)


class ParentExpander:
    """
    Expands a list of reranked child chunks into deduplicated parent contexts.

    For each reranked candidate:
      - Resolves parent_window_id → parent_text via ParentWindowResolver
      - Groups candidates sharing the same parent_window_id
      - Merges source_chunks, matched_queries, retrieval_sources
      - Keeps the highest rerank_score and best rrf_score per parent
    """

    def __init__(self, resolver: ParentWindowResolver):
        self.resolver = resolver

    def expand(
        self,
        candidates: List[RetrievalCandidate],
        max_contexts: int = config.MAX_CONTEXTS,
    ) -> List[FinalContext]:
        """
        Deduplicate candidates by parent_window_id and resolve parent text.
        Returns up to max_contexts FinalContext objects sorted by best_rerank_score DESC.
        """
        # parent_window_id → merged state
        parent_map: Dict[str, dict] = {}

        for candidate in candidates:
            pid = candidate.parent_window_id

            if pid not in parent_map:
                parent_map[pid] = {
                    "parent_window_id": pid,
                    "doc_id": candidate.doc_id,
                    "source_chunks": [],
                    "best_rerank_score": candidate.rerank_score if candidate.rerank_score is not None else float("-inf"),
                    "best_rrf_score": candidate.rrf_score,
                    "matched_queries": set(),
                    "retrieval_sources": set(),
                }

            state = parent_map[pid]
            state["source_chunks"].append(candidate.chunk_id)

            # Keep best (highest) scores
            if candidate.rerank_score is not None:
                if candidate.rerank_score > state["best_rerank_score"]:
                    state["best_rerank_score"] = candidate.rerank_score

            if candidate.rrf_score > state["best_rrf_score"]:
                state["best_rrf_score"] = candidate.rrf_score

            state["matched_queries"].update(candidate.matched_queries)
            state["retrieval_sources"].update(candidate.retrieval_sources)

        # Sort by best_rerank_score descending
        sorted_states = sorted(
            parent_map.values(),
            key=lambda s: s["best_rerank_score"],
            reverse=True
        )

        # Resolve parent text and build FinalContext objects
        final_contexts: List[FinalContext] = []
        for rank, state in enumerate(sorted_states[:max_contexts], start=1):
            pid = state["parent_window_id"]
            parent_text = self.resolver.resolve(pid)

            # Fallback: if parent is "none" or resolution fails, skip this context
            if parent_text is None:
                logger.warning("Could not resolve parent text for '%s' — skipping.", pid)
                continue

            ctx = FinalContext(
                rank=rank,
                parent_window_id=pid,
                doc_id=state["doc_id"],
                parent_text=parent_text,
                source_chunks=state["source_chunks"],
                best_rerank_score=state["best_rerank_score"],
                best_rrf_score=state["best_rrf_score"],
                matched_queries=sorted(state["matched_queries"]),
                retrieval_sources=sorted(state["retrieval_sources"]),
            )
            final_contexts.append(ctx)

        # Re-assign contiguous ranks after any skips
        for i, ctx in enumerate(final_contexts, start=1):
            ctx.rank = i

        return final_contexts


class ContextAssembler:
    """
    Assembles the final RetrievalResult from reranked candidates and parent contexts.
    Orchestrates ParentExpander and wraps everything into the LLM-ready output object.
    """

    def __init__(self, kb_dir: str):
        self.resolver = ParentWindowResolver(kb_dir)
        self.expander = ParentExpander(self.resolver)

    def assemble(
        self,
        query: str,
        top_child_chunks: List[RetrievalCandidate],
        max_contexts: int = config.MAX_CONTEXTS,
    ) -> RetrievalResult:
        """
        Takes the reranked child chunks and returns a complete RetrievalResult
        with final_contexts ready to be passed directly to the LLM generation stage.
        """
        logger.info(
            "Assembling contexts for %d child chunks (max_contexts=%d)",
            len(top_child_chunks),
            max_contexts,
        )

        final_contexts = self.expander.expand(top_child_chunks, max_contexts=max_contexts)

        logger.info("Assembled %d final contexts.", len(final_contexts))

        return RetrievalResult(
            query=query,
            top_child_chunks=top_child_chunks,
            final_contexts=final_contexts,
        )

import logging
from typing import List, Tuple
from retrieval.schema import FinalContext
from generation.schema import SourceReference
from generation import config

logger = logging.getLogger(__name__)

class ContextBuilder:
    """
    Transforms a list of FinalContext objects into a unified, token-bounded 
    string block, and keeps track of the used sources.
    """
    
    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or config.MAX_CONTEXT_TOKENS

    def build(self, contexts: List[FinalContext]) -> Tuple[str, List[SourceReference]]:
        """
        Merge contexts and remove duplicate content, respecting token budgets.
        """
        # Sort by rerank score descending (should already be done, but ensure)
        sorted_contexts = sorted(contexts, key=lambda c: c.best_rerank_score, reverse=True)

        merged_text_parts = []
        sources = []
        current_char_count = 0
        max_chars = self.max_tokens * config.CHARS_PER_TOKEN
        
        seen_parent_ids = set()

        for ctx in sorted_contexts:
            # Deduplication at the parent_window_id level
            if ctx.parent_window_id in seen_parent_ids:
                continue
            seen_parent_ids.add(ctx.parent_window_id)

            block_text = f"--- [Document {len(sources) + 1} | Source: {ctx.doc_id}] ---\n{ctx.parent_text}\n"
            block_chars = len(block_text)

            if current_char_count + block_chars > max_chars:
                logger.warning(
                    "Context limit reached. Dropped remaining contexts (used %d/%d chars).",
                    current_char_count, max_chars
                )
                break
                
            merged_text_parts.append(block_text)
            current_char_count += block_chars
            
            sources.append(
                SourceReference(
                    doc_id=ctx.doc_id,
                    parent_window_id=ctx.parent_window_id
                )
            )

        final_context_string = "\n".join(merged_text_parts)
        return final_context_string, sources

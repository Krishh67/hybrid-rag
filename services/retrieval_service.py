import os
import json
import logging
import datetime
import time
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.parent_expander import ContextAssembler
from generation.pipeline import GenerationPipeline

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, kb_dir: str):
        self.kb_dir = kb_dir
        
    def execute_query(self, query: str, progress_callback=None) -> dict:
        """
        Executes the full pipeline and returns a trace dictionary for visualization.
        """
        trace = {
            "query": query,
            "query_variants": [],
            "dense_results": [],
            "bm25_results": [],
            "rrf_results": [],
            "reranked_results": [],
            "final_contexts": [],
            "answer": "",
            "sources": [],
            "model_used": "",
            "usage_metadata": {},
            "timing": {},
            "error": None
        }
        
        try:
            t_start = time.time()
            # 1. Retrieval
            if progress_callback: progress_callback("Initializing Hybrid Retriever...")
            with open("trace.log", "a") as f:
                f.write("TRACE: Entering RetrievalService HybridRetriever init\n")
            retriever = HybridRetriever(self.kb_dir)
            with open("trace.log", "a") as f:
                f.write("TRACE: Exiting RetrievalService HybridRetriever init\n")
            t_retrieval_start = time.time()
            
            trace["timing"]["initialization"] = round(t_retrieval_start - t_start, 2)
            
            import retrieval.config as rcfg
            
            with open("trace.log", "a") as f:
                f.write("TRACE: About to call retriever.retrieve()\n")
            # Retrieve (this actually does dense + bm25 + rrf + rerank under the hood)
            rerank_result = retriever.retrieve(
                query,
                top_k_dense=rcfg.TOP_K_RRF,
                top_k_bm25=rcfg.TOP_K_RRF,
                final_k=rcfg.TOP_K_RRF,
                progress_callback=progress_callback
            )
            t_retrieval_end = time.time()
            trace["timing"]["retrieval_and_rerank"] = round(t_retrieval_end - t_retrieval_start, 2)
            
            trace["query_variants"] = rerank_result.query_variants
            
            top_chunks = rerank_result.top_k_candidates
            
            # For the trace, we will just store the chunks that were reranked
            trace["reranked_results"] = top_chunks
            
            if progress_callback: progress_callback("Expanding Parent Contexts...")
            # 2. Parent Expansion
            t_expand_start = time.time()
            assembler = ContextAssembler(self.kb_dir)
            assembler_result = assembler.assemble(
                query=query,
                top_child_chunks=top_chunks,
                max_contexts=rcfg.MAX_CONTEXTS
            )
            t_expand_end = time.time()
            trace["timing"]["parent_expansion"] = round(t_expand_end - t_expand_start, 2)
            
            trace["final_contexts"] = assembler_result.final_contexts
            if progress_callback: progress_callback("LLM Waiting: Sending context to Gemini...")
            
            # 3. Generation
            t_gen_start = time.time()
            gen_pipeline = GenerationPipeline()
            gen_result = gen_pipeline.generate(query, assembler_result.final_contexts)
            t_gen_end = time.time()
            trace["timing"]["llm_generation"] = round(t_gen_end - t_gen_start, 2)
            trace["timing"]["total_time"] = round(t_gen_end - t_start, 2)
            
            trace["answer"] = gen_result.answer
            trace["sources"] = gen_result.sources
            trace["model_used"] = gen_result.model_used
            trace["usage_metadata"] = gen_result.usage_metadata
            
            # Log to local file
            self._save_query_log(query, trace)
            
        except Exception as e:
            logger.error(f"Error in RetrievalService: {e}", exc_info=True)
            trace["error"] = str(e)
            
        return trace
        
    def _save_query_log(self, query: str, trace: dict):
        log_file = "knowledge_bases/past_queries.jsonl"
        
        # Serialize trace to save it, but remove heavy objects like full text if needed, 
        # or just save everything so we can expand it later.
        # We need to make sure trace is JSON serializable (FinalContext objects are Pydantic models).
        
        serializable_contexts = []
        for ctx in trace["final_contexts"]:
            serializable_contexts.append(ctx.model_dump() if hasattr(ctx, 'model_dump') else ctx)
            
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "kb_used": self.kb_dir,
            "response_length": len(trace["answer"]),
            "model_used": trace["model_used"],
            "answer": trace["answer"],
            "final_contexts": serializable_contexts
        }
        
        try:
            os.makedirs("knowledge_bases", exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to save query log: {e}")

import streamlit as st
from ui.theme import card_entry, stat_box, chip, divider

def render_retrieval_trace(trace: dict):
    """
    Renders an expandable section showing exactly how the answer was formed.
    Uses theme.py components.
    """
    with st.expander("🔍 Retrieval Pipeline Diagnostics", expanded=False):
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        
        # Technical Details
        cols = st.columns(4)
        with cols[0]:
            stat_box(len(trace.get("reranked_results", [])), "Reranked Chunks")
        with cols[1]:
            stat_box(len(trace.get("final_contexts", [])), "Final Contexts")
        with cols[2]:
            stat_box(trace.get("usage_metadata", {}).get("total_tokens", 0), "LLM Tokens")
        with cols[3]:
            # Mocking time for now, as we don't have timer in trace
            stat_box(trace.get("model_used", "N/A"), "Model")
            
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        
        divider("Pipeline Execution")
        
        st.markdown("**Original Query:**")
        chip(trace["query"])
        
        if trace.get("query_variants"):
            st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Multi-Query Expansions:**")
            for qv in trace["query_variants"]:
                if qv != trace["query"]:
                    chip(qv, warning=True)
                    
        if trace.get("timing"):
            st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("**Execution Times:**")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                stat_box(f"{trace['timing'].get('initialization', 0)}s", "Model Init")
            with col2:
                stat_box(f"{trace['timing'].get('retrieval_and_rerank', 0)}s", "Retriever & Rerank")
            with col3:
                stat_box(f"{trace['timing'].get('parent_expansion', 0)}s", "Context Assemble")
            with col4:
                stat_box(f"{trace['timing'].get('llm_generation', 0)}s", "LLM Generation")
            with col5:
                stat_box(f"{trace['timing'].get('total_time', 0)}s", "Total Pipeline")
        
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("**Final Contexts Sent to LLM:**")
        
        for ctx in trace.get("final_contexts", []):
            card_entry(
                rank=ctx.rank,
                title=f"Doc ID: {ctx.doc_id}",
                badges=[(f"Rerank Score: {ctx.best_rerank_score:.3f}", "score"), (f"RRF: {ctx.best_rrf_score:.3f}", "year")],
                quote=ctx.parent_text[:200].replace('\n', ' ')
            )

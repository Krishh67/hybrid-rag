import streamlit as st
import json
import os

from state.session import init_session_state
from ui.theme import set_page_config, inject_css, hero, card_entry, empty_state
from ui.sidebar import render_sidebar

init_session_state()
set_page_config(page_title="Past Queries", page_icon="📜")
inject_css()

render_sidebar(current_page="past_queries")

hero(
    badge_text="HISTORY",
    title="Past Queries",
    subtitle="View your previous searches and generation metadata."
)

log_file = "knowledge_bases/past_queries.jsonl"

if os.path.exists(log_file):
    try:
        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        
        # Sort logs by timestamp descending if needed, or assume they are appended.
        # Actually our append logic puts newest at the bottom, so reverse to show newest first.
        logs.reverse()
            
        if not logs:
            empty_state("No History", "No queries have been executed yet.")
        else:
            for i, log in enumerate(logs):
                ts = log.get("timestamp", "")
                ts = ts.split("T")[0] + " " + ts.split("T")[1][:5] if "T" in ts else ts
                
                # Use standard expander for Q&A history
                with st.expander(f"{ts} | {log.get('query', 'Unknown Query')}", expanded=False):
                    st.markdown("**Query:**")
                    st.info(log.get("query", ""))
                    st.markdown("**Answer:**")
                    st.markdown(log.get("answer", "No answer recorded."))
                    
                    st.markdown(f"**KB:** `{log.get('kb_used', 'Unknown')}` | **Model:** `{log.get('model_used', 'Unknown')}` | **Length:** `{log.get('response_length', 0)} chars`")
                    
                    if "final_contexts" in log and log["final_contexts"]:
                        st.caption("Sources:")
                        for ctx in log["final_contexts"]:
                            if isinstance(ctx, dict):
                                st.caption(f"- Doc ID: {ctx.get('doc_id')}")
    except Exception as e:
        st.error(f"Failed to load query history: {e}")
else:
    empty_state("No History", "No queries have been executed yet.")

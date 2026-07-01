import streamlit as st
import glob
import os

from state.session import init_session_state, apply_settings_to_backend
from ui.theme import set_page_config, inject_css, hero, divider
from ui.sidebar import render_sidebar

init_session_state()
set_page_config(page_title="Settings", page_icon="⚙️")
inject_css()

render_sidebar(current_page="settings")

hero(
    badge_text="CONFIGURATION",
    title="System Settings",
    subtitle="Manage Knowledge Bases and configure pipeline parameters."
)

with st.form("create_kb_form"):
    divider("Create Knowledge Base")
    new_kb_name = st.text_input("New Knowledge Base Name (e.g., finance_kb, kb_005)", placeholder="Enter name...")
    if st.form_submit_button("Create"):
        if new_kb_name.strip():
            clean_name = "".join(c for c in new_kb_name if c.isalnum() or c in ("_", "-"))
            if clean_name:
                kb_dir = os.path.join("knowledge_bases", clean_name)
                os.makedirs(kb_dir, exist_ok=True)
                # Create default subdirectories so it's fully initialized
                for sub in ("parsed", "chunks", "embeddings", "index", "parent_windows"):
                    os.makedirs(os.path.join(kb_dir, sub), exist_ok=True)
                st.session_state.selected_kb = kb_dir
                st.success(f"Knowledge Base '{clean_name}' created successfully!")
                st.rerun()
            else:
                st.error("Invalid name. Use letters, numbers, hyphens, and underscores.")

with st.form("settings_form"):
    divider("Select Knowledge Base")
    
    kbs = sorted([d for d in glob.glob("knowledge_bases/*") if os.path.isdir(d)])
    if not kbs:
        st.warning("No Knowledge Bases found. They will be created during upload.")
        kbs = ["knowledge_bases/kb_001"]
        
    kb_options = [os.path.basename(kb) for kb in kbs]
    current_kb_idx = 0
    if st.session_state.selected_kb in kbs:
        current_kb_idx = kbs.index(st.session_state.selected_kb)
        
    selected_kb_name = st.selectbox("Active Knowledge Base", options=kb_options, index=current_kb_idx)
    
    divider("Retrieval Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k_rrf = st.number_input("Top K RRF", value=st.session_state.top_k_rrf, min_value=1)
    with col2:
        top_k_rerank = st.number_input("Top K Rerank", value=st.session_state.top_k_rerank, min_value=1)
    with col3:
        max_contexts = st.number_input("Final Context Count", value=st.session_state.max_contexts, min_value=1)
        
    divider("LLM Settings")
    col4, col5 = st.columns(2)
    with col4:
        llm_model = st.selectbox("Model (Fallback pipeline applied automatically)", 
                                 options=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash"],
                                 index=["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash"].index(st.session_state.llm_model) if st.session_state.llm_model in ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash"] else 0)
    with col5:
        temperature = st.slider("Temperature", value=st.session_state.temperature, min_value=0.0, max_value=1.0, step=0.1)
        
    submitted = st.form_submit_button("Save Configuration")
    if submitted:
        # Update State
        st.session_state.selected_kb = next((kb for kb in kbs if os.path.basename(kb) == selected_kb_name), kbs[0])
        st.session_state.top_k_rrf = top_k_rrf
        st.session_state.top_k_rerank = top_k_rerank
        st.session_state.max_contexts = max_contexts
        st.session_state.llm_model = llm_model
        st.session_state.temperature = temperature
        
        # Sync to backend
        apply_settings_to_backend()
        
        st.success("Settings updated successfully!")

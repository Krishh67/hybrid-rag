import streamlit as st
import os
import glob
from retrieval import config as retrieval_config
from generation import config as generation_config

def get_first_kb():
    kbs = sorted([d for d in glob.glob("knowledge_bases/*") if os.path.isdir(d)])
    return kbs[0] if kbs else "knowledge_bases/kb_001"

def init_session_state():
    """Initializes all necessary session state variables for the frontend."""
    
    # 1. Knowledge Base
    if "selected_kb" not in st.session_state:
        st.session_state.selected_kb = get_first_kb()
        
    # 2. History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
        
    # 3. Settings (Retrieval)
    if "top_k_rrf" not in st.session_state:
        st.session_state.top_k_rrf = retrieval_config.TOP_K_RRF
        
    if "top_k_rerank" not in st.session_state:
        st.session_state.top_k_rerank = retrieval_config.TOP_K_RERANK
        
    if "max_contexts" not in st.session_state:
        st.session_state.max_contexts = retrieval_config.MAX_CONTEXTS
        
    # 4. Settings (Generation)
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = generation_config.LLM_MODEL
        
    if "temperature" not in st.session_state:
        st.session_state.temperature = generation_config.TEMPERATURE
        
    if "max_output_tokens" not in st.session_state:
        st.session_state.max_output_tokens = generation_config.MAX_OUTPUT_TOKENS

def apply_settings_to_backend():
    """Overrides backend configs with current session state."""
    retrieval_config.TOP_K_RRF = st.session_state.top_k_rrf
    retrieval_config.TOP_K_RERANK = st.session_state.top_k_rerank
    retrieval_config.MAX_CONTEXTS = st.session_state.max_contexts
    
    generation_config.LLM_MODEL = st.session_state.llm_model
    generation_config.TEMPERATURE = st.session_state.temperature
    generation_config.MAX_OUTPUT_TOKENS = st.session_state.max_output_tokens

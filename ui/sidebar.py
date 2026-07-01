import streamlit as st
import os
from ui.theme import sidebar_brand, sidebar_section_label, status_pill, sidebar_hr

def render_sidebar(current_page="app"):
    """Renders the common left navigation sidebar using Streamlit's native menu."""
    with st.sidebar:
        sidebar_brand(eyebrow="RAG SYSTEM", title_html="Enterprise<br>Retrieval")
        
        # Native Streamlit page navigation handles the menu automatically
        sidebar_hr()
        sidebar_section_label("System Status")
        status_pill("Service Online", status="ok")
        kb_name = os.path.basename(st.session_state.selected_kb) if "selected_kb" in st.session_state else "None"
        status_pill(f"Active KB: {kb_name}", status="ok")
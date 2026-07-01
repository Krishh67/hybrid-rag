import streamlit as st
import os
import shutil

# MUST be set before ANY C++ libraries are imported
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from state.session import init_session_state
from ui.theme import set_page_config, inject_css, hero, divider, badge
from ui.sidebar import render_sidebar
from ui.upload_progress import render_upload_progress
from services.ingestion_service import run_ingestion_generator

init_session_state()
set_page_config(page_title="Upload Documents", page_icon="☁️")
inject_css()

render_sidebar(current_page="upload")

kb_name = os.path.basename(st.session_state.selected_kb)
hero(
    badge_text="INGESTION",
    title="Upload Documents",
    subtitle=f"Add new PDFs, DOCX, TXT, or MD files to {kb_name}."
)

divider(f"Target: {kb_name}")

uploaded_files = st.file_uploader(
    "Choose files to add to the Knowledge Base",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

if st.button("Process Documents", disabled=not uploaded_files, use_container_width=True):
    # Copy files to inputs directory
    os.makedirs("inputs", exist_ok=True)
    file_paths = []
    
    for file in uploaded_files:
        file_path = os.path.join("inputs", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    
    progress_container = st.empty()
    
    def live_update(status):
        render_upload_progress(progress_container, status)
        
    # Run the generator service
    generator = run_ingestion_generator(st.session_state.selected_kb, file_paths, progress_callback=live_update)
    for status in generator:
        live_update(status)

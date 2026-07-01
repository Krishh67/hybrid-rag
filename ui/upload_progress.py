import streamlit as st
from ui.theme import success_bar, error_box

def render_upload_progress(progress_container, status):
    """
    Visualizes the live upload/ingestion pipeline progress.
    status is a dict with: stage, message, progress_pct
    """
    if status["stage"] == "error":
        with progress_container:
            error_box("Ingestion Failed", status["message"])
    elif status["stage"] == "completed":
        with progress_container:
            success_bar(
                message="Successfully Added to Knowledge Base",
                right_label="Completed",
                right_sublabel=status["message"]
            )
    else:
        with progress_container:
            st.markdown(f"**Current Stage:** {status['stage'].capitalize()}")
            st.progress(status["progress_pct"] / 100.0)
            st.caption(status["message"])

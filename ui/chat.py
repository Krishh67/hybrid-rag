import streamlit as st

def render_chat_history(chat_history):
    """Renders the chat history in the standard ChatGPT format."""
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If there's an expandable trace, render it
            if message.get("trace"):
                from ui.retrieval_view import render_retrieval_trace
                render_retrieval_trace(message["trace"])

import streamlit as st
import os

# MUST be set before ANY C++ libraries are imported
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



from state.session import init_session_state, apply_settings_to_backend
from ui.theme import set_page_config, inject_css, hero, empty_state, answer_box_label, answer_body, badge
from ui.sidebar import render_sidebar
from ui.chat import render_chat_history
# Initialize state and styling
init_session_state()
apply_settings_to_backend()
set_page_config(page_title="RAG Chat", page_icon="💬")
inject_css()

# Render Sidebar Navigation
render_sidebar(current_page="app")

# Page Content
kb_name = os.path.basename(st.session_state.selected_kb)
hero(
    badge_text="KNOWLEDGE BASE CHAT",
    title="RAG Assistant",
    subtitle=f"Ask questions based on the documents stored in {kb_name}.",
)

# Render Chat History
def render_chat_history_with_avatars(chat_history):
    for message in chat_history:
        avatar = message.get("avatar")
        # Ensure we don't pass emoji strings to avatar if we want them removed
        if avatar in ("🧑‍💻", "🤖"):
            avatar = None
            
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("trace"):
                from ui.retrieval_view import render_retrieval_trace
                render_retrieval_trace(message["trace"])

render_chat_history_with_avatars(st.session_state.chat_history)

if not st.session_state.chat_history:
    empty_state(
        eyebrow="Engine Standing By",
        message=f"Enter a query below to search {kb_name}."
    )

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to state and UI
    st.session_state.chat_history.append({"role": "user", "content": prompt, "trace": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        
        # Start query in background thread
        if "rag_worker" not in st.session_state:
            from rag_worker import RAGWorker
            st.session_state.rag_worker = RAGWorker()
            
        worker = st.session_state.rag_worker
        worker.query(st.session_state.selected_kb, prompt)
        
        trace = None
        while True:
            # Check for progress updates
            updates = worker.get_progress()
            for msg in updates:
                # Render a nice animated pill or badge for the status
                status_html = f"""
                <div style='display:flex;align-items:center;gap:8px;padding:0.75rem 1.25rem;
                            background:#f8fafc;border:1px solid #cbd5e1;border-radius:99px;
                            margin-bottom:1rem;width:fit-content;box-shadow:0 2px 4px rgba(0,0,0,0.02);'>
                    <div style='width:12px;height:12px;border:2px solid #2563eb;border-top-color:transparent;
                                border-radius:50%;animation:spin 1s linear infinite;'></div>
                    <span style='font-size:0.9rem;font-weight:600;color:#334155;font-family:"Inter",sans-serif;'>
                        {msg}
                    </span>
                </div>
                <style>@keyframes spin {{ 100% {{ transform:rotate(360deg); }} }}</style>
                """
                status_placeholder.markdown(status_html, unsafe_allow_html=True)
                
            # Check if finished
            result = worker.get_result(timeout=0.1)
            if result is not None:
                if result["status"] == "error":
                    trace = {"error": result["error"]}
                else:
                    trace = result["trace"]
                break
            
            import time
            time.sleep(0.1)
        
        # Clear the status indicator once complete
        status_placeholder.empty()
            
        if trace and trace.get("error"):
            st.error(f"Error: {trace['error']}")
            response_text = "Sorry, an error occurred during retrieval."
        else:
            response_text = trace["answer"]
            
            # Render answer box styling
            answer_box_label(icon="✧", label="Generated Answer")
            answer_body(response_text)
            
            # Render the expandable trace
            from ui.retrieval_view import render_retrieval_trace
            render_retrieval_trace(trace)
                
            # Add to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "trace": trace
            })
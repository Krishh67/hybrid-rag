# Enterprise-Grade Hybrid RAG System

An advanced Retrieval-Augmented Generation (RAG) system built for robustness, accuracy, and high performance. This project implements state-of-the-art information retrieval techniques, combining Dense and Sparse vector searches, Cross-Encoder Reranking, and Parent-Child Context Expansion to provide highly accurate, hallucination-free answers powered by Google's Gemini models.

---

## 🏛️ System Architecture

### Phase 1: Ingestion & Embedding Pipeline
<!-- INGESTION ARCHITECTURE IMAGE PLACEHOLDER -->
<img width="1822" height="600" alt="image" src="https://github.com/user-attachments/assets/79c022c7-9a9d-4f16-b3f1-79ff5c0ba0e7" />




1. **Document Parsing**: Extracts text from unstructured PDFs, Word documents (DOCX), Markdown, and plain text.
2. **Parent-Child Chunking**: Splits documents into large "Parent" windows (for context) and smaller "Child" chunks (for precise embedding retrieval).
3. **Dense Embedding**: Generates semantic vector embeddings for child chunks using `BAAI/bge-m3`.
4. **Sparse Indexing**: Builds a term-frequency BM25 index over the chunks for exact keyword matching.
5. **Vector Store**: Persists embeddings into a high-performance local FAISS index.

### Phase 2: Hybrid Retrieval & Generation Pipeline
<!-- RETRIEVAL ARCHITECTURE IMAGE PLACEHOLDER -->
<img width="1811" height="524" alt="image" src="https://github.com/user-attachments/assets/d6d06e78-647d-4074-a666-a71b3027113b" />


1. **Query Rewriting**: Expands the user's query into multiple semantic variants using the LLM to maximize recall.
2. **Hybrid Search**: Executes parallel searches across the FAISS Dense index and the BM25 Sparse index.
3. **Reciprocal Rank Fusion (RRF)**: Merges the Dense and Sparse results mathematically to surface chunks that have both exact keyword overlap and high semantic similarity.
4. **Cross-Encoder Reranking**: Passes the top candidates through `BAAI/bge-reranker-v2-m3` for deep semantic scoring against the original query.
5. **Context Assembly (Parent Expansion)**: Replaces the retrieved tiny child chunks with their surrounding Parent Windows to give the LLM maximum surrounding context.
6. **LLM Generation**: Feeds the assembled context to `gemini-2.5-flash` to generate a final synthesized answer with citations.

---

## 🧠 AI Models Used

This system relies on the following models. They will automatically be downloaded and cached by HuggingFace the first time the system runs.

* **Dense Embedding Model**: `BAAI/bge-m3` (via FlagEmbedding) - Generates high-dimensional semantic vectors.
* **Semantic Reranker Model**: `BAAI/bge-reranker-v2-m3` - A Cross-Encoder that scores the relevance between a query and a document chunk.
* **Sparse Retrieval Model**: `BM25 (Okapi)` - Mathematical keyword frequency algorithm (built-in).
* **Generative LLM**: `gemini-2.5-flash` (via Google GenAI SDK) - Used for query rewriting and final answer generation.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```

2. **Create and activate a virtual environment (Python 3.10+ recommended):**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure PyTorch, FAISS, FlagEmbedding, and Streamlit are installed).*

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

---

## 🚀 Usage Guide

### 1. Command Line Interface (CLI)
The `main.py` script serves as the primary orchestrator for the backend pipeline.

* **Ingest Documents:**
  ```bash
  python main.py ingest "path/to/your/document.pdf" --kb_dir knowledge_bases/kb_001
  ```
* **Query the Knowledge Base:**
  ```bash
  python main.py query "What is the main topic of the document?" --kb_dir knowledge_bases/kb_001
  ```

### 2. Streamlit Web Interface
The project includes a fully-featured, dynamic web interface for managing knowledge bases, uploading files, and chatting with your documents.

**CRITICAL NOTE FOR WINDOWS USERS:** 
Do *not* run `streamlit run app.py` directly. You **must** use the `run.py` launcher. The launcher safely initializes PyTorch and FAISS C++ OpenMP thread pools on the absolute main thread before Streamlit boots up. Failing to use the launcher will result in a fatal `Exit Code 1` C++ thread abort when querying.

* **Start the App:**
  ```bash
  python run.py
  ```
  *(Note: The main Streamlit file may be named `terminal_working_app.py` or `app.py` depending on your current configuration).*

### Web App Architecture Highlights:
* **Persistent RAG Worker:** To guarantee thread-safety on Windows, the Streamlit app delegates all PyTorch and FAISS pipeline operations to a persistent, dedicated background worker thread (`rag_worker.py`), fully isolating the C++ runtime from Streamlit's volatile ScriptRunner threads.
* **Live Streaming:** Streams upload progress and retrieval status trace logs directly to the UI.

---

## 📂 Project Structure

```text
├── chunking/               # Parent-Child document splitting logic
├── embedding/              # BGE-M3 Dense Embedding generation and hashing cache
├── indexing/               # FAISS and BM25 index builders and manifest tracking
├── ingestion/              # Multi-format document parsers (PDF, DOCX, TXT, MD)
├── retrieval/              # Hybrid Search, Query Rewriting, and Reranking logic
├── generation/             # Gemini LLM integration for final answer synthesis
├── ui/                     # Streamlit custom CSS themes, sidebars, and chat views
├── pages/                  # Streamlit Multi-page routes (Upload, Settings)
├── main.py                 # Core CLI orchestrator
├── run.py                  # Thread-safe Streamlit Launcher
├── rag_worker.py           # Persistent background thread for Streamlit safe-execution
└── terminal_working_app.py # Main Streamlit entry point
```



**Embedding outputs**

`index/manifest.json`: Tracks all files that have ever been uploaded to prevent duplicates.

`index/faiss.index`: The compiled FAISS Database (containing all the dense vectors optimized for blazing-fast similarity search).

`index/bm25.pkl`: The compiled BM25 Database (containing keyword frequencies for sparse search).

`index/metadata.pkl`: An internal map linking the FAISS vector IDs back to your original child chunks.

`index/parent_lookup.pkl`: An internal map linking every tiny child chunk to its massive parent window.


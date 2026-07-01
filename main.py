"""
RAG Pipeline Orchestrator — main.py

Stages supported:
  1. Ingestion     (PDF/DOCX/TXT/ZIP → ParsedDocument)
  2. Chunking      (ParsedDocument → Chunks + Parent Windows)
  3. Embedding     (Chunks → dense vectors via BGE-M3)
  4. Indexing      (FAISS + BM25 + metadata + parent_lookup)
  5. Query         (Multi-Query → Dense+BM25 → RRF → Cross-Encoder → Parent Expansion)

Usage:
  python main.py
"""
import os
import glob
import json
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# ── Ensure core dirs exist ───────────────────────────────────────────────────
os.makedirs("inputs", exist_ok=True)
os.makedirs("knowledge_bases", exist_ok=True)


# ── Lazy imports (only pulled when that stage is actually needed) ─────────────
def _import_ingestion():
    from ingestion.pipeline import ingest_path
    return ingest_path

def _import_chunking():
    from chunking.pipeline import chunk_document
    return chunk_document

def _import_embedding():
    from embedding.pipeline import embed_document
    return embed_document

def _import_indexing():
    from indexing.pipeline import index_kb
    from indexing.manifest import load_manifest, is_duplicate
    return index_kb, load_manifest, is_duplicate

def _import_retrieval():
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.parent_expander import ContextAssembler
    return HybridRetriever, ContextAssembler

def _import_generation():
    from generation.pipeline import GenerationPipeline
    return GenerationPipeline


# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_print(text: str) -> None:
    """Print text, replacing unencodable characters for the current console."""
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


def list_inputs() -> list:
    files = []
    for ext in [".pdf", ".docx", ".txt", ".zip"]:
        files.extend(glob.glob(f"inputs/*{ext}"))
    return sorted(files)


def list_existing_kbs() -> list:
    return sorted(glob.glob("knowledge_bases/kb_*"))


def get_next_kb_dir() -> str:
    existing = list_existing_kbs()
    if not existing:
        return "knowledge_bases/kb_001"
    max_id = 0
    for e in existing:
        try:
            num = int(os.path.basename(e).replace("kb_", ""))
            max_id = max(max_id, num)
        except ValueError:
            pass
    return f"knowledge_bases/kb_{max_id + 1:03d}"


def pick_kb() -> str:
    """Interactively select or create a KB directory."""
    print("\n── Knowledge Base ──────────────────────────────────────")
    choice = input("  Create new KB (N) or use existing (E)? [N/E]: ").strip().lower()

    if choice == "e":
        kbs = list_existing_kbs()
        if not kbs:
            print("  No existing KBs. Creating a new one.")
            return get_next_kb_dir()
        print("\n  Existing KBs:")
        for i, kb in enumerate(kbs):
            print(f"    [{i}] {kb}")
        idx = int(input("  Select number: ").strip())
        return kbs[idx]
    return get_next_kb_dir()


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_ingest_to_index(kb_dir: str, selected_files: list) -> None:
    """Run Stages 1–4 (ingest → chunk → embed → index)."""
    index_kb, load_manifest, is_duplicate = _import_indexing()
    ingest_path   = _import_ingestion()
    chunk_document = _import_chunking()
    embed_document = _import_embedding()

    # Ensure KB folder structure
    for sub in ("parsed", "chunks", "embeddings", "index", "parent_windows"):
        os.makedirs(os.path.join(kb_dir, sub), exist_ok=True)

    manifest_dir = Path(kb_dir) / "index"
    kb_id        = os.path.basename(kb_dir)
    kb_manifest  = load_manifest(manifest_dir, kb_id)

    for file_path in selected_files:
        print(f"\n── Processing: {file_path}")

        # Stage 1: Ingestion
        print("  [Stage 1] Ingestion...")
        try:
            parsed_docs = ingest_path(file_path)
        except Exception as e:
            print(f"  ERROR during ingestion: {e}")
            continue

        if not parsed_docs:
            print(f"  Skipped (no content extracted).")
            continue

        for parsed_doc in parsed_docs:
            # Deduplication
            if is_duplicate(kb_manifest, parsed_doc.content_hash):
                existing = kb_manifest["indexed_content_hashes"][parsed_doc.content_hash]
                print(
                    f"  [DUPLICATE] Already in KB as '{existing.get('original_filename', '?')}' "
                    f"(doc_id: {existing.get('doc_id', '?')[:8]}...) — skipping."
                )
                continue

            # Save parsed
            parsed_out = os.path.join(kb_dir, "parsed", f"{parsed_doc.doc_id}.json")
            with open(parsed_out, "w", encoding="utf-8") as f:
                f.write(parsed_doc.model_dump_json(indent=2))
            print(f"  Parsed  → {parsed_out}")

            # Stage 2: Chunking
            print("  [Stage 2] Chunking...")
            chunks, parent_windows = chunk_document(parsed_doc)

            chunks_out = os.path.join(kb_dir, "chunks", f"{parsed_doc.doc_id}_chunks.json")
            with open(chunks_out, "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in chunks], f, indent=2)
            print(f"  Chunks  → {len(chunks)} chunks saved")

            pw_out = os.path.join(kb_dir, "parent_windows", f"{parsed_doc.doc_id}_parent_windows.json")
            with open(pw_out, "w", encoding="utf-8") as f:
                json.dump(parent_windows, f, indent=2)
            print(f"  Parents → {len(parent_windows)} parent windows saved")

            # Stage 3: Embedding
            print("  [Stage 3] Embedding...")
            try:
                embedded = embed_document(chunks, kb_dir, parsed_doc.doc_id)
                print(f"  Embeds  → {len(embedded)} vectors saved")
            except Exception as e:
                print(f"  ERROR during embedding: {e}")

    # Stage 4: Indexing (always run, even if no new docs — it's safe)
    print("\n  [Stage 4] Indexing (FAISS + BM25 + parent_lookup)...")
    try:
        result = index_kb(kb_dir)
        print(
            f"  Index   → {result.new_chunks_added} new chunks | "
            f"{result.new_documents_added} new docs | "
            f"{result.total_chunks} total"
        )
        for w in result.warnings:
            print(f"  WARNING: {w}")
    except Exception as e:
        print(f"  ERROR during indexing: {e}")


def run_query(kb_dir: str) -> None:
    """Run the full retrieval pipeline (Stages 5–7): multi-query → rerank → parent expand."""
    HybridRetriever, ContextAssembler = _import_retrieval()

    print(f"\n── Query Mode  (KB: {kb_dir})")
    query = input("  Enter query: ").strip()
    if not query:
        print("  Empty query — aborting.")
        return

    try:
        top_k_rrf    = int(input("  RRF candidates to retrieve [50]: ").strip() or "50")
        top_k_rerank = int(input("  Top-N after reranking [5]: ").strip() or "5")
        max_contexts = int(input("  Final parent contexts to return [3]: ").strip() or "3")
    except ValueError:
        print("  Invalid number — using defaults.")
        top_k_rrf    = 50
        top_k_rerank = 5
        max_contexts = 3

    # Dynamically override config
    import retrieval.config as rcfg
    rcfg.TOP_K_RRF    = top_k_rrf
    rcfg.TOP_K_RERANK = top_k_rerank
    rcfg.MAX_CONTEXTS = max_contexts

    print("\n  Loading retriever...")
    retriever = HybridRetriever(kb_dir)

    print(f"  Querying: '{query}'\n")
    rerank_result = retriever.retrieve(
        query,
        top_k_dense=top_k_rrf,
        top_k_bm25=top_k_rrf,
        final_k=top_k_rrf,
    )

    top_chunks = rerank_result.top_k_candidates

    print(f"── Reranked Child Chunks ({len(top_chunks)} returned) ──")
    for i, c in enumerate(top_chunks):
        preview = c.chunk_text.replace("\n", " ")
        safe_print(f"  [{i+1}] score={c.rerank_score:.3f} | {preview[:120]}...")

    print(f"\n  Expanding to parent contexts (max={max_contexts})...")
    assembler = ContextAssembler(kb_dir)
    result = assembler.assemble(
        query=query,
        top_child_chunks=top_chunks,
        max_contexts=max_contexts,
    )

    print(f"\n{'='*60}")
    print(f"  RETRIEVAL RESULT")
    print(f"  Query          : {result.query}")
    print(f"  Child chunks   : {len(result.top_child_chunks)}")
    print(f"  Final contexts : {len(result.final_contexts)}")
    print(f"{'='*60}")

    for ctx in result.final_contexts:
        print(f"\n  [Context {ctx.rank}]")
        print(f"    Parent ID   : {ctx.parent_window_id}")
        print(f"    Doc ID      : {ctx.doc_id}")
        safe_print(f"    Text        : {ctx.parent_text.replace(chr(10), ' ')[:400]}...")
        print()
        
    print(f"\n  Generating answer with LLM...")
    GenerationPipeline = _import_generation()
    gen_pipeline = GenerationPipeline()
    try:
        gen_result = gen_pipeline.generate(query, result.final_contexts)
        print(f"\n{'='*60}")
        print(f"  FINAL ANSWER (Model: {gen_result.model_used})")
        print(f"{'='*60}")
        safe_print(f"\n{gen_result.answer}\n")
        print(f"{'-'*60}")
        print("  Sources Cited:")
        for src in gen_result.sources:
            print(f"    - Doc: {src.doc_id}, Parent: {src.parent_window_id}")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        print(f"\n  [ERROR] Failed to generate answer: {e}")


# ── Main menu ────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════╗")
    print("║       RAG Pipeline Orchestrator              ║")
    print("╚══════════════════════════════════════════════╝")
    print("\nWhat would you like to do?")
    print("  [1] Ingest documents → Build/Update KB Index")
    print("  [2] Query a Knowledge Base")
    print("  [0] Exit")

    choice = input("\nSelect option: ").strip()

    if choice == "0":
        return

    elif choice == "1":
        inputs = list_inputs()
        if not inputs:
            print("\nNo files found in inputs/. Add PDFs, DOCX, TXT or ZIPs and try again.")
            return

        print("\nAvailable input files:")
        for i, f in enumerate(inputs):
            print(f"  [{i}] {f}")

        selection = input("\nFile numbers to process (e.g. 0,1,2 or 'all'): ").strip()
        if selection.lower() == "all":
            selected_files = inputs
        else:
            try:
                indices = [int(x) for x in selection.split(",")]
                selected_files = [inputs[i] for i in indices]
            except (ValueError, IndexError):
                print("Invalid selection.")
                return

        kb_dir = pick_kb()
        print(f"\n  Using KB: {kb_dir}")
        run_ingest_to_index(kb_dir, selected_files)
        print("\n✔ Pipeline complete.")

    elif choice == "2":
        kbs = list_existing_kbs()
        if not kbs:
            print("\nNo KBs found. Run option 1 first.")
            return

        print("\nAvailable KBs:")
        for i, kb in enumerate(kbs):
            print(f"  [{i}] {kb}")

        idx = int(input("Select KB number: ").strip())
        run_query(kbs[idx])

    else:
        print("Unknown option.")


if __name__ == "__main__":
    main()

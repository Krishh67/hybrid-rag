import os
import json
import logging
from pathlib import Path

from ingestion.pipeline import ingest_path
from chunking.pipeline import chunk_document
from embedding.pipeline import embed_document
from indexing.pipeline import index_kb
from indexing.manifest import load_manifest, is_duplicate

logger = logging.getLogger(__name__)

def run_ingestion_generator(kb_dir: str, file_paths: list, progress_callback=None):
    """
    Generator that yields progress status strings so the UI can update live.
    Yields: dict with keys "stage", "message", "progress_pct"
    """
    yield {"stage": "init", "message": "Starting pipeline...", "progress_pct": 0}
    
    for sub in ("parsed", "chunks", "embeddings", "index", "parent_windows"):
        os.makedirs(os.path.join(kb_dir, sub), exist_ok=True)
        
    manifest_dir = Path(kb_dir) / "index"
    kb_id = os.path.basename(kb_dir)
    kb_manifest = load_manifest(manifest_dir, kb_id)
    
    total_files = len(file_paths)
    
    for i, file_path in enumerate(file_paths):
        yield {"stage": "parsing", "message": f"Parsing {os.path.basename(file_path)}...", "progress_pct": int(10 + (i / total_files) * 20)}
        
        try:
            parsed_docs = ingest_path(file_path)
        except Exception as e:
            yield {"stage": "error", "message": f"Error parsing {file_path}: {e}", "progress_pct": 0}
            continue
            
        if not parsed_docs:
            continue
            
        for parsed_doc in parsed_docs:
            if is_duplicate(kb_manifest, parsed_doc.content_hash):
                yield {"stage": "parsing", "message": f"Skipped duplicate: {os.path.basename(file_path)}", "progress_pct": int(10 + ((i+0.5) / total_files) * 20)}
                continue
                
            parsed_out = os.path.join(kb_dir, "parsed", f"{parsed_doc.doc_id}.json")
            with open(parsed_out, "w", encoding="utf-8") as f:
                f.write(parsed_doc.model_dump_json(indent=2))
                
            yield {"stage": "chunking", "message": f"Chunking {os.path.basename(file_path)}...", "progress_pct": int(30 + (i / total_files) * 20)}
            chunks, parent_windows = chunk_document(parsed_doc)
            
            chunks_out = os.path.join(kb_dir, "chunks", f"{parsed_doc.doc_id}_chunks.json")
            with open(chunks_out, "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in chunks], f, indent=2)
                
            pw_out = os.path.join(kb_dir, "parent_windows", f"{parsed_doc.doc_id}_parent_windows.json")
            with open(pw_out, "w", encoding="utf-8") as f:
                json.dump(parent_windows, f, indent=2)
                
            yield {"stage": "embedding", "message": f"Embedding {len(chunks)} chunks for {os.path.basename(file_path)}...", "progress_pct": int(50 + (i / total_files) * 30)}
            
            def embed_cb(processed, total):
                if progress_callback:
                    local_pct = processed / total if total > 0 else 1.0
                    overall_pct = int(50 + (i / total_files) * 30 + local_pct * (30 / total_files))
                    msg = f"Embedding {total} chunks for {os.path.basename(file_path)}... ({processed}/{total})"
                    progress_callback({"stage": "embedding", "message": msg, "progress_pct": overall_pct})
                    
            embed_document(chunks, kb_dir, parsed_doc.doc_id, progress_callback=embed_cb)
            
    yield {"stage": "indexing", "message": "Building Knowledge Base Index...", "progress_pct": 90}
    try:
        result = index_kb(kb_dir)
        msg = f"Index updated: {result.new_chunks_added} chunks added. Total: {result.total_chunks}."
        yield {"stage": "completed", "message": msg, "progress_pct": 100}
    except Exception as e:
        yield {"stage": "error", "message": f"Error during indexing: {e}", "progress_pct": 90}

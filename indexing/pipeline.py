"""Stage 5 indexing pipeline.

Entry point:
    ``index_kb(kb_dir: str) -> IndexResult``  — full build / incremental update.
    ``index_document(doc_id, kb_dir) -> IndexResult``  — single-document convenience.

Algorithm (``index_kb``):
1. Scan ``{kb_dir}/embeddings/`` for all ``{doc_id}_dense.npy`` + ``{doc_id}_meta.json`` pairs.
2. Load the existing index state (FAISS + BM25 + metadata + manifest).
3. Identify doc_ids not yet indexed (incremental — already-indexed docs are skipped).
4. For each new document:
   a. Load its ``_dense.npy`` and ``_meta.json`` from embeddings/.
   b. Load its ``_chunks.json`` from chunks/ to get chunk_text + metadata.
   c. Build ``RetrievalRecord`` objects for this document.
5. Stack all new dense vectors; append to FAISS index.
6. Extract chunk_text list; append to BM25 index.
7. Extend metadata list; reassign global_chunk_id to preserve order.
8. Validate alignment: ``faiss.ntotal == len(metadata) == manifest.total_chunks``.
9. Persist all four outputs to ``{kb_dir}/index/``.
10. Return ``IndexResult``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .bm25_builder import BM25State, append_to_bm25, build_bm25, load_bm25, save_bm25
from .faiss_builder import (
    append_to_index,
    build_index,
    load_index,
    save_index,
)
from .manifest import load_manifest, save_manifest, update_manifest
from .metadata_builder import (
    RetrievalRecord,
    build_retrieval_records,
    load_chunk_lookup,
    load_metadata,
    save_metadata,
)
from .parent_lookup_builder import load_parent_lookup, save_parent_lookup

logger = logging.getLogger(__name__)

INDEX_SUBDIR = "index"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class IndexResult:
    """Summary of what was built / updated by the pipeline."""

    kb_id: str
    total_chunks: int
    total_documents: int
    new_chunks_added: int
    new_documents_added: int
    index_dir: Path
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_doc_id(npy_path: Path) -> str:
    """Extract doc_id from a filename like ``{doc_id}_dense.npy``."""
    return npy_path.stem.replace("_dense", "")


def _load_embedding_meta(emb_dir: Path, doc_id: str) -> Optional[List[dict]]:
    """Load ``{doc_id}_meta.json`` from the embeddings directory.

    Returns ``None`` and logs an error if the file is missing or unreadable.
    """
    path = emb_dir / f"{doc_id}_meta.json"
    if not path.exists():
        logger.error("Embedding meta file not found: %s", path)
        return None
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return None


def _load_dense_npy(emb_dir: Path, doc_id: str) -> Optional[np.ndarray]:
    """Load ``{doc_id}_dense.npy`` and validate shape/dtype.

    Returns ``None`` and logs an error if the file is missing or invalid.
    """
    path = emb_dir / f"{doc_id}_dense.npy"
    if not path.exists():
        logger.error("Dense npy file not found: %s", path)
        return None
    try:
        arr = np.load(str(path), allow_pickle=False).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 1024:
            logger.error(
                "Dense npy %s has unexpected shape %s — skipped.", path, arr.shape
            )
            return None
        return arr
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None


def _validate_alignment(
    faiss_total: int,
    metadata_len: int,
    manifest_total: int,
    index_dir: Path,
) -> None:
    """Raise ``RuntimeError`` if FAISS / metadata / manifest counts diverge."""
    mismatches = []
    if faiss_total != metadata_len:
        mismatches.append(
            f"FAISS.ntotal={faiss_total} != len(metadata)={metadata_len}"
        )
    if faiss_total != manifest_total:
        mismatches.append(
            f"FAISS.ntotal={faiss_total} != manifest.total_chunks={manifest_total}"
        )
    if mismatches:
        msg = "Index alignment validation FAILED in %s: %s" % (
            index_dir,
            "; ".join(mismatches),
        )
        logger.critical(msg)
        raise RuntimeError(msg)
    logger.info(
        "Index alignment OK: %d chunks in FAISS, metadata, and manifest.",
        faiss_total,
    )


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def index_kb(kb_dir: str) -> IndexResult:
    """Build or incrementally update the full KB index.

    Scans ``{kb_dir}/embeddings/`` for all documents and adds any not yet
    present in the index.  Already-indexed documents are skipped.

    Args:
        kb_dir: Root knowledge-base directory (e.g. ``"knowledge_bases/kb_001"``).

    Returns:
        ``IndexResult`` with counts and the path of the written index directory.
    """
    kb_path = Path(kb_dir)
    kb_id = kb_path.name
    emb_dir = kb_path / "embeddings"
    chunks_dir = kb_path / "chunks"
    index_dir = kb_path / INDEX_SUBDIR

    # ------------------------------------------------------------------
    # Discover all available documents in the embedding directory
    # ------------------------------------------------------------------
    npy_files = sorted(emb_dir.glob("*_dense.npy"))
    if not npy_files:
        logger.warning("No embedding files found in '%s'. Empty KB — nothing to index.", emb_dir)
        # Still write a valid empty manifest so subsequent runs don't crash.
        manifest = load_manifest(index_dir, kb_id)
        save_manifest(manifest, index_dir)
        return IndexResult(
            kb_id=kb_id,
            total_chunks=0,
            total_documents=0,
            new_chunks_added=0,
            new_documents_added=0,
            index_dir=index_dir,
        )

    all_doc_ids = [_extract_doc_id(p) for p in npy_files]
    logger.info("Found %d document(s) in embeddings dir.", len(all_doc_ids))

    # ------------------------------------------------------------------
    # Load existing index state (may be empty on first run)
    # ------------------------------------------------------------------
    manifest = load_manifest(index_dir, kb_id)
    already_indexed: List[str] = manifest.get("indexed_doc_ids", [])

    try:
        faiss_index = load_index(index_dir)
        all_metadata: List[RetrievalRecord] = load_metadata(index_dir)
        bm25_state = load_bm25(index_dir)
        parent_lookup = load_parent_lookup(index_dir)
        logger.info(
            "Loaded existing index: %d chunks, %d docs indexed.",
            faiss_index.ntotal,
            len(already_indexed),
        )
    except FileNotFoundError:
        faiss_index = None
        all_metadata = []
        bm25_state = None
        parent_lookup = {}
        logger.info("No existing index found — building from scratch.")

    # ------------------------------------------------------------------
    # Identify new (not yet indexed) documents
    # ------------------------------------------------------------------
    new_doc_ids = [d for d in all_doc_ids if d not in already_indexed]
    if not new_doc_ids:
        logger.info("All documents already indexed. Nothing to do.")
        total = faiss_index.ntotal if faiss_index else 0
        return IndexResult(
            kb_id=kb_id,
            total_chunks=total,
            total_documents=len(already_indexed),
            new_chunks_added=0,
            new_documents_added=0,
            index_dir=index_dir,
        )

    logger.info("%d new document(s) to index: %s", len(new_doc_ids), new_doc_ids)

    # ------------------------------------------------------------------
    # Load and aggregate new document data
    # ------------------------------------------------------------------
    new_vectors_list: List[np.ndarray] = []
    new_records: List[RetrievalRecord] = []
    failed_docs: List[str] = []
    start_gid = len(all_metadata)

    for doc_id in new_doc_ids:
        emb_meta = _load_embedding_meta(emb_dir, doc_id)
        dense_arr = _load_dense_npy(emb_dir, doc_id)

        if emb_meta is None or dense_arr is None:
            logger.error("Skipping doc '%s' due to load errors.", doc_id)
            failed_docs.append(doc_id)
            continue

        # Sanity: npy rows must match meta entries
        if len(dense_arr) != len(emb_meta):
            logger.error(
                "Doc '%s': dense.npy has %d rows but meta.json has %d entries — skipped.",
                doc_id,
                len(dense_arr),
                len(emb_meta),
            )
            failed_docs.append(doc_id)
            continue

        chunk_lookup = load_chunk_lookup(chunks_dir, doc_id)
        doc_records = build_retrieval_records(
            embedding_meta=emb_meta,
            chunk_lookup=chunk_lookup,
            start_global_id=start_gid + len(new_records),
        )

        # Align vectors to records (some chunks may have been skipped by metadata builder)
        # Build a chunk_id→row mapping from the npy file using meta.json order
        chunk_id_to_row: Dict[str, int] = {
            entry["chunk_id"]: idx for idx, entry in enumerate(emb_meta)
        }
        aligned_vecs = np.array(
            [dense_arr[chunk_id_to_row[r.chunk_id]] for r in doc_records
             if r.chunk_id in chunk_id_to_row],
            dtype=np.float32,
        )
        aligned_records = [
            r for r in doc_records if r.chunk_id in chunk_id_to_row
        ]

        if len(aligned_vecs) == 0:
            logger.warning("Doc '%s' produced no aligned records — skipped.", doc_id)
            failed_docs.append(doc_id)
            continue

        new_vectors_list.append(aligned_vecs)
        new_records.extend(aligned_records)
        logger.info(
            "Loaded doc '%s': %d chunks.", doc_id, len(aligned_records)
        )

    if not new_records:
        logger.warning("No new records to add after processing all new documents.")
        total = faiss_index.ntotal if faiss_index else 0
        return IndexResult(
            kb_id=kb_id,
            total_chunks=total,
            total_documents=len(already_indexed),
            new_chunks_added=0,
            new_documents_added=0,
            index_dir=index_dir,
            warnings=[f"Failed to load: {d}" for d in failed_docs],
        )

    # ------------------------------------------------------------------
    # Stack new vectors
    # ------------------------------------------------------------------
    new_vectors = np.vstack(new_vectors_list).astype(np.float32)
    logger.info("Stacked %d new vectors for indexing.", len(new_vectors))

    # ------------------------------------------------------------------
    # Update / build FAISS index
    # ------------------------------------------------------------------
    if faiss_index is None:
        faiss_index = build_index(new_vectors)
    else:
        faiss_index = append_to_index(faiss_index, new_vectors)

    # ------------------------------------------------------------------
    # Update / build BM25 index
    # ------------------------------------------------------------------
    new_texts = [r.chunk_text for r in new_records]
    if bm25_state is None:
        bm25_state = build_bm25(new_texts)
    else:
        bm25_state = append_to_bm25(bm25_state, new_texts)

    # ------------------------------------------------------------------
    # Extend metadata list and reassign global_chunk_ids
    # ------------------------------------------------------------------
    all_metadata.extend(new_records)
    # Re-assign global_chunk_id sequentially to guarantee alignment
    for i, rec in enumerate(all_metadata):
        rec.global_chunk_id = i

    # ------------------------------------------------------------------
    # Persist all outputs
    # ------------------------------------------------------------------
    save_index(faiss_index, index_dir)
    save_bm25(bm25_state, index_dir)
    save_metadata(all_metadata, index_dir)

    successfully_added_doc_ids = [
        d for d in new_doc_ids if d not in failed_docs
    ]
    
    # ------------------------------------------------------------------
    # Update and save Parent Lookup
    # ------------------------------------------------------------------
    pw_dir = kb_path / "parent_windows"
    for doc_id in successfully_added_doc_ids:
        pw_file = pw_dir / f"{doc_id}_parent_windows.json"
        if pw_file.exists():
            try:
                with pw_file.open(encoding="utf-8") as fh:
                    pw_data = json.load(fh)
                    for w in pw_data:
                        parent_lookup[w["parent_window_id"]] = {
                            "doc_id": w["doc_id"],
                            "char_start": w["char_start"],
                            "char_end": w["char_end"]
                        }
            except Exception as exc:
                logger.error("Failed to parse parent windows for '%s': %s", doc_id, exc)
                
    save_parent_lookup(parent_lookup, index_dir)

    # Collect content_hash provenance from parsed/ files for the manifest.
    new_content_hashes: Dict[str, dict] = {}
    parsed_dir = kb_path / "parsed"
    for doc_id in successfully_added_doc_ids:
        parsed_path = parsed_dir / f"{doc_id}.json"
        if parsed_path.exists():
            try:
                with parsed_path.open(encoding="utf-8") as fh:
                    parsed_data = json.load(fh)
                ch = parsed_data.get("content_hash", "")
                if ch:
                    new_content_hashes[ch] = {
                        "doc_id": doc_id,
                        "original_filename": parsed_data.get("original_filename", ""),
                    }
            except Exception as exc:
                logger.warning("Could not read content_hash for doc '%s': %s", doc_id, exc)

    manifest = update_manifest(
        manifest,
        new_doc_ids=successfully_added_doc_ids,
        new_chunk_count=len(new_records),
        new_content_hashes=new_content_hashes,
    )
    save_manifest(manifest, index_dir)

    # ------------------------------------------------------------------
    # Validate alignment
    # ------------------------------------------------------------------
    _validate_alignment(
        faiss_total=faiss_index.ntotal,
        metadata_len=len(all_metadata),
        manifest_total=manifest["total_chunks"],
        index_dir=index_dir,
    )

    return IndexResult(
        kb_id=kb_id,
        total_chunks=faiss_index.ntotal,
        total_documents=manifest["total_documents"],
        new_chunks_added=len(new_records),
        new_documents_added=len(successfully_added_doc_ids),
        index_dir=index_dir,
        warnings=[f"Failed to load: {d}" for d in failed_docs],
    )


def index_document(doc_id: str, kb_dir: str) -> IndexResult:
    """Convenience wrapper: index a single document (incremental update).

    The document's ``{doc_id}_dense.npy`` and ``{doc_id}_meta.json`` must
    already exist in ``{kb_dir}/embeddings/`` (produced by Stage 4).
    If the document is already in the index, this is a no-op.
    """
    # The general index_kb() will skip already-indexed docs automatically.
    return index_kb(kb_dir)

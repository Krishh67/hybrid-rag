"""KB-wide metadata builder.

Loads Stage 4 ``{doc_id}_meta.json`` files and the corresponding Stage 3
``{doc_id}_chunks.json`` files to construct the ``RetrievalRecord`` list.
Persists to ``metadata.pkl`` for fast reload.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from .schema import RetrievalRecord

logger = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.pkl"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_retrieval_records(
    embedding_meta: List[dict],
    chunk_lookup: Dict[str, dict],
    start_global_id: int = 0,
) -> List[RetrievalRecord]:
    """Construct ``RetrievalRecord`` objects for a single document.

    Args:
        embedding_meta: Parsed list from ``{doc_id}_meta.json`` (Stage 4 output).
                        Each entry has ``chunk_id``, ``embed_text_hash``, etc.
        chunk_lookup:   ``{chunk_id: chunk_dict}`` built from ``{doc_id}_chunks.json``.
                        Provides ``chunk_text``, ``metadata`` (heading_path, etc.).
        start_global_id: Starting ``global_chunk_id`` for this batch (incremental support).

    Returns:
        Ordered list of ``RetrievalRecord`` objects.  Entries whose ``chunk_id``
        is missing from ``chunk_lookup`` are skipped with a warning.
    """
    records: List[RetrievalRecord] = []

    for i, emb_entry in enumerate(embedding_meta):
        chunk_id = emb_entry["chunk_id"]
        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            logger.warning(
                "Chunk '%s' found in embedding meta but missing from chunks file — skipped.",
                chunk_id,
            )
            continue

        meta = chunk.get("metadata", {})

        records.append(
            RetrievalRecord(
                global_chunk_id=start_global_id + len(records),
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                parent_window_id=chunk.get("parent_window_id", "none"),
                segment_id=chunk.get("segment_id", "none"),
                title=meta.get("title"),
                heading_path=meta.get("heading_path", []),
                original_filename=meta.get("original_filename", ""),
                chunk_type=meta.get("chunk_type", "text"),
                chunk_text=chunk.get("chunk_text", ""),
                embed_text_hash=emb_entry.get("embed_text_hash", ""),
            )
        )

    return records


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metadata(records: List[RetrievalRecord], out_dir: Path) -> Path:
    """Pickle the full KB-wide ``RetrievalRecord`` list.

    Returns the path that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / METADATA_FILENAME
    with path.open("wb") as fh:
        pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved metadata → %s  (%d records)", path, len(records))
    return path


def load_metadata(index_dir: Path) -> List[RetrievalRecord]:
    """Load the KB-wide ``RetrievalRecord`` list from pickle.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    path = index_dir / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Metadata pickle not found: {path}")

    with path.open("rb") as fh:
        records = pickle.load(fh)

    if not isinstance(records, list):
        raise TypeError(
            f"Expected list[RetrievalRecord] but got {type(records).__name__}."
        )
    logger.info("Loaded metadata from %s  (%d records)", path, len(records))
    return records


# ---------------------------------------------------------------------------
# Chunk file loader helper
# ---------------------------------------------------------------------------

def load_chunk_lookup(chunks_dir: Path, doc_id: str) -> Dict[str, dict]:
    """Load ``{doc_id}_chunks.json`` and return a ``{chunk_id: chunk_dict}`` map."""
    path = chunks_dir / f"{doc_id}_chunks.json"
    if not path.exists():
        logger.warning("Chunks file not found for doc '%s': %s", doc_id, path)
        return {}

    with path.open(encoding="utf-8") as fh:
        chunks: List[dict] = json.load(fh)

    return {c["chunk_id"]: c for c in chunks}

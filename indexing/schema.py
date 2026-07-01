"""Schema for Stage 5 indexing layer.

Defines the KB-wide metadata record that maps every FAISS row / BM25 doc ID
back to its source chunk and document.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class RetrievalRecord(BaseModel):
    """KB-wide metadata mapping for a single indexed chunk.

    FAISS row index ``i`` and BM25 document index ``i`` both map to
    ``metadata[i]`` — the alignment is strictly maintained by the pipeline.
    """

    global_chunk_id: int
    """Zero-based row index in the KB-wide FAISS index and BM25 corpus.
    Assigned sequentially as chunks are added to the index.
    """

    chunk_id: str
    """Original chunk UUID from Stage 3."""

    doc_id: str
    """Document UUID from Stage 1."""

    parent_window_id: str
    """Foreign key to the parent window store."""

    segment_id: str
    """Foreign key to the segment."""

    title: Optional[str]
    """Document title (may be None for Tier 3 / untitled documents)."""

    heading_path: List[str]
    """Breadcrumb heading path from Stage 3 segmentation."""

    original_filename: str
    """Source filename for display / citation."""

    chunk_type: str
    """'text' | 'table' | 'code' | 'figure'"""

    chunk_text: str
    """Raw chunk text — used as the BM25 corpus document."""

    embed_text_hash: str
    """SHA-256 of the embedded text; useful for dedup and cache checks."""

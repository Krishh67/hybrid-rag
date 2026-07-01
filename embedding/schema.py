from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, Field


class EmbeddedChunk(BaseModel):
    """Output contract for Stage 4 embedding generation."""

    chunk_id: str
    """The chunk ID from Stage 3, used to correlate back to the source Chunk."""

    dense_vector: List[float]
    """1024-dimensional L2-normalized dense embedding from BGE-M3."""

    sparse_vector: Dict[str, float]
    """Lexical-weight sparse vector: token string → weight (non-zero entries only)."""

    embedding_model: str = "BAAI/bge-m3"
    """Model identifier — always 'BAAI/bge-m3' in this build."""

    embedding_model_version: str
    """Pinned model revision hash for auditability; detectable when the model is swapped."""

    embed_text_hash: str
    """SHA-256 hex digest of the embed_text that produced this embedding.
    Used as the cache key — if the text changes, the hash changes and re-embedding triggers."""

    dim: int = 1024
    """Dimensionality of the dense vector; always 1024 for BGE-M3."""

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    """UTC timestamp when this embedding was generated."""

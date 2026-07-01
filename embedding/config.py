"""Configuration for Stage 4 embedding generation.

All values can be overridden via environment variables.
"""
from __future__ import annotations

import os


class EmbeddingConfig:
    """Central config for the embedding module.

    Environment variable overrides (all optional):
        EMBEDDING_MODEL       — model name (default: BAAI/bge-m3)
        EMBEDDING_BATCH_SIZE  — chunks per encode() call (default: 12)
        EMBEDDING_DEVICE      — 'cpu' only in this build (default: cpu)
        FORCE_REEMBED         — set to '1' to bypass cache and re-embed everything
    """

    # --- Model ---
    MODEL_NAME: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
    """BGE-M3 model identifier — hard-pinned to CPU, no GPU fallback."""

    # Hard-pin: regardless of env override, always CPU-only.
    DEVICE: str = "cpu"
    USE_FP16: bool = True

    # Pinned revision so future model swaps are detectable.
    # The FlagEmbedding library will use this when downloading/loading the model.
    MODEL_REVISION: str = "colbert-only"  # Stable tag for BGE-M3 on HuggingFace Hub

    # --- Batching ---
    BATCH_SIZE: int = int(os.environ.get("EMBEDDING_BATCH_SIZE", "12"))
    """Chunks per encode() call. Tune down for low-RAM machines."""

    # --- Cache ---
    FORCE_REEMBED: bool = os.environ.get("FORCE_REEMBED", "0").strip() == "1"
    """If True, bypass cache and re-embed all chunks unconditionally."""

    # --- Storage ---
    EMBEDDINGS_SUBDIR: str = "embeddings"
    """Sub-directory under the KB directory where embedding files are written."""


config = EmbeddingConfig()

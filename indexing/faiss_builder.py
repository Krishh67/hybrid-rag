"""FAISS index builder for the KB-wide dense vector index.

Uses ``IndexFlatIP`` (inner-product) because Stage 4 vectors are already
L2-normalized — inner product equals cosine similarity for unit vectors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILENAME = "faiss.index"
_DIM = 1024


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build a fresh ``IndexFlatIP`` from a 2-D float32 array.

    Args:
        vectors: Shape ``[n, 1024]``, dtype ``float32``.  Must not be empty.

    Returns:
        Populated FAISS index with ``index.ntotal == len(vectors)``.
    """
    if vectors.ndim != 2 or vectors.shape[1] != _DIM:
        raise ValueError(
            f"Expected float32 array of shape [n, {_DIM}], got {vectors.shape}"
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    index = faiss.IndexFlatIP(_DIM)
    index.add(vectors)
    logger.info("Built FAISS IndexFlatIP: %d vectors × %d dim", index.ntotal, _DIM)
    return index


def append_to_index(index: faiss.IndexFlatIP, new_vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Append new vectors to an existing index (incremental update).

    Args:
        index:       Existing populated index.
        new_vectors: Shape ``[m, 1024]``, dtype ``float32``.

    Returns:
        The same index object (mutated in-place) with ``m`` more rows.
    """
    if new_vectors.ndim != 2 or new_vectors.shape[1] != _DIM:
        raise ValueError(
            f"new_vectors must have shape [m, {_DIM}], got {new_vectors.shape}"
        )
    if new_vectors.dtype != np.float32:
        new_vectors = new_vectors.astype(np.float32)

    before = index.ntotal
    index.add(new_vectors)
    logger.info(
        "Appended %d vectors to FAISS index (total: %d → %d)",
        new_vectors.shape[0],
        before,
        index.ntotal,
    )
    return index


def save_index(index: faiss.IndexFlatIP, out_dir: Path) -> Path:
    """Persist the FAISS index to ``{out_dir}/faiss.index``.

    Returns the path that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / INDEX_FILENAME
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index → %s  (ntotal=%d)", path, index.ntotal)
    return path


def load_index(index_dir: Path) -> faiss.IndexFlatIP:
    """Load a previously saved FAISS index.

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError:      if the loaded object is not an ``IndexFlatIP``.
    """
    path = index_dir / INDEX_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")

    index = faiss.read_index(str(path))
    if not isinstance(index, faiss.IndexFlatIP):
        raise RuntimeError(
            f"Expected IndexFlatIP but loaded {type(index).__name__}. "
            "The index file may be from a different build."
        )
    logger.info("Loaded FAISS index from %s  (ntotal=%d)", path, index.ntotal)
    return index

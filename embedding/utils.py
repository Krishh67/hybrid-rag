"""Utility helpers: text hashing, L2 normalization, NumPy I/O."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import List

import numpy as np


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text* (UTF-8 encoded).

    Used as the cache key for each chunk's embed_text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_l2(vec: List[float]) -> List[float]:
    """L2-normalize *vec* in-place (returns the same list).

    BGE-M3's dense head returns vectors that are already normalized when
    ``normalize_embeddings=True`` is passed to ``encode()``.  This function
    is the explicit fallback that makes the normalization contract obvious
    and testable regardless of what the library does internally.

    If the vector is the zero vector (e.g. degenerate input) the original
    list is returned unchanged — callers should handle/flag this separately.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def save_npy(path: Path, arr: np.ndarray) -> None:
    """Save a NumPy array to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def load_npy(path: Path) -> np.ndarray:
    """Load a NumPy array from *path*.

    Raises ``FileNotFoundError`` if the file does not exist — callers should
    check existence themselves if the file is expected to be optional.
    """
    return np.load(str(path), allow_pickle=False)

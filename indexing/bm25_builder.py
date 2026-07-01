"""BM25 lexical index builder.

Uses ``rank_bm25.BM25Okapi`` — a well-tested, pure-Python BM25 implementation
with no native dependencies.  The BM25 corpus is built from ``chunk_text``
stored in the KB-wide ``RetrievalRecord`` list.

Serialization: ``pickle`` (standard for ML objects that expose no native
save API).  The pickle file stores a ``BM25State`` dataclass that includes
both the fitted model and the tokenized corpus, enabling fast reload.
"""
from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

BM25_FILENAME = "bm25.pkl"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25.

    Lowercases and splits on non-alphanumeric characters.  Preserves numbers
    (important for academic / structured content like grade records).
    """
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


# ---------------------------------------------------------------------------
# State container — persisted to pickle
# ---------------------------------------------------------------------------

@dataclass
class BM25State:
    """Persisted BM25 state: fitted model + tokenized corpus."""

    model: BM25Okapi
    tokenized_corpus: List[List[str]] = field(default_factory=list)
    """Kept alongside the model so incremental updates can rebuild quickly."""


# ---------------------------------------------------------------------------
# Build / update
# ---------------------------------------------------------------------------

def build_bm25(texts: List[str]) -> BM25State:
    """Build a fresh BM25 index from a list of chunk texts.

    Args:
        texts: Ordered list of ``chunk_text`` strings, aligned with the
               FAISS index rows (same order as ``RetrievalRecord`` list).

    Returns:
        A populated ``BM25State``.
    """
    tokenized = [tokenize(t) for t in texts]
    model = BM25Okapi(tokenized)
    logger.info("Built BM25 index over %d documents.", len(tokenized))
    return BM25State(model=model, tokenized_corpus=tokenized)


def append_to_bm25(state: BM25State, new_texts: List[str]) -> BM25State:
    """Incrementally update a BM25 index with new documents.

    BM25Okapi does not support in-place append — we rebuild from the full
    combined corpus.  This is fast because the tokenized corpus is cached
    in ``BM25State``.

    Args:
        state:     Existing ``BM25State``.
        new_texts: Ordered list of new ``chunk_text`` strings to add.

    Returns:
        A new ``BM25State`` built over the combined corpus.
    """
    new_tokenized = [tokenize(t) for t in new_texts]
    combined = state.tokenized_corpus + new_tokenized
    model = BM25Okapi(combined)
    logger.info(
        "Rebuilt BM25 index: %d existing + %d new = %d total documents.",
        len(state.tokenized_corpus),
        len(new_tokenized),
        len(combined),
    )
    return BM25State(model=model, tokenized_corpus=combined)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_bm25(state: BM25State, out_dir: Path) -> Path:
    """Pickle the BM25 state to ``{out_dir}/bm25.pkl``.

    Returns the path that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / BM25_FILENAME
    with path.open("wb") as fh:
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "Saved BM25 index → %s  (%d docs)", path, len(state.tokenized_corpus)
    )
    return path


def load_bm25(index_dir: Path) -> BM25State:
    """Load a previously saved BM25 state from pickle.

    Raises:
        FileNotFoundError: if the file does not exist.
        TypeError:         if the pickled object is not a ``BM25State``.
    """
    path = index_dir / BM25_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"BM25 pickle not found: {path}")

    with path.open("rb") as fh:
        state = pickle.load(fh)

    if not isinstance(state, BM25State):
        raise TypeError(
            f"Expected BM25State but loaded {type(state).__name__}. "
            "The pickle may be from an incompatible build."
        )
    logger.info(
        "Loaded BM25 index from %s  (%d docs)", path, len(state.tokenized_corpus)
    )
    return state

"""Per-document hash-based embedding cache.

Uses existing ``{doc_id}_meta.json`` files as the cache store — no separate
cache database is required.  On a re-run, only chunks whose ``embed_text``
has changed (detected by SHA-256 hash mismatch) are re-embedded.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .schema import EmbeddedChunk
from .utils import save_npy

logger = logging.getLogger(__name__)


def load_cache(meta_path: Path) -> Dict[str, EmbeddedChunk]:
    """Load an existing ``{doc_id}_meta.json`` and build a hash→EmbeddedChunk lookup.

    Args:
        meta_path: Path to the ``{doc_id}_meta.json`` file.

    Returns:
        Dict mapping ``embed_text_hash`` → ``EmbeddedChunk``.
        Returns an empty dict if the file does not exist or cannot be parsed.
    """
    if not meta_path.exists():
        return {}

    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            raw: List[dict] = json.load(fh)
        cache: Dict[str, EmbeddedChunk] = {}
        for item in raw:
            ec = EmbeddedChunk.model_validate(item)
            cache[ec.embed_text_hash] = ec
        logger.debug("Loaded %d cached embeddings from %s", len(cache), meta_path)
        return cache
    except Exception as exc:
        logger.warning(
            "Could not parse cache file %s (will re-embed everything): %s",
            meta_path,
            exc,
        )
        return {}


def merge_and_save(
    chunk_ids: List[str],
    chunk_hashes: List[str],
    new_results: Dict[str, EmbeddedChunk],
    cached: Dict[str, EmbeddedChunk],
    out_dir: Path,
    doc_id: str,
) -> List[EmbeddedChunk]:
    """Merge newly-computed embeddings with cached ones and persist to disk.

    Guarantees that:
    * The output ``meta.json`` and ``.npy`` are in the same order as the
      original chunk list (``chunk_ids`` / ``chunk_hashes``).
    * The ``.npy`` row *i* corresponds exactly to ``meta.json`` entry *i*.
    * Chunks that failed encoding (not in ``new_results`` and not in ``cached``)
      are omitted from output with a warning — they will be retried next run.

    Args:
        chunk_ids:    Ordered list of chunk IDs (original Stage 3 order).
        chunk_hashes: Ordered list of ``embed_text_hash`` strings, aligned with chunk_ids.
        new_results:  ``{embed_text_hash: EmbeddedChunk}`` for freshly-encoded chunks.
        cached:       ``{embed_text_hash: EmbeddedChunk}`` loaded from the previous run.
        out_dir:      Directory to write ``{doc_id}_dense.npy`` and ``{doc_id}_meta.json``.
        doc_id:       Document identifier used for file naming.

    Returns:
        Ordered list of ``EmbeddedChunk`` objects (same order as input chunks).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / f"{doc_id}_meta.json"
    npy_path = out_dir / f"{doc_id}_dense.npy"

    ordered: List[EmbeddedChunk] = []
    skipped = 0

    for cid, h in zip(chunk_ids, chunk_hashes):
        ec: Optional[EmbeddedChunk] = new_results.get(h) or cached.get(h)
        if ec is None:
            logger.warning(
                "Chunk '%s' (hash %s) has no embedding — skipped from output.", cid, h
            )
            skipped += 1
            continue
        ordered.append(ec)

    if skipped:
        logger.warning("%d chunk(s) were omitted from the embedding output.", skipped)

    if not ordered:
        logger.warning("No embeddings to persist for doc '%s'.", doc_id)
        return ordered

    # --- Write dense .npy ---
    dense_matrix = np.array(
        [ec.dense_vector for ec in ordered], dtype=np.float32
    )
    save_npy(npy_path, dense_matrix)
    logger.info(
        "Saved dense matrix %s → %s", dense_matrix.shape, npy_path
    )

    # --- Write meta .json ---
    meta_list = [ec.model_dump(mode="json") for ec in ordered]
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta_list, fh, indent=2, default=str)
    logger.info("Saved meta JSON → %s (%d entries)", meta_path, len(ordered))

    return ordered

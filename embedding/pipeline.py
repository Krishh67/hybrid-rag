"""Stage 4 embedding pipeline.

Entry point: ``embed_document(chunks, kb_dir, doc_id) -> list[EmbeddedChunk]``

Orchestrates:
1. Hash each chunk's embed_text.
2. Load per-document cache from a previous run (if any).
3. Identify which chunks need encoding (cache misses).
4. Batch-encode cache-miss chunks via model_wrapper.encode_batch().
5. Validate each result (dim, sparse non-empty, no degenerate inputs).
6. Merge with cached results, persist .npy + meta.json, return ordered list.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Callable

from chunking.schema import Chunk

from .cache import load_cache, merge_and_save
from .config import config
from .model_wrapper import encode_batch, get_model_version
from .schema import EmbeddedChunk
from .utils import hash_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_EXPECTED_DIM = 1024


def _validate_embedding(
    dense: List[float],
    sparse: Dict[str, float],
    chunk_id: str,
    embed_text: str,
) -> bool:
    """Run lightweight sanity checks on a single embedding result.

    Returns True if the result is valid, False otherwise (with logged warnings).
    """
    ok = True

    if len(dense) != _EXPECTED_DIM:
        logger.error(
            "Chunk '%s': dense vector has dim=%d, expected %d.",
            chunk_id,
            len(dense),
            _EXPECTED_DIM,
        )
        ok = False

    if not sparse:
        logger.warning(
            "Chunk '%s' produced an empty sparse vector — this usually means "
            "near-empty or degenerate input. embed_text preview: %r",
            chunk_id,
            embed_text[:100],
        )
        # Don't set ok=False — an empty sparse is flagged but not fatal.

    return ok


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------


def embed_document(
    chunks: List[Chunk],
    kb_dir: str,
    doc_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[EmbeddedChunk]:
    """Embed all chunks for a single document, using cache to skip unchanged chunks.

    Args:
        chunks:  Ordered list of ``Chunk`` objects from Stage 3.
        kb_dir:  Root knowledge-base directory (e.g. ``"knowledge_bases/kb_001"``).
        doc_id:  Document ID (used for file naming and logging).

    Returns:
        Ordered list of ``EmbeddedChunk`` objects, aligned with the input *chunks*.
        Chunks that fail encoding AND are not in the cache are omitted (logged).
    """
    if not chunks:
        logger.warning("embed_document called with empty chunk list for doc '%s'.", doc_id)
        return []

    out_dir = Path(kb_dir) / config.EMBEDDINGS_SUBDIR
    meta_path = out_dir / f"{doc_id}_meta.json"

    # ------------------------------------------------------------------
    # Step 1: Hash every chunk's embed_text
    # ------------------------------------------------------------------
    chunk_ids: List[str] = []
    chunk_hashes: List[str] = []
    hash_to_chunk: Dict[str, Chunk] = {}

    for chunk in chunks:
        embed_text = chunk.embed_text
        h = hash_text(embed_text)
        chunk_ids.append(chunk.chunk_id)
        chunk_hashes.append(h)
        hash_to_chunk[h] = chunk

    # ------------------------------------------------------------------
    # Step 2: Load existing cache
    # ------------------------------------------------------------------
    cached: Dict[str, EmbeddedChunk] = {}
    if not config.FORCE_REEMBED:
        cached = load_cache(meta_path)
    else:
        logger.info("FORCE_REEMBED=True — bypassing cache for doc '%s'.", doc_id)

    # ------------------------------------------------------------------
    # Step 3: Separate cache hits from misses
    # ------------------------------------------------------------------
    to_encode: List[Chunk] = []
    to_encode_hashes: List[str] = []

    for h, chunk in hash_to_chunk.items():
        if h not in cached:
            # Filter out empty/whitespace embed_text — warn and skip.
            if not chunk.embed_text.strip():
                logger.warning(
                    "Chunk '%s' has empty/whitespace embed_text — skipping (no embedding generated).",
                    chunk.chunk_id,
                )
                continue
            to_encode.append(chunk)
            to_encode_hashes.append(h)

    n_cached = len(chunk_hashes) - len(to_encode)
    logger.info(
        "Doc '%s': %d chunks total — %d from cache, %d to encode.",
        doc_id,
        len(chunks),
        n_cached,
        len(to_encode),
    )

    if not to_encode and progress_callback:
        progress_callback(n_cached, len(chunks))

    # ------------------------------------------------------------------
    # Step 4: Batch-encode cache misses
    # ------------------------------------------------------------------
    new_results: Dict[str, EmbeddedChunk] = {}
    model_version = get_model_version() if not to_encode else ""

    if to_encode:
        model_version = get_model_version() if not model_version else model_version
        batches = _make_batches(to_encode, config.BATCH_SIZE)
        total_batches = len(batches)
        t_start = time.monotonic()
        encoded_so_far = 0

        for batch_idx, batch_chunks in enumerate(batches):
            batch_texts = [c.embed_text for c in batch_chunks]
            batch_hashes = [hash_text(t) for t in batch_texts]

            logger.info(
                "Encoding batch %d/%d (%d chunks) for doc '%s'…",
                batch_idx + 1,
                total_batches,
                len(batch_chunks),
                doc_id,
            )

            results = encode_batch(batch_texts)

            for i, (chunk, h, result) in enumerate(
                zip(batch_chunks, batch_hashes, results)
            ):
                if result is None:
                    logger.error(
                        "Chunk '%s' failed encoding — will be omitted from output.",
                        chunk.chunk_id,
                    )
                    continue

                dense, sparse = result

                if not _validate_embedding(dense, sparse, chunk.chunk_id, chunk.embed_text):
                    logger.error(
                        "Chunk '%s' failed validation — skipped.", chunk.chunk_id
                    )
                    continue

                ec = EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    dense_vector=dense,
                    sparse_vector=sparse,
                    embedding_model=config.MODEL_NAME,
                    embedding_model_version=model_version,
                    embed_text_hash=h,
                    dim=len(dense),
                    generated_at=datetime.now(timezone.utc),
                )
                new_results[h] = ec

            encoded_so_far += len(batch_chunks)

            # Progress logging with estimated time remaining
            elapsed = time.monotonic() - t_start
            if encoded_so_far > 0 and total_batches > 1:
                rate = encoded_so_far / elapsed
                remaining = (len(to_encode) - encoded_so_far) / max(rate, 1e-9)
                logger.info(
                    "Progress: %d/%d chunks encoded — %.1f chunks/s — ETA %.0fs",
                    encoded_so_far,
                    len(to_encode),
                    rate,
                    remaining,
                )
            if progress_callback:
                progress_callback(n_cached + encoded_so_far, len(chunks))

    # ------------------------------------------------------------------
    # Step 5: Merge, persist, return
    # ------------------------------------------------------------------
    ordered = merge_and_save(
        chunk_ids=chunk_ids,
        chunk_hashes=chunk_hashes,
        new_results=new_results,
        cached=cached,
        out_dir=out_dir,
        doc_id=doc_id,
    )

    logger.info(
        "embed_document complete for '%s': %d EmbeddedChunks written.",
        doc_id,
        len(ordered),
    )
    return ordered


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_batches(chunks: List[Chunk], batch_size: int) -> List[List[Chunk]]:
    """Split *chunks* into sub-lists of at most *batch_size* items."""
    return [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

"""KB manifest: tracks high-level KB statistics and provenance.

The manifest is a lightweight JSON file that records what is in the index,
allowing quick health checks and human inspection without loading pickle/npy files.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"


def _default_manifest(kb_id: str) -> dict:
    return {
        "kb_id": kb_id,
        "total_documents": 0,
        "total_chunks": 0,
        "embedding_model": "BAAI/bge-m3",
        "indexed_doc_ids": [],
        "indexed_content_hashes": {},
        # ^ maps content_hash -> {"doc_id": ..., "original_filename": ...}
        # stored as a dict for O(1) lookup and human-readable provenance
    }


def load_manifest(index_dir: Path, kb_id: str) -> dict:
    """Load an existing manifest, or return a fresh default if none exists."""
    path = index_dir / MANIFEST_FILENAME
    if not path.exists():
        return _default_manifest(kb_id)

    with path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    # Back-fill keys missing from older manifests (forward-compatible)
    manifest.setdefault("indexed_doc_ids", [])
    manifest.setdefault("embedding_model", "BAAI/bge-m3")
    manifest.setdefault("indexed_content_hashes", {})
    return manifest


def update_manifest(
    manifest: dict,
    new_doc_ids: List[str],
    new_chunk_count: int,
    new_content_hashes: dict | None = None,
) -> dict:
    """Update manifest counts after adding new documents.

    Args:
        manifest:           Existing manifest dict (mutated in-place).
        new_doc_ids:        List of newly added ``doc_id`` strings.
        new_chunk_count:    Number of new chunks added to the index.
        new_content_hashes: Optional ``{content_hash: {"doc_id": ..., "original_filename": ...}}``
                            mapping for newly indexed documents.

    Returns:
        The updated manifest dict.
    """
    existing_ids: List[str] = manifest.get("indexed_doc_ids", [])
    added = [d for d in new_doc_ids if d not in existing_ids]

    manifest["indexed_doc_ids"] = existing_ids + added
    manifest["total_documents"] = len(manifest["indexed_doc_ids"])
    manifest["total_chunks"] = manifest.get("total_chunks", 0) + new_chunk_count

    if new_content_hashes:
        existing_hashes: dict = manifest.setdefault("indexed_content_hashes", {})
        for h, info in new_content_hashes.items():
            if h not in existing_hashes:
                existing_hashes[h] = info

    return manifest


def is_duplicate(manifest: dict, content_hash: str) -> bool:
    """Return True if *content_hash* is already present in the manifest.

    Args:
        manifest:     Loaded manifest dict.
        content_hash: SHA-256 hex digest of the raw file bytes.

    Returns:
        True  → document is already in the KB; caller should skip all stages.
        False → document is new; proceed normally.
    """
    return content_hash in manifest.get("indexed_content_hashes", {})


def save_manifest(manifest: dict, out_dir: Path) -> Path:
    """Write the manifest to ``{out_dir}/manifest.json``.

    Returns the path that was written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_FILENAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info(
        "Saved manifest → %s  (docs=%d, chunks=%d)",
        path,
        manifest["total_documents"],
        manifest["total_chunks"],
    )
    return path

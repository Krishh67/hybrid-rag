import json
import logging
from pathlib import Path
from typing import Dict, Optional

from indexing.parent_lookup_builder import load_parent_lookup

logger = logging.getLogger(__name__)


class ParentWindowResolver:
    """
    Resolves a parent_window_id to its actual text using O(1) span lookup.

    Flow:
        parent_window_id
         → parent_lookup.pkl  (doc_id, char_start, char_end)
         → parsed/{doc_id}.json  (full_text)
         → full_text[char_start:char_end]
    """

    def __init__(self, kb_dir: str):
        self.kb_dir     = Path(kb_dir)
        self.parsed_dir = self.kb_dir / "parsed"
        self.index_dir  = self.kb_dir / "index"

        # doc_id → full_text  (lazy loaded, kept in memory)
        self._document_cache: Dict[str, str] = {}
        self._lookup: Dict[str, dict] = {}
        self._build_lookup()

    def _build_lookup(self) -> None:
        """Load parent_lookup.pkl into memory."""
        try:
            self._lookup = load_parent_lookup(self.index_dir)
        except Exception as exc:
            logger.error("Failed to load parent lookup: %s", exc)
            self._lookup = {}

    def resolve(self, parent_window_id: str) -> Optional[str]:
        """Return the parent text for the given ID, or None if not resolvable."""
        if parent_window_id == "none" or parent_window_id not in self._lookup:
            return None

        record     = self._lookup[parent_window_id]
        doc_id     = record["doc_id"]
        char_start = record["char_start"]
        char_end   = record["char_end"]

        # Lazy-load document full text
        if doc_id not in self._document_cache:
            doc_path = self.parsed_dir / f"{doc_id}.json"
            if not doc_path.exists():
                logger.error("Parsed document not found: %s", doc_path)
                return None
            try:
                with open(doc_path, "r", encoding="utf-8") as fh:
                    self._document_cache[doc_id] = json.load(fh).get("full_text", "")
            except Exception as exc:
                logger.error("Failed to load %s: %s", doc_path, exc)
                return None

        full_text = self._document_cache[doc_id]

        if char_start < 0 or char_end > len(full_text):
            logger.warning(
                "Span out of bounds for parent '%s' (text len=%d)", parent_window_id, len(full_text)
            )

        return full_text[char_start:char_end]

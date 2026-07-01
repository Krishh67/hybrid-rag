import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

from ..schema import ParsedDocument, HeadingEntry, TableEntry
from ..config import config
from ..utils import compute_file_hash, logger, extract_text_tables

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


# ---------------------------------------------------------------------------
# False-positive guard: patterns that look structural but are almost never
# headings (figures, tables, captions, page numbers, lone years, etc.)
# ---------------------------------------------------------------------------
_FALSE_POSITIVE_RE = re.compile(
    r"""
    ^(
        (figure|fig\.?|table|tbl\.?|chart|exhibit|plate|scheme)\s*[\d\w]  # captions
        | page\s+\d+                                                        # page refs
        | \d{1,4}$                                                          # lone numbers / years
        | \b(et\s+al|ibid|op\.?\s*cit)\b                                   # citation fragments
        | [\W_]{3,}$                                                        # repeated punctuation / separators
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Numbered heading: "1 Intro", "1. Intro", "2.3 Methods", "Chapter 4", "Appendix A"
_NUMBERED_HEADING_RE = re.compile(
    r"""
    ^(
        (chapter|appendix|section|part|annex)\s+[\dA-Z]          # named sections
        | \d+(\.\d+)*\.?\s+\S                                     # numeric like 1. or 2.3.4
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Tier-2 heading detection helpers
# ---------------------------------------------------------------------------

def _collect_font_stats(doc: fitz.Document) -> Dict[float, float]:
    """
    Single-pass over all spans to build a character-weighted font-size
    frequency map.  Returns {font_size: total_char_count}.

    Complexity: O(total_spans)
    """
    font_stats: Dict[float, float] = defaultdict(float)
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:          # skip image blocks
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    if text.strip():
                        # round to 0.5pt to collapse near-identical sizes
                        size = round(span["size"] * 2) / 2
                        font_stats[size] += len(text)
    return font_stats


def _dominant_body_size(font_stats: Dict[float, float]) -> float:
    """
    Return the font size that accounts for the most character data.
    This is the body text size of the document.
    Falls back to 12.0 if stats are empty (should not happen in practice).
    """
    if not font_stats:
        return 12.0
    return max(font_stats, key=lambda s: font_stats[s])


def _infer_heading_levels(sizes: List[float]) -> Dict[float, int]:
    """
    Given a sorted-descending list of distinct heading font sizes found in
    the document, cluster them into heading levels (1 = largest).

    Two sizes belong to the same cluster when their gap is < 1.5 pt.
    Each cluster becomes the next heading level.

    Complexity: O(k log k) where k = number of distinct sizes (typically < 10).
    """
    if not sizes:
        return {}

    sorted_sizes = sorted(set(sizes), reverse=True)
    level_map: Dict[float, int] = {}
    current_level = 1
    prev = sorted_sizes[0]
    level_map[prev] = current_level

    for size in sorted_sizes[1:]:
        if (prev - size) >= 1.5:           # meaningful gap → new level
            current_level += 1
        level_map[size] = current_level
        prev = size

    return level_map


def _score_span_as_heading(
    span: dict,
    line: dict,
    page_width: float,
    body_font_size: float,
) -> float:
    """
    Compute a numeric confidence score for a single span being a heading.
    Returns a float; caller decides the acceptance threshold.

    Signals and their weights:
        font_ratio ≥ 1.1   → +1.5  (slightly larger than body)
        font_ratio ≥ 1.3   → +1.0  (bonus for significantly larger)
        bold flag           → +1.5
        numbered heading    → +2.0
        short line (≤ 80)   → +1.0
        all-caps            → +0.5  (low weight — common in footers too)
        centered            → +0.5

    Rejection (returns 0.0 immediately):
        - matches false-positive pattern
        - font_ratio < 1.05 AND no bold AND no numbered heading
    """
    text = span["text"].strip()
    if not text or len(text) < 2:
        return 0.0

    # Reject decorative/separator lines: must contain at least one alphanumeric
    # character. Catches underscores, dashes, dots, tildes, box-drawing chars, etc.
    if not any(c.isalnum() for c in text):
        return 0.0

    if _FALSE_POSITIVE_RE.match(text):
        return 0.0

    span_size = round(span["size"] * 2) / 2
    font_ratio = span_size / body_font_size if body_font_size > 0 else 1.0

    is_bold = bool(span.get("flags", 0) & 2**4)  # bit 4 = bold in PyMuPDF
    is_numbered = bool(_NUMBERED_HEADING_RE.match(text))

    # Fast rejection: must have at least one strong signal to continue scoring
    if font_ratio < 1.05 and not is_bold and not is_numbered:
        return 0.0

    score = 0.0

    # --- Font size signals ---
    if font_ratio >= 1.1:
        score += 1.5
    if font_ratio >= 1.3:
        score += 1.0   # bonus, stacks with the above

    # --- Bold ---
    if is_bold:
        score += 1.5

    # --- Numbered heading pattern ---
    if is_numbered:
        score += 2.0

    # --- Line-level signals (require the parent line object) ---
    # Reconstruct the full line text to check for standalone / short line
    line_text = "".join(s["text"] for s in line["spans"]).strip()

    if len(line_text) <= 80:
        score += 1.0

    if len(line_text) <= 120:
        # Centered heuristic: origin x is roughly in the middle third of the page
        if page_width > 0:
            x0 = line["bbox"][0]
            x1 = line["bbox"][2]
            line_center = (x0 + x1) / 2
            if abs(line_center - page_width / 2) < page_width * 0.15:
                score += 0.5

    # --- All-caps (low weight) ---
    if text.isupper() and len(text) > 3:
        score += 0.5

    return score


# ---------------------------------------------------------------------------
# Main Tier-2 entry point — replaces the original heuristic block
# ---------------------------------------------------------------------------

def _extract_tier2_headings(
    doc: fitz.Document,
    full_text_parts: List[str],
    font_stats: Dict[float, float],
) -> Tuple[List[HeadingEntry], int]:
    """
    Perform Tier-2 heading detection over the already-extracted page texts.

    Args:
        doc:             Open fitz.Document (for span-level data).
        full_text_parts: List of per-page text strings (already extracted).
        font_stats:      Character-weighted font frequency map from
                         _collect_font_stats().

    Returns:
        (headings, structure_tier)  — structure_tier is 2 if any headings
        were found, 3 otherwise.

    Complexity: O(total_spans) — single pass per page.
    """
    SCORE_THRESHOLD = 3.0   # minimum score to accept a span as a heading

    body_font_size = _dominant_body_size(font_stats)

    # --- Pass 1: score every span, collect candidate sizes ---
    # We need two passes through the pages: first to collect candidate sizes
    # for level inference, then to assign levels.  Both are O(total_spans) and
    # we do NOT re-read all text; we reuse full_text_parts for offsets.
    #
    # To avoid two full doc scans we do it in one loop and defer level
    # assignment until after we know all sizes.

    raw_candidates = []   # list of (page_num, span, line, score, page_width)

    for page_num, page in enumerate(doc, 1):
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    score = _score_span_as_heading(span, line, page_width, body_font_size)
                    if score >= SCORE_THRESHOLD:
                        raw_candidates.append((page_num, span, line, score, page_width))

    if not raw_candidates:
        return [], 3

    # --- Infer heading levels from the candidate font sizes ---
    candidate_sizes = [round(c[1]["size"] * 2) / 2 for c in raw_candidates]
    level_map = _infer_heading_levels(candidate_sizes)

    # --- Pass 2: build HeadingEntry list with reliable char offsets ---
    #
    # Reliable offset strategy:
    #   • We maintain a `search_from` cursor *per page* so that when the same
    #     heading text appears more than once, we find the next occurrence
    #     rather than always the first.
    #   • page_start_offset is pre-computed from full_text_parts so we never
    #     re-scan earlier pages.
    #
    # Build page start offsets once — O(pages).
    page_start_offsets: List[int] = []
    cumulative = 0
    for part in full_text_parts:
        page_start_offsets.append(cumulative)
        cumulative += len(part) + 1  # +1 for the "\n" join separator

    # search_from[page_num] tracks our within-page scan cursor (1-indexed pages)
    search_from: Dict[int, int] = defaultdict(int)

    headings: List[HeadingEntry] = []

    # Deduplicate: track (page_num, text) pairs so the same text on the same
    # page only becomes one heading entry (e.g. multi-span bold headings).
    seen: set = set()

    for page_num, span, line, score, page_width in raw_candidates:
        text = span["text"].strip()
        key = (page_num, text)
        if key in seen:
            continue
        seen.add(key)

        span_size = round(span["size"] * 2) / 2
        level = level_map.get(span_size, 1)

        # Compute char_offset relative to full document text
        page_idx = page_num - 1
        if page_idx < len(full_text_parts):
            page_text = full_text_parts[page_idx]
            local_start = search_from[page_num]
            pos = page_text.find(text, local_start)
            if pos == -1:
                # Fallback: search from the beginning of the page
                pos = page_text.find(text)
            if pos != -1:
                search_from[page_num] = pos + len(text)   # advance cursor
                char_offset = page_start_offsets[page_idx] + pos
            else:
                char_offset = page_start_offsets[page_idx]
        else:
            char_offset = 0

        headings.append(HeadingEntry(
            level=level,
            text=text,
            char_offset=char_offset,
        ))

    return headings, (2 if headings else 3)


# ---------------------------------------------------------------------------
# Table extraction (unchanged from original)
# ---------------------------------------------------------------------------

def _extract_tables(page: fitz.Page, page_num: int) -> Tuple[str, List[TableEntry]]:
    """Extract tables from a PyMuPDF page as markdown and return text without
    the tables + table entries."""
    tabs = page.find_tables()
    tables = []

    if not tabs.tables:
        return page.get_text("text"), []

    for tab in tabs.tables:
        df = tab.to_pandas()
        md_table = df.to_markdown(index=False)
        if md_table:
            tables.append(TableEntry(page=page_num, markdown=md_table))

    return page.get_text("text"), tables


# ---------------------------------------------------------------------------
# Main parser (Tier 1 / 2 / 3)  — Tier-2 block is the only changed section
# ---------------------------------------------------------------------------

def parse_pdf(file_path: str | Path, doc_id: str, original_filename: str) -> ParsedDocument:
    """Parses a PDF file (Tier 1/2/3). Handles OCR and passwords."""
    path = Path(file_path)
    content_hash = compute_file_hash(path)

    warnings = []
    errors = []
    headings = []
    tables = []
    full_text_parts = []

    structure_tier = 3
    title_source = "none"
    title = None
    ocr_used = False

    char_offset = 0
    doc = None

    try:
        doc = fitz.open(str(path))

        if doc.needs_pass:
            errors.append("PDF is password protected.")
            return _create_error_doc(doc_id, path, original_filename, content_hash, errors)

        toc = doc.get_toc()
        if toc:
            structure_tier = 1
            title_source = "native"

        page_offsets = {}

        for page_num, page in enumerate(doc, 1):
            page_offsets[page_num] = char_offset
            text, page_tables = _extract_tables(page, page_num)

            if not page_tables:
                page_tables = extract_text_tables(text, page_num)

            tables.extend(page_tables)

            if len(text.strip()) < 50 and config.OCR_ENABLED:
                ocr_used = True
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img, lang=config.OCR_LANGUAGE)
                text = ocr_text

            full_text_parts.append(text)
            char_offset += len(text) + 1  # +1 for newline

        full_text = "\n".join(full_text_parts)

        # ---------------------------------------------------------------
        # Tier 1: Native TOC/bookmarks
        # ---------------------------------------------------------------
        if toc:
            for item in toc:
                level, heading_text, page_num = item[0], item[1].strip(), item[2]
                page_start_offset = page_offsets.get(page_num, 0)

                if 1 <= page_num <= len(full_text_parts):
                    page_text = full_text_parts[page_num - 1]
                    local_offset = page_text.find(heading_text)
                    final_offset = (
                        page_start_offset + local_offset
                        if local_offset != -1
                        else page_start_offset
                    )
                else:
                    final_offset = page_start_offset

                headings.append(HeadingEntry(
                    level=level,
                    text=heading_text,
                    char_offset=final_offset,
                ))

        # ---------------------------------------------------------------
        # Tier 2: Font/layout-based heading inference
        # (only when no native TOC and OCR was not needed)
        # ---------------------------------------------------------------
        elif not ocr_used:
            # Collect character-weighted font statistics in a single pass.
            # This is O(total_spans) and uses the already-open doc object —
            # no additional I/O.
            font_stats = _collect_font_stats(doc)

            headings, structure_tier = _extract_tier2_headings(
                doc, full_text_parts, font_stats
            )

            if headings:
                title_source = "heuristic"

        # ---------------------------------------------------------------
        # Tier 3: OCR path — no structural inference possible
        # ---------------------------------------------------------------
        # (already handled: ocr_used=True means we skip Tier 2 above)

        if not headings and structure_tier != 1:
            structure_tier = 3
            title_source = "none"

        if title_source in ("heuristic", "native") and headings and title is None:
            title = headings[0].text.strip()

        if len(full_text) < config.LOW_TEXT_THRESHOLD:
            warnings.append(f"Low extracted text length: {len(full_text)} characters.")

        page_count = len(doc)

        return ParsedDocument(
            doc_id=doc_id,
            source_path=str(path),
            original_filename=original_filename,
            file_type="pdf",
            content_hash=content_hash,
            full_text=full_text,
            structure_tier=structure_tier,
            headings=headings,
            title=title,
            title_source=title_source,
            page_count=page_count,
            language=None,
            ocr_used=ocr_used,
            parser_used="pdf_parser",
            tables=tables,
            extraction_warnings=warnings,
            extraction_errors=errors,
            metadata={},
        )

    except Exception as e:
        import traceback
        errors.append(f"Failed to parse PDF: {str(e)}")
        logger.error(
            f"PDF parsing failed for {original_filename}: {e}\n{traceback.format_exc()}"
        )
        return _create_error_doc(doc_id, path, original_filename, content_hash, errors)
    finally:
        if doc is not None:
            doc.close()


def _create_error_doc(
    doc_id: str,
    path: Path,
    original_filename: str,
    content_hash: str,
    errors: List[str],
) -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(path),
        original_filename=original_filename,
        file_type="pdf",
        content_hash=content_hash,
        full_text="",
        structure_tier=3,
        headings=[],
        title=None,
        title_source="none",
        page_count=None,
        language=None,
        ocr_used=False,
        parser_used="pdf_parser",
        tables=[],
        extraction_warnings=[],
        extraction_errors=errors,
        metadata={},
    )
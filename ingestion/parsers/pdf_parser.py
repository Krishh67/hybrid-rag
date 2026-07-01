import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Set

from ..schema import ParsedDocument, HeadingEntry, TableEntry
from ..config import config
from ..utils import compute_file_hash, logger, extract_text_tables

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


# ---------------------------------------------------------------------------
# Scoring constants — centralised so they are easy to tune
# ---------------------------------------------------------------------------

# Minimum net score for a span to be accepted as a heading.
# Raised from 3.0 to 4.0 to require stronger multi-signal evidence.
_SCORE_THRESHOLD = 4.0

# Positive weights
_W_SIZE_SLIGHT   = 1.5   # font_ratio ≥ 1.10
_W_SIZE_LARGE    = 1.0   # font_ratio ≥ 1.30  (stacks with above)
_W_BOLD          = 1.5
_W_NUMBERED      = 2.0
_W_SHORT_LINE    = 1.0   # line ≤ 80 chars
_W_CENTERED      = 0.5
_W_ALLCAPS       = 0.5   # low weight — footers use all-caps too

# Negative weights (applied as penalties, i.e. score -= value)
_P_REPEATED      = 3.5   # text seen on ≥ REPEAT_PAGE_FRACTION of all pages
_P_TABLE_OVERLAP = 5.0   # span bbox overlaps a detected table region (hard suppress)
_P_METADATA_ZONE = 2.5   # span falls in a detected metadata cluster zone
_P_DENSE_BLOCK   = 1.5   # span is in a block with many same-size lines (body paragraph)

# A candidate text is considered "repeated decoration" if it appears on at
# least this fraction of total pages (e.g. 0.3 → present on 30 %+ of pages).
_REPEAT_PAGE_FRACTION = 0.30

# Metadata zone: top/bottom margin fraction of page height to examine.
_METADATA_MARGIN_FRAC = 0.12   # top 12 % and bottom 12 % of each page

# Metadata zone: max lines in a cluster for it to be considered a dense
# short-line metadata block (author lists, affiliations, journal info).
_METADATA_MAX_CLUSTER_LINES = 8

# Minimum number of alphanumeric characters required in a heading candidate.
_MIN_ALNUM_CHARS = 2


# ---------------------------------------------------------------------------
# False-positive guard regex
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
        (chapter|appendix|section|part|annex)\s+[\dA-Z]   # named sections
        | \d+(\.\d+)*\.?\s+\S                             # numeric like 1. or 2.3.4
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Helper: font statistics
# ---------------------------------------------------------------------------

def _collect_font_stats(doc: fitz.Document) -> Dict[float, float]:
    """
    Single-pass over all spans: build a character-weighted font-size
    frequency map.  Returns {font_size_rounded: total_char_count}.

    Complexity: O(total_spans)
    """
    font_stats: Dict[float, float] = defaultdict(float)
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    if text.strip():
                        size = round(span["size"] * 2) / 2   # collapse near-identical sizes
                        font_stats[size] += len(text)
    return font_stats


def _dominant_body_size(font_stats: Dict[float, float]) -> float:
    """Return the font size responsible for the most character data (body text)."""
    if not font_stats:
        return 12.0
    return max(font_stats, key=lambda s: font_stats[s])


# ---------------------------------------------------------------------------
# Helper: table bounding boxes per page
# ---------------------------------------------------------------------------

def _collect_table_bboxes(doc: fitz.Document) -> Dict[int, List[fitz.Rect]]:
    """
    For every page, collect the bounding boxes of all detected tables.
    Returns {page_num (1-indexed): [fitz.Rect, ...]}.

    We reuse PyMuPDF's find_tables() which is already called during text
    extraction — here we call it again only during the Tier-2 analysis pass,
    keeping this strictly in the Tier-2 code path.

    Complexity: O(pages * table_detection_cost) — dominated by find_tables(),
    which is O(spans_per_page).  Net result: O(total_spans).
    """
    table_bboxes: Dict[int, List[fitz.Rect]] = defaultdict(list)
    for page_num, page in enumerate(doc, 1):
        try:
            tabs = page.find_tables()
            for tab in tabs.tables:
                table_bboxes[page_num].append(fitz.Rect(tab.bbox))
        except Exception:
            pass   # find_tables can fail on malformed pages; skip silently
    return table_bboxes


def _span_in_table(span_bbox: tuple, page_num: int,
                   table_bboxes: Dict[int, List[fitz.Rect]]) -> bool:
    """
    Return True if the span's bounding box overlaps any detected table
    region on that page.  Uses a simple rect-intersection test.

    Complexity: O(tables_per_page) — typically 0–5.
    """
    bboxes = table_bboxes.get(page_num)
    if not bboxes:
        return False
    sr = fitz.Rect(span_bbox)
    for tr in bboxes:
        if not (sr & tr).is_empty:
            return True
    return False


# ---------------------------------------------------------------------------
# Helper: metadata zone detection
# ---------------------------------------------------------------------------

def _detect_metadata_zones(doc: fitz.Document) -> Dict[int, List[Tuple[float, float]]]:
    """
    Identify vertical bands on early pages (1–3) that contain dense clusters
    of short, heterogeneous-font lines — characteristic of author blocks,
    affiliations, journal headers, and similar metadata.

    Returns {page_num: [(y0, y1), ...]} — a list of vertical intervals on
    that page that should be treated with suspicion.

    Algorithm:
      1. For each of the first min(3, page_count) pages:
         a. Examine the top METADATA_MARGIN_FRAC and bottom METADATA_MARGIN_FRAC
            of the page height.
         b. Within each margin band, collect all text lines.
         c. If a band contains 2–METADATA_MAX_CLUSTER_LINES lines and more than
            one distinct font size → mark the entire band as a metadata zone.
      2. Also scan all pages for the header/footer bands (very top / very bottom
         5 % of page), which are likely running headers/footers everywhere.

    Complexity: O(spans on pages 1–3 + spans in header/footer bands on all pages)
    — negligible relative to O(total_spans).
    """
    metadata_zones: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    page_count = len(doc)

    # --- Early-page margin analysis (pages 1–3 only) ---
    early_pages = min(3, page_count)
    for page_num in range(1, early_pages + 1):
        page = doc[page_num - 1]
        page_h = page.rect.height
        margin = page_h * _METADATA_MARGIN_FRAC

        for band_y0, band_y1 in [(0, margin), (page_h - margin, page_h)]:
            lines_in_band = []
            sizes_in_band = set()

            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    ly0 = line["bbox"][1]
                    ly1 = line["bbox"][3]
                    # Line must be substantially inside the band
                    if ly0 >= band_y0 and ly1 <= band_y1:
                        line_text = "".join(s["text"] for s in line["spans"]).strip()
                        if line_text:
                            lines_in_band.append(line)
                            for s in line["spans"]:
                                sizes_in_band.add(round(s["size"] * 2) / 2)

            # Dense short-line cluster with font variation → metadata zone
            if (2 <= len(lines_in_band) <= _METADATA_MAX_CLUSTER_LINES
                    and len(sizes_in_band) > 1):
                metadata_zones[page_num].append((band_y0, band_y1))

    # --- Running header/footer band on all pages (top/bottom 5 %) ---
    header_footer_frac = 0.05
    for page_num in range(1, page_count + 1):
        page = doc[page_num - 1]
        page_h = page.rect.height
        hf_margin = page_h * header_footer_frac
        # Add narrow header/footer bands (they may already be covered above
        # for early pages, but duplicates in the list are harmless — the
        # overlap check is idempotent).
        metadata_zones[page_num].append((0, hf_margin))
        metadata_zones[page_num].append((page_h - hf_margin, page_h))

    return metadata_zones


def _span_in_metadata_zone(span_bbox: tuple, page_num: int,
                            metadata_zones: Dict[int, List[Tuple[float, float]]]) -> bool:
    """
    Return True if the span's vertical centre falls inside any metadata zone
    on that page.

    Complexity: O(zones_per_page) — typically 2–4.
    """
    zones = metadata_zones.get(page_num)
    if not zones:
        return False
    span_cy = (span_bbox[1] + span_bbox[3]) / 2
    for y0, y1 in zones:
        if y0 <= span_cy <= y1:
            return True
    return False


# ---------------------------------------------------------------------------
# Helper: repetition counting
# ---------------------------------------------------------------------------

def _count_candidate_page_occurrences(
    doc: fitz.Document,
    body_font_size: float,
) -> Dict[str, int]:
    """
    Single pass: for every span that passes the basic pre-filter, record
    which pages its normalised text appears on.  Returns
    {normalised_text: page_count}.

    We normalise by lowercasing and collapsing whitespace so minor rendering
    differences across pages don't prevent matching.

    Complexity: O(total_spans)
    """
    # text → set of page numbers it appears on
    text_pages: Dict[str, Set[int]] = defaultdict(set)

    for page_num, page in enumerate(doc, 1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    raw = span["text"].strip()
                    if not raw or len(raw) < 2:
                        continue
                    if not any(c.isalnum() for c in raw):
                        continue
                    # Only track spans that could plausibly be headings
                    span_size = round(span["size"] * 2) / 2
                    font_ratio = span_size / body_font_size if body_font_size > 0 else 1.0
                    is_bold = bool(span.get("flags", 0) & 2**4)
                    is_numbered = bool(_NUMBERED_HEADING_RE.match(raw))
                    if font_ratio < 1.05 and not is_bold and not is_numbered:
                        continue
                    key = " ".join(raw.lower().split())
                    text_pages[key].add(page_num)

    return {k: len(v) for k, v in text_pages.items()}


# ---------------------------------------------------------------------------
# Helper: level inference
# ---------------------------------------------------------------------------

def _infer_heading_levels(sizes: List[float]) -> Dict[float, int]:
    """
    Cluster distinct accepted heading font sizes into levels (1 = largest).
    Two sizes in the same cluster when gap < 1.5 pt.

    Complexity: O(k log k), k = distinct sizes (typically < 10).
    """
    if not sizes:
        return {}
    sorted_sizes = sorted(set(sizes), reverse=True)
    level_map: Dict[float, int] = {}
    current_level = 1
    prev = sorted_sizes[0]
    level_map[prev] = current_level
    for size in sorted_sizes[1:]:
        if (prev - size) >= 1.5:
            current_level += 1
        level_map[size] = current_level
        prev = size
    return level_map


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

def _score_span(
    span: dict,
    line: dict,
    page_num: int,
    page_width: float,
    body_font_size: float,
    page_line_count: int,
    # document-level context
    repetition_counts: Dict[str, int],
    total_pages: int,
    table_bboxes: Dict[int, List[fitz.Rect]],
    metadata_zones: Dict[int, List[Tuple[float, float]]],
) -> float:
    """
    Compute a signed confidence score for one span.
    Positive signals indicate heading-like properties.
    Negative signals (penalties) indicate decoration / noise.

    Returns the net score; caller applies the threshold.
    """
    text = span["text"].strip()

    # --- Hard pre-filters (zero cost, applied first) ---
    if not text or len(text) < 2:
        return 0.0
    alnum_count = sum(1 for c in text if c.isalnum())
    if alnum_count < _MIN_ALNUM_CHARS:
        return 0.0
    if _FALSE_POSITIVE_RE.match(text):
        return 0.0

    span_size = round(span["size"] * 2) / 2
    font_ratio = span_size / body_font_size if body_font_size > 0 else 1.0
    is_bold = bool(span.get("flags", 0) & 2**4)
    is_numbered = bool(_NUMBERED_HEADING_RE.match(text))

    # Fast rejection: at least one weak positive signal must be present
    if font_ratio < 1.05 and not is_bold and not is_numbered:
        return 0.0

    score = 0.0

    # -----------------------------------------------------------------------
    # Positive evidence
    # -----------------------------------------------------------------------
    if font_ratio >= 1.10:
        score += _W_SIZE_SLIGHT
    if font_ratio >= 1.30:
        score += _W_SIZE_LARGE      # stacks with above

    if is_bold:
        score += _W_BOLD

    if is_numbered:
        score += _W_NUMBERED

    line_text = "".join(s["text"] for s in line["spans"]).strip()

    if len(line_text) <= 80:
        score += _W_SHORT_LINE

    if len(line_text) <= 120 and page_width > 0:
        x0, x1 = line["bbox"][0], line["bbox"][2]
        line_center = (x0 + x1) / 2
        if abs(line_center - page_width / 2) < page_width * 0.15:
            score += _W_CENTERED

    if text.isupper() and len(text) > 3:
        score += _W_ALLCAPS

    # -----------------------------------------------------------------------
    # Negative evidence
    # -----------------------------------------------------------------------

    # 1. Repetition penalty — text present on many pages → decoration
    if total_pages > 0:
        norm_text = " ".join(text.lower().split())
        count = repetition_counts.get(norm_text, 0)
        if count / total_pages >= _REPEAT_PAGE_FRACTION:
            score -= _P_REPEATED

    # 2. Table region — geometric overlap with a detected table bbox
    if _span_in_table(span["bbox"], page_num, table_bboxes):
        score -= _P_TABLE_OVERLAP

    # 3. Metadata zone — span sits in a detected metadata cluster band
    if _span_in_metadata_zone(span["bbox"], page_num, metadata_zones):
        score -= _P_METADATA_ZONE

    # 4. Dense same-size block — span is in a block where many lines share
    #    its font size, which characterises body paragraphs and table cells
    #    that happen to be bold (e.g. column headers embedded in dense text).
    #    We use page_line_count as a proxy: if the page has very few lines
    #    overall this heuristic is not informative, so we only apply it on
    #    pages with enough lines to distinguish blocks.
    if page_line_count > 20:
        block_lines = sum(1 for bl in span.get("_block_lines", []) if bl)
        # _block_lines is injected by the caller (see _extract_tier2_headings)
        if block_lines > 4:
            score -= _P_DENSE_BLOCK

    return score


# ---------------------------------------------------------------------------
# Main Tier-2 entry point
# ---------------------------------------------------------------------------

def _extract_tier2_headings(
    doc: fitz.Document,
    full_text_parts: List[str],
    font_stats: Dict[float, float],
) -> Tuple[List[HeadingEntry], int]:
    """
    Perform Tier-2 heading detection.

    Pipeline:
      1. Compute body font size from character-weighted stats.
      2. Collect document-level context in O(total_spans):
           - candidate repetition counts
           - table bounding boxes per page
           - metadata zone intervals per page
      3. Score every span with both positive and negative evidence.
      4. Cluster accepted font sizes → heading levels.
      5. Build HeadingEntry list with reliable char offsets.

    Complexity: O(total_spans)  — no repeated full-document scans.
    """
    body_font_size = _dominant_body_size(font_stats)
    total_pages = len(doc)

    # --- Build document-level context structures (each O(total_spans)) ---
    repetition_counts = _count_candidate_page_occurrences(doc, body_font_size)
    table_bboxes      = _collect_table_bboxes(doc)
    metadata_zones    = _detect_metadata_zones(doc)

    # --- Scoring pass ---
    # raw_candidates: (page_num, span, line, net_score, page_width)
    raw_candidates = []

    for page_num, page in enumerate(doc, 1):
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]

        # Count total lines on this page for the dense-block heuristic
        page_line_count = sum(
            len(b["lines"]) for b in blocks if b["type"] == 0
        )

        for block in blocks:
            if block["type"] != 0:
                continue

            # Count lines in this block that share each font size —
            # used by the dense-block penalty inside _score_span.
            block_size_line_counts: Dict[float, int] = defaultdict(int)
            for line in block["lines"]:
                line_sizes = {round(s["size"] * 2) / 2 for s in line["spans"] if s["text"].strip()}
                for sz in line_sizes:
                    block_size_line_counts[sz] += 1

            for line in block["lines"]:
                for span in line["spans"]:
                    span_size = round(span["size"] * 2) / 2
                    # Inject same-size line count into span for the scorer
                    span["_block_lines"] = [True] * block_size_line_counts.get(span_size, 0)

                    net_score = _score_span(
                        span=span,
                        line=line,
                        page_num=page_num,
                        page_width=page_width,
                        body_font_size=body_font_size,
                        page_line_count=page_line_count,
                        repetition_counts=repetition_counts,
                        total_pages=total_pages,
                        table_bboxes=table_bboxes,
                        metadata_zones=metadata_zones,
                    )
                    if net_score >= _SCORE_THRESHOLD:
                        raw_candidates.append((page_num, span, line, net_score, page_width))

    if not raw_candidates:
        return [], 3

    # --- Infer heading levels from accepted candidate font sizes ---
    candidate_sizes = [round(c[1]["size"] * 2) / 2 for c in raw_candidates]
    level_map = _infer_heading_levels(candidate_sizes)

    # --- Build page start offsets for reliable char_offset calculation ---
    page_start_offsets: List[int] = []
    cumulative = 0
    for part in full_text_parts:
        page_start_offsets.append(cumulative)
        cumulative += len(part) + 1   # +1 for the "\n" join separator

    # Per-page search cursor: advances after each match so repeated text on
    # the same page resolves to the correct (next) occurrence.
    search_from: Dict[int, int] = defaultdict(int)

    headings: List[HeadingEntry] = []
    seen: Set[Tuple[int, str]] = set()   # (page_num, text) dedup

    for page_num, span, line, net_score, page_width in raw_candidates:
        text = span["text"].strip()
        key = (page_num, text)
        if key in seen:
            continue
        seen.add(key)

        span_size = round(span["size"] * 2) / 2
        level = level_map.get(span_size, 1)

        page_idx = page_num - 1
        if page_idx < len(full_text_parts):
            page_text = full_text_parts[page_idx]
            local_start = search_from[page_num]
            pos = page_text.find(text, local_start)
            if pos == -1:
                pos = page_text.find(text)   # fallback: search from start of page
            if pos != -1:
                search_from[page_num] = pos + len(text)
                char_offset = page_start_offsets[page_idx] + pos
            else:
                char_offset = page_start_offsets[page_idx]
        else:
            char_offset = 0

        headings.append(HeadingEntry(level=level, text=text, char_offset=char_offset))

    return headings, (2 if headings else 3)


# ---------------------------------------------------------------------------
# Table extraction (unchanged)
# ---------------------------------------------------------------------------

def _page_likely_has_table(page: fitz.Page) -> bool:
    """
    Minimal safe pre-filter: returns False only when the page has zero vector
    drawings. A page with no drawings cannot contain a table border.

    This is safe for any PDF type — no tuning to specific document styles.
    Saves ~0.2-0.3s per pure-text page that has no graphics at all.
    """
    return bool(page.get_drawings())



def _is_valid_table(md_table: str) -> bool:
    """
    Check if a markdown table has enough real content to be considered a true table.
    Filters out layout artifacts, row numbers, and empty placeholders (e.g., 'Col1').
    """
    import re
    # Extract words that are not markdown syntax
    words = [w.strip() for w in re.split(r'\||\s+', md_table) if len(w.strip()) > 0]
    # Filter out table syntax
    real_words = [w for w in words if not set(w) <= set('-: ')]
    # Filter out auto-generated pandas column names like Col0, Col1
    content_words = [w for w in real_words if not re.match(r'^Col\d+$', w, re.IGNORECASE)]
    
    # Require at least 4 real content words
    return len(content_words) >= 4


def _extract_tables(page: fitz.Page, page_num: int) -> Tuple[str, List[TableEntry]]:
    """Extract tables from a PyMuPDF page as markdown."""
    # Skip expensive find_tables if page has no stroke drawings (no table borders)
    if not _page_likely_has_table(page):
        return page.get_text("text"), []

    tabs = page.find_tables()
    tables = []
    if not tabs.tables:
        return page.get_text("text"), []
    for tab in tabs.tables:
        df = tab.to_pandas()
        md_table = df.to_markdown(index=False)
        if md_table and _is_valid_table(md_table):
            tables.append(TableEntry(page=page_num, markdown=md_table))
    return page.get_text("text"), tables





# ---------------------------------------------------------------------------
# Main parser — Tier 1 / 2 / 3
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
            char_offset += len(text) + 1

        full_text = "\n".join(full_text_parts)

        # -------------------------------------------------------------------
        # Tier 1: Native TOC / bookmarks
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # Tier 2: Font/layout-based heading inference
        # -------------------------------------------------------------------
        elif not ocr_used:
            font_stats = _collect_font_stats(doc)
            headings, structure_tier = _extract_tier2_headings(
                doc, full_text_parts, font_stats
            )
            if headings:
                title_source = "heuristic"

        # -------------------------------------------------------------------
        # Tier 3: OCR fallback — no structural inference possible
        # -------------------------------------------------------------------

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
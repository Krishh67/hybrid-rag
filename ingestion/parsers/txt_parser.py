from pathlib import Path
from charset_normalizer import from_path
from typing import Optional

from ..schema import ParsedDocument
from ..config import config
from ..utils import compute_file_hash, logger, extract_text_tables

def parse_txt(file_path: str | Path, doc_id: str, original_filename: str) -> ParsedDocument:
    """
    Parses a plain text file.
    Always Tier 3 (no structure). Detects encoding robustly.
    """
    path = Path(file_path)
    content_hash = compute_file_hash(path)
    
    warnings = []
    errors = []
    
    try:
        # Detect encoding using charset_normalizer
        matches = from_path(path)
        best_match = matches.best()
        
        if best_match is not None:
            text = str(best_match)
            # Check confidence (optional depending on charset-normalizer version, but we can look at chaos or similar)
            # charset-normalizer usually is quite accurate. We can log if it's not utf-8 or ascii.
            if best_match.encoding not in ('utf-8', 'ascii'):
                if len(text) < 200 and best_match.encoding not in ('cp1252', 'windows-1252', 'latin_1', 'iso8859_1'):
                    logger.info(f"Short text ({len(text)} chars) with rare encoding {best_match.encoding}. Forcing cp1252.")
                    with open(path, "r", encoding="cp1252", errors="replace") as f:
                        text = f.read()
                else:
                    logger.info(f"Detected encoding {best_match.encoding} for {original_filename}")
        else:
            # Fallback
            warnings.append("Low encoding confidence. Falling back to utf-8 with replacement.")
            logger.warning(f"Low encoding confidence for {original_filename}. Using utf-8 replace.")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
                
    except Exception as e:
        errors.append(f"Failed to read txt file: {str(e)}")
        text = ""

    if len(text) < config.LOW_TEXT_THRESHOLD and not errors:
        warnings.append(f"Low extracted text length: {len(text)} characters.")
        
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(path),
        original_filename=original_filename,
        file_type="txt",
        content_hash=content_hash,
        full_text=text,
        structure_tier=3,
        headings=[],
        title=None,
        title_source="none",
        page_count=None,
        language=None,
        ocr_used=False,
        parser_used="txt_parser",
        tables=extract_text_tables(text) if text else [],
        extraction_warnings=warnings,
        extraction_errors=errors,
        metadata={}
    )

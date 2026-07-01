from pathlib import Path
from typing import Any
import docx

from ..schema import ParsedDocument, HeadingEntry, TableEntry
from ..config import config
from ..utils import compute_file_hash, logger

def _table_to_markdown(table: docx.table.Table) -> str:
    """Converts a docx Table to markdown format."""
    md_lines = []
    for i, row in enumerate(table.rows):
        row_data = [cell.text.replace('\n', ' ').strip() for cell in row.cells]
        md_lines.append("| " + " | ".join(row_data) + " |")
        
        if i == 0:
            # Header separator
            separator = ["---"] * len(row.cells)
            md_lines.append("| " + " | ".join(separator) + " |")
            
    return "\n".join(md_lines)

def parse_docx(file_path: str | Path, doc_id: str, original_filename: str) -> ParsedDocument:
    """
    Parses a DOCX file.
    Tier 1: extracts heading styles directly.
    """
    path = Path(file_path)
    content_hash = compute_file_hash(path)
    
    warnings = []
    errors = []
    
    full_text_parts = []
    headings = []
    tables = []
    char_offset = 0
    
    try:
        doc = docx.Document(str(path))
        
        # We need to preserve order of paragraphs and tables.
        # python-docx doesn't provide a unified iterator out of the box easily, 
        # but we can iterate over doc.element.body to get them in order,
        # or we just extract paragraphs then tables and lose interleave order.
        # Let's iterate block-level elements in body.
        
        for block in doc.element.body:
            if block.tag.endswith('p'):
                # Paragraph
                p = docx.text.paragraph.Paragraph(block, doc)
                text = p.text.strip()
                if not text:
                    continue
                    
                style_name = p.style.name if p.style else ""
                
                # Check for headings or Title
                if style_name == "Title" or style_name.startswith("Heading"):
                    level = 1
                    if style_name.startswith("Heading"):
                        try:
                            level = int(style_name.split()[-1])
                        except ValueError:
                            level = 1 # Fallback if style is just "Heading"
                        
                    headings.append(HeadingEntry(
                        level=level,
                        text=text,
                        char_offset=char_offset
                    ))
                    
                full_text_parts.append(text)
                char_offset += len(text) + 1 # +1 for newline
                
            elif block.tag.endswith('tbl'):
                # Table
                t = docx.table.Table(block, doc)
                md_table = _table_to_markdown(t)
                
                start_off = char_offset
                full_text_parts.append(md_table)
                char_offset += len(md_table) + 1
                
                tables.append(TableEntry(page=None, markdown=md_table, char_start=start_off, char_end=char_offset - 1))
                
    except Exception as e:
        errors.append(f"Failed to parse DOCX: {str(e)}")
        logger.error(f"DOCX parsing failed for {original_filename}: {e}")
        
    full_text = "\n".join(full_text_parts)
    
    if len(full_text) < config.LOW_TEXT_THRESHOLD and not errors:
        warnings.append(f"Low extracted text length: {len(full_text)} characters.")
        
    title = None
    title_source = "none"
    if headings:
        title = headings[0].text.strip()
        title_source = "native"
        
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(path),
        original_filename=original_filename,
        file_type="docx",
        content_hash=content_hash,
        full_text=full_text,
        structure_tier=1, # DOCX is always tier 1 due to styles
        headings=headings,
        title=title,
        title_source=title_source,
        page_count=None,
        language=None,
        ocr_used=False,
        parser_used="docx_parser",
        tables=tables,
        extraction_warnings=warnings,
        extraction_errors=errors,
        metadata={}
    )

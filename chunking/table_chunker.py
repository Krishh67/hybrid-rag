import uuid
import re
from typing import List
from ingestion.schema import ParsedDocument, TableEntry
from .schema import Chunk, ChunkMetadata
from .utils import count_tokens, format_embed_text

def _find_table_start(doc: ParsedDocument, table: TableEntry, cursor: int) -> int:
    if getattr(table, "char_start", None) is not None:
        return table.char_start
        
    # Temporary cursor-based matching strategy for parsers that do not provide spans (e.g. PDF).
    # We extract the first non-empty word of the table to find its position.
    words = [w.strip() for w in re.split(r'\||\s+', table.markdown) if len(w.strip()) > 1 and not set(w.strip()) <= set('-: ')]
    if not words:
        return cursor
        
    first_word = words[0]
    # Look ahead from cursor
    match = re.search(re.escape(first_word), doc.full_text[cursor:])
    if match:
        return cursor + match.start()
        
    # Fallback search from start if out of order
    match = re.search(re.escape(first_word), doc.full_text)
    if match:
        return match.start()
        
    # If all else fails, stay at cursor
    return cursor

def _get_heading_path(doc: ParsedDocument, abs_start: int) -> List[str]:
    """Calculate the active heading path for a given absolute offset."""
    active_headings = []
    for h in doc.headings:
        if h.char_offset <= abs_start:
            # Pop headings of equal or greater depth
            active_headings = [ah for ah in active_headings if ah.level < h.level]
            active_headings.append(h)
        else:
            break
    return [h.text for h in active_headings]


def chunk_table(doc: ParsedDocument, table: TableEntry, search_cursor: int = 0) -> tuple[List[Chunk], int]:
    """
    Chunks a table. If small, returns a single chunk.
    If large, chunks row-by-row, ensuring header is prepended to each chunk.
    Returns the list of chunks and the updated search cursor.
    """
    chunks = []
    lines = table.markdown.strip().split('\n')
    
    if not lines:
        return [], search_cursor
        
    token_count = count_tokens(table.markdown)
    table_start = _find_table_start(doc, table, search_cursor)
    heading_path = _get_heading_path(doc, table_start)
    
    # Simple chunking if small enough
    if token_count <= 400 or len(lines) <= 3:
        chunk_id = str(uuid.uuid4())
        
        embed_text = format_embed_text(doc.title or "", heading_path, table.markdown)
        
        abs_end = table_start + len(table.markdown)
        new_cursor = abs_end
        
        metadata = ChunkMetadata(
            doc_id=doc.doc_id,
            original_filename=doc.original_filename,
            title=doc.title,
            heading_path=heading_path,
            chunk_type="table"
        )
        
        chunk = Chunk(
            chunk_id=chunk_id,
            parent_window_id="none",
            segment_id="none",
            chunk_text=table.markdown,
            embed_text=embed_text,
            char_start=table_start,
            char_end=abs_end,
            token_count=token_count,
            metadata=metadata
        )
        return [chunk], new_cursor
        
    # Large table chunking
    # Assuming standard markdown table: 
    # Header:   | a | b |
    # Sep:      |---|---|
    # Rows:     | 1 | 2 |
    
    header_lines = []
    row_lines = []
    
    if len(lines) >= 2 and '---' in lines[1]:
        header_lines = lines[:2]
        row_lines = lines[2:]
    else:
        # No clear header separator, just treat first line as header
        header_lines = [lines[0]]
        row_lines = lines[1:]
        
    current_chunk_lines = list(header_lines)
    current_chunk_tokens = count_tokens("\n".join(current_chunk_lines))
    
    new_cursor = table_start + len(table.markdown)
    
    for row in row_lines:
        row_tokens = count_tokens(row)
        
        if current_chunk_tokens + row_tokens > 400 and len(current_chunk_lines) > len(header_lines):
            # Finalize chunk
            chunk_text = "\n".join(current_chunk_lines)
            chunk_id = str(uuid.uuid4())
            metadata = ChunkMetadata(
                doc_id=doc.doc_id,
                original_filename=doc.original_filename,
                title=doc.title,
                heading_path=heading_path,
                chunk_type="table"
            )
            chunk = Chunk(
                chunk_id=chunk_id,
                parent_window_id="none",
                segment_id="none",
                chunk_text=chunk_text,
                embed_text=format_embed_text(doc.title or "", heading_path, chunk_text),
                char_start=table_start,
                char_end=table_start + len(table.markdown),
                token_count=current_chunk_tokens,
                metadata=metadata
            )
            chunks.append(chunk)
            
            # Reset
            current_chunk_lines = list(header_lines)
            current_chunk_lines.append(row)
            current_chunk_tokens = count_tokens("\n".join(current_chunk_lines))
        else:
            current_chunk_lines.append(row)
            current_chunk_tokens += row_tokens
            
    # Add remainder
    if len(current_chunk_lines) > len(header_lines):
        chunk_text = "\n".join(current_chunk_lines)
        chunk_id = str(uuid.uuid4())
        metadata = ChunkMetadata(
            doc_id=doc.doc_id,
            original_filename=doc.original_filename,
            title=doc.title,
            heading_path=heading_path,
            chunk_type="table"
        )
        chunk = Chunk(
            chunk_id=chunk_id,
            parent_window_id="none",
            segment_id="none",
            chunk_text=chunk_text,
            embed_text=format_embed_text(doc.title or "", heading_path, chunk_text),
            char_start=table_start,
            char_end=table_start + len(table.markdown),
            token_count=current_chunk_tokens,
            metadata=metadata
        )
        chunks.append(chunk)
        
    return chunks, new_cursor

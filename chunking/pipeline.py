from typing import List
from ingestion.schema import ParsedDocument
from .schema import Chunk
from .segmenter import segment_document
from .parent_window_builder import build_parent_windows
from .chunk_builder import build_chunks_from_window
from .table_chunker import chunk_table
from .code_chunker import extract_and_chunk_code

def chunk_document(doc: ParsedDocument) -> tuple[List[Chunk], List[dict]]:
    """
    Main entry point for Stage 2: Chunking.
    Takes a ParsedDocument and returns a list of Chunks ready for embedding,
    and a list of serialized parent window dicts for persistence.
    """
    all_chunks = []
    all_parent_windows = []
    
    # 1. Process Tables
    table_search_cursor = 0
    for table in doc.tables:
        table_chunks, table_search_cursor = chunk_table(doc, table, table_search_cursor)
        all_chunks.extend(table_chunks)
        
    # 2. Process Code Blocks and remove them from main text to avoid double chunking
    # For simplicity, code_chunker assumes no headings for code blocks unless we map them,
    # but the simplest way is to pass an empty heading_path or try to guess.
    # To guess, we could do it after segmentation, but segmenter needs clean text.
    clean_text, code_chunks = extract_and_chunk_code(doc, doc.full_text, heading_path=[])
    all_chunks.extend(code_chunks)
    
    # 3. Segment the remaining text
    segments = segment_document(doc, clean_text)
    
    # 4. Build Parent Windows and Chunks
    for segment in segments:
        parent_windows = build_parent_windows(doc, segment)
        segment.parent_windows = parent_windows
        
        for window in parent_windows:
            all_parent_windows.append({
                "parent_window_id": window.parent_id,
                "segment_id": window.segment_id,
                "doc_id": doc.doc_id,
                "char_start": window.char_start,
                "char_end": window.char_end,
                "token_count": window.token_count
            })
            
            text_chunks = build_chunks_from_window(doc, window, segment.heading_path)
            window.chunks = text_chunks
            all_chunks.extend(text_chunks)
            
    return all_chunks, all_parent_windows

import re
import uuid
from typing import List
from ingestion.schema import ParsedDocument
from .schema import Chunk, ChunkMetadata
from .utils import count_tokens, format_embed_text

# Regex to find markdown code blocks
# Use [^`] to avoid catastrophic backtracking on large documents with DOTALL
CODE_BLOCK_PATTERN = re.compile(r'```([a-zA-Z0-9]*)\n(.*?)(?=```|\Z)', re.DOTALL)

def extract_and_chunk_code(doc: ParsedDocument, parent_text: str, heading_path: List[str]) -> tuple[str, List[Chunk]]:
    """
    Extracts code blocks from the given text, chunks them based on function/class boundaries,
    and returns the cleaned text (with code blocks removed) and the extracted chunks.
    """
    chunks = []
    clean_text = parent_text
    
    # Fast early exit: no code fences present — avoids full regex scan on large documents
    if '```' not in parent_text:
        return clean_text, chunks
    
    for match in CODE_BLOCK_PATTERN.finditer(parent_text):
        lang = match.group(1)
        code = match.group(2)
        match_start = match.start()
        code_start = match_start + len(f"```{lang}\n")
        
        # Split on function boundaries (simplistic regex for python/js/etc)
        # Match 'def ' or 'class ' or 'function ' at start of line
        boundary_pattern = re.compile(r'^(?:def |class |function |const \w+ = \(.*?\) =>)', re.MULTILINE)
        
        splits = []
        last_idx = 0
        
        for bound in boundary_pattern.finditer(code):
            if bound.start() > last_idx:
                splits.append({
                    "text": code[last_idx:bound.start()].strip(),
                    "offset": last_idx + (len(code[last_idx:bound.start()]) - len(code[last_idx:bound.start()].lstrip()))
                })
            last_idx = bound.start()
            
        if last_idx < len(code):
            splits.append({
                "text": code[last_idx:].strip(),
                "offset": last_idx + (len(code[last_idx:]) - len(code[last_idx:].lstrip()))
            })
            
        # If no boundaries found, it's a single chunk
        if not splits:
            splits = [{"text": code.strip(), "offset": 0}]
            
        for split_item in splits:
            split_code = split_item["text"]
            split_offset = split_item["offset"]
            if not split_code:
                continue
                
            formatted_code = f"```{lang}\n{split_code}\n```"
            
            chunk_id = str(uuid.uuid4())
            embed_text = format_embed_text(doc.title or "", heading_path, formatted_code)
            
            # Use exact match offsets instead of string searching
            abs_start = code_start + split_offset
            abs_end = abs_start + len(split_code)
            
            metadata = ChunkMetadata(
                doc_id=doc.doc_id,
                original_filename=doc.original_filename,
                title=doc.title,
                heading_path=heading_path,
                chunk_type="code"
            )
            
            chunk = Chunk(
                chunk_id=chunk_id,
                parent_window_id="none", # Code blocks act standalone
                segment_id="none",
                chunk_text=formatted_code,
                embed_text=embed_text,
                char_start=abs_start,
                char_end=abs_end,
                token_count=count_tokens(formatted_code),
                metadata=metadata
            )
            chunks.append(chunk)
            
        # Remove from text
        clean_text = clean_text.replace(match.group(0), "", 1)  # only first occurrence
        
    return clean_text, chunks

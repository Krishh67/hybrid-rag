import uuid
from typing import List
from ingestion.schema import ParsedDocument
from .schema import Segment
from .config import config
from .utils import count_tokens

def _create_pseudo_segments(segment_text: str, doc_id: str, heading_path: List[str], base_char_start: int) -> List[Segment]:
    """Splits a single large segment into pseudo-segments respecting paragraphs."""
    paragraphs = segment_text.split('\n\n')
    segments = []
    current_paragraphs = []
    current_tokens = 0
    chunk_start_rel = 0
    
    for p in paragraphs:
        p_tokens = count_tokens(p)
        if current_tokens + p_tokens > config.MAX_SEGMENT_TOKENS and current_paragraphs:
            text = "\n\n".join(current_paragraphs)
            
            idx = segment_text.find(text, chunk_start_rel)
            if idx != -1:
                abs_start = base_char_start + idx
                chunk_start_rel = idx + len(text)
            else:
                abs_start = base_char_start + chunk_start_rel
                chunk_start_rel += len(text)
                
            abs_end = abs_start + len(text)
            
            segments.append(Segment(
                segment_id=str(uuid.uuid4()),
                doc_id=doc_id,
                heading_path=heading_path,
                text=text,
                char_start=abs_start,
                char_end=abs_end,
                token_count=current_tokens
            ))
            current_paragraphs = [p]
            current_tokens = p_tokens
        else:
            current_paragraphs.append(p)
            current_tokens += p_tokens
            
    if current_paragraphs:
        text = "\n\n".join(current_paragraphs)
        idx = segment_text.find(text, chunk_start_rel)
        if idx != -1:
            abs_start = base_char_start + idx
        else:
            abs_start = base_char_start + chunk_start_rel
            
        abs_end = abs_start + len(text)
            
        segments.append(Segment(
            segment_id=str(uuid.uuid4()),
            doc_id=doc_id,
            heading_path=heading_path,
            text=text,
            char_start=abs_start,
            char_end=abs_end,
            token_count=current_tokens
        ))
        
    return segments

def segment_document(doc: ParsedDocument, clean_text: str) -> List[Segment]:
    """
    Segments the document text.
    Tier 1/2: Uses headings.
    Tier 3: Uses paragraphs.
    """
    segments = []
    
    if doc.structure_tier in (1, 2) and doc.headings:
        # Sort headings by char_offset
        headings = sorted(doc.headings, key=lambda h: h.char_offset)
        
        # Heading path tracking
        current_path = []
        
        last_offset = 0
        
        # If there's text before the first heading
        if headings[0].char_offset > 0:
            pre_text = clean_text[:headings[0].char_offset].strip()
            if pre_text:
                pre_start = clean_text.find(pre_text)
                if pre_start == -1: pre_start = 0
                segments.extend(_create_pseudo_segments(pre_text, doc.doc_id, [], pre_start))
                
        for i, heading in enumerate(headings):
            start = heading.char_offset
            end = headings[i+1].char_offset if i + 1 < len(headings) else len(clean_text)
            
            # Text for this section includes the heading itself if we want, or we can just use the body.
            # Using the exact slice
            text_slice = clean_text[start:end].strip()
            
            # Update heading path stack
            # Remove headings of equal or greater level
            current_path = [h for h in current_path if h["level"] < heading.level]
            current_path.append({"level": heading.level, "text": heading.text})
            
            path_strings = [h["text"] for h in current_path]
            
            if text_slice:
                slice_start = clean_text.find(text_slice, start)
                if slice_start == -1: slice_start = start
                slice_tokens = count_tokens(text_slice)
                
                if slice_tokens > config.MAX_SEGMENT_TOKENS:
                    segments.extend(_create_pseudo_segments(text_slice, doc.doc_id, path_strings, slice_start))
                else:
                    segments.append(Segment(
                        segment_id=str(uuid.uuid4()),
                        doc_id=doc.doc_id,
                        heading_path=path_strings,
                        text=text_slice,
                        char_start=slice_start,
                        char_end=slice_start + len(text_slice),
                        token_count=slice_tokens
                    ))
    else:
        # Tier 3
        segments.extend(_create_pseudo_segments(clean_text, doc.doc_id, [], 0))
        
    return segments

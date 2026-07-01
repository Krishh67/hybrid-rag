import uuid
from typing import List
from ingestion.schema import ParsedDocument
from .schema import Segment, ParentWindow
from .config import config
from .sentence_splitter import split_sentences
from .utils import count_tokens

def build_parent_windows(doc: ParsedDocument, segment: Segment) -> List[ParentWindow]:
    """
    Takes a Segment and splits it into ParentWindow objects.
    Sentence aligned. Cap is MAX_PARENT_WINDOW_TOKENS.
    """
    windows = []
    
    if segment.token_count <= config.MAX_PARENT_WINDOW_TOKENS:
        window = ParentWindow(
            parent_id=str(uuid.uuid4()),
            segment_id=segment.segment_id,
            text=segment.text,
            char_start=segment.char_start,
            char_end=segment.char_end,
            token_count=segment.token_count
        )
        windows.append(window)
        return windows
        
    sentences = split_sentences(segment.text)
    # Pre-compute all token counts ONCE — same fix as chunk_builder
    sentence_token_counts = [count_tokens(s) for s in sentences]
    current_window_sentences = []
    current_tokens = 0
    
    current_idx = 0
    
    for idx_s, sentence in enumerate(sentences):
        sentence_tokens = sentence_token_counts[idx_s]
        
        if current_tokens + sentence_tokens > config.MAX_PARENT_WINDOW_TOKENS and current_window_sentences:
            window_text = "".join(current_window_sentences)
            
            idx = segment.text.find(window_text, current_idx)
            if idx != -1:
                abs_start = segment.char_start + idx
                current_idx = idx + len(window_text)
            else:
                abs_start = segment.char_start + current_idx
                current_idx += len(window_text)
                
            abs_end = abs_start + len(window_text)
            
            window = ParentWindow(
                parent_id=str(uuid.uuid4()),
                segment_id=segment.segment_id,
                text=window_text,
                char_start=abs_start,
                char_end=abs_end,
                token_count=current_tokens
            )
            windows.append(window)
            
            current_window_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            current_window_sentences.append(sentence)
            current_tokens += sentence_tokens
            
    if current_window_sentences:
        window_text = "".join(current_window_sentences)
        
        idx = segment.text.find(window_text, current_idx)
        if idx != -1:
            abs_start = segment.char_start + idx
        else:
            abs_start = segment.char_start + current_idx
            
        abs_end = abs_start + len(window_text)
        
        window = ParentWindow(
            parent_id=str(uuid.uuid4()),
            segment_id=segment.segment_id,
            text=window_text,
            char_start=abs_start,
            char_end=abs_end,
            token_count=current_tokens
        )
        windows.append(window)
        
    return windows

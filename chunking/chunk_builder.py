import uuid
from typing import List
from ingestion.schema import ParsedDocument
from .schema import ParentWindow, Chunk, ChunkMetadata
from .config import config
from .sentence_splitter import split_sentences
from .utils import count_tokens, format_embed_text

def build_chunks_from_window(doc: ParsedDocument, parent: ParentWindow, heading_path: List[str]) -> List[Chunk]:
    """
    Builds 200-400 token chunks using a sentence-aligned sliding window.
    Overlap is strictly within the parent window.
    """
    chunks = []
    sentences = split_sentences(parent.text)
    
    if not sentences:
        return []
    
    # Pre-compute all token counts ONCE — avoids calling tiktoken on every boundary check
    sentence_tokens = [count_tokens(s) for s in sentences]
        
    current_chunk_sentences = []
    current_tokens = 0
    
    current_idx = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        s_tokens = sentence_tokens[i]
        
        # If a single sentence exceeds the token limit, force split it by words to guarantee cap compliance
        if s_tokens > config.MAX_CHUNK_TOKENS:
            import re
            parts = re.split(r'([\s\n]+)', sentence)
            sub_sentences = []
            curr_sub = ""
            curr_tokens = 0
            
            for j in range(0, len(parts)-1, 2):
                word_delim = parts[j] + parts[j+1]
                wt = count_tokens(word_delim)
                
                # Extreme fallback for insanely long unbroken strings (e.g. base64)
                if wt > config.MAX_CHUNK_TOKENS:
                    if curr_sub:
                        sub_sentences.append(curr_sub)
                        curr_sub = ""
                        curr_tokens = 0
                    
                    chunk_sz = 1000
                    for k in range(0, len(word_delim), chunk_sz):
                        sub_sentences.append(word_delim[k:k+chunk_sz])
                    continue
                    
                if curr_tokens + wt > config.MAX_CHUNK_TOKENS and curr_sub:
                    sub_sentences.append(curr_sub)
                    curr_sub = word_delim
                    curr_tokens = wt
                else:
                    curr_sub += word_delim
                    curr_tokens += wt
                    
            tail = parts[-1] if len(parts) % 2 != 0 else ""
            if tail:
                wt = count_tokens(tail)
                if curr_tokens + wt > config.MAX_CHUNK_TOKENS and curr_sub:
                    sub_sentences.append(curr_sub)
                    curr_sub = tail
                else:
                    curr_sub += tail
                    
            if curr_sub:
                sub_sentences.append(curr_sub)
                
            # Replace the massive sentence with our sub-sentences and continue processing
            sentences = sentences[:i] + sub_sentences + sentences[i+1:]
            new_tokens = [count_tokens(s) for s in sub_sentences]
            sentence_tokens = sentence_tokens[:i] + new_tokens + sentence_tokens[i+1:]
            continue
            
        # If adding this sentence would exceed the limit and we have content, flush current chunk
        if current_tokens + s_tokens > config.MAX_CHUNK_TOKENS and current_chunk_sentences:
            # Finalize current chunk
            chunk_text_raw = "".join(current_chunk_sentences)
            chunk_text = chunk_text_raw.strip()
            
            idx = parent.text.find(chunk_text_raw, current_idx)
            
            overlap_sentences = current_chunk_sentences[-config.CHUNK_OVERLAP_SENTENCES:] if config.CHUNK_OVERLAP_SENTENCES > 0 else []
            
            if idx != -1:
                strip_idx = chunk_text_raw.find(chunk_text)
                abs_start = parent.char_start + idx + (strip_idx if strip_idx != -1 else 0)
                current_idx = idx + len(chunk_text_raw) - sum(len(s) for s in overlap_sentences)
            else:
                abs_start = parent.char_start + current_idx
                current_idx += len(chunk_text_raw)
                
            abs_end = abs_start + len(chunk_text)
            
            chunk_id = str(uuid.uuid4())
            embed_text = format_embed_text(doc.title or "", heading_path, chunk_text)
            
            metadata = ChunkMetadata(
                doc_id=doc.doc_id,
                original_filename=doc.original_filename,
                title=doc.title,
                heading_path=heading_path,
                chunk_type="text"
            )
            
            chunk = Chunk(
                chunk_id=chunk_id,
                parent_window_id=parent.parent_id,
                segment_id=parent.segment_id,
                chunk_text=chunk_text,
                embed_text=embed_text,
                char_start=abs_start,
                char_end=abs_end,
                token_count=current_tokens,
                metadata=metadata
            )
            chunks.append(chunk)
            
            # Start new chunk with overlap sentences, then immediately add current sentence
            # IMPORTANT: always advance i to prevent infinite loop
            overlap_sentences = current_chunk_sentences[-config.CHUNK_OVERLAP_SENTENCES:] if config.CHUNK_OVERLAP_SENTENCES > 0 else []
            
            # Calculate token count of the overlap
            overlap_tokens = 0
            if overlap_sentences:
                overlap_start_idx = i - len(overlap_sentences)
                overlap_tokens = sum(sentence_tokens[j] for j in range(overlap_start_idx, i))
                
            # Prune overlap from the left if adding the current sentence would violate the cap
            while overlap_sentences and overlap_tokens + s_tokens > config.MAX_CHUNK_TOKENS:
                dropped_idx = i - len(overlap_sentences)
                overlap_tokens -= sentence_tokens[dropped_idx]
                overlap_sentences.pop(0)
                
            current_chunk_sentences = list(overlap_sentences) + [sentence]
            current_tokens = overlap_tokens + s_tokens
            i += 1
            continue
            
        current_chunk_sentences.append(sentence)
        current_tokens += s_tokens
        
        i += 1

        
    # Finalize remainder
    if current_chunk_sentences:
        chunk_text_raw = "".join(current_chunk_sentences)
        chunk_text = chunk_text_raw.strip()
        if chunk_text:
            idx = parent.text.find(chunk_text_raw, current_idx)
            if idx != -1:
                strip_idx = chunk_text_raw.find(chunk_text)
                abs_start = parent.char_start + idx + (strip_idx if strip_idx != -1 else 0)
            else:
                abs_start = parent.char_start + current_idx
                
            abs_end = abs_start + len(chunk_text)
            
            chunk_id = str(uuid.uuid4())
            embed_text = format_embed_text(doc.title or "", heading_path, chunk_text)
            
            metadata = ChunkMetadata(
                doc_id=doc.doc_id,
                original_filename=doc.original_filename,
                title=doc.title,
                heading_path=heading_path,
                chunk_type="text"
            )
            
            chunk = Chunk(
                chunk_id=chunk_id,
                parent_window_id=parent.parent_id,
                segment_id=parent.segment_id,
                chunk_text=chunk_text,
                embed_text=embed_text,
                char_start=abs_start,
                char_end=abs_end,
                token_count=count_tokens(chunk_text),
                metadata=metadata
            )
            chunks.append(chunk)
            
    return chunks

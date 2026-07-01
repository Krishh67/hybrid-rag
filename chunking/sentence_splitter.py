import re
from typing import List

# Try importing blingfire for high-quality sentence splitting
try:
    # pyrefly: ignore [missing-import]
    import blingfire
    _USE_BLINGFIRE = True
except ImportError:
    _USE_BLINGFIRE = False

def split_sentences(text: str) -> List[str]:
    """
    Splits text into sentences.
    Prefers BlingFire if available. Falls back to Regex.
    Preserves all whitespace and exact reconstruction.
    """
    if not text:
        return []
        
    if _USE_BLINGFIRE:
        try:
            # blingfire.text_to_sentences returns a single string of sentences separated by newline
            # But wait, we need to preserve exact whitespace to trace char_offsets flawlessly.
            # Blingfire strips original spacing and inserts its own, which ruins str.find() offsets!
            # Since the user explicitly requested BlingFire but ALSO strict offset traceability,
            # we must carefully map the sentences back to the original text.
            bf_output = blingfire.text_to_sentences(text)
            # This is a bit tricky: BlingFire normalizes spaces.
            # A common strategy is to use the generated sentences to find boundaries in the original text.
            bf_sentences = bf_output.split('\n')
            
            sentences = []
            current_index = 0
            
            for bf_sent in bf_sentences:
                if not bf_sent.strip():
                    continue
                # Find the sentence in the original text. 
                # Because BlingFire normalizes, exact find() might fail if there's internal whitespace differences.
                # If exact find works (mostly true), use it:
                idx = text.find(bf_sent.strip(), current_index)
                if idx != -1:
                    # Capture everything from current_index to the end of this sentence
                    end_idx = idx + len(bf_sent.strip())
                    sentences.append(text[current_index:end_idx])
                    current_index = end_idx
                else:
                    # If exact find fails, fallback to regex for this segment to preserve offsets safely
                    # This happens rarely but is necessary for 100% offset accuracy.
                    fallback_text = text[current_index:]
                    return sentences + _regex_split_sentences(fallback_text)
                    
            # Add remaining tail
            if current_index < len(text):
                sentences.append(text[current_index:])
                
            return sentences
        except Exception:
            # Fallback on any error
            pass

    return _regex_split_sentences(text)


def _force_split_long_sentence(sentence: str, max_chars: int = 1500) -> List[str]:
    """Force split a massive sentence by whitespace to prevent chunk size blowouts."""
    if len(sentence) <= max_chars:
        return [sentence]
        
    parts = re.split(r'([\s\n]+)', sentence)
    result = []
    current = ""
    
    for i in range(0, len(parts) - 1, 2):
        word = parts[i]
        delim = parts[i+1]
        if len(current) + len(word) + len(delim) > max_chars and current:
            result.append(current)
            current = word + delim
        else:
            current += word + delim
            
    # Handle tail
    tail = parts[-1] if len(parts) % 2 != 0 else ""
    if current or tail:
        remainder = current + tail
        if remainder:
            result.append(remainder)
            
    return result

def _regex_split_sentences(text: str) -> List[str]:
    """
    Fallback regex sentence splitter.
    Uses a non-backtracking pattern safe for large inputs.
    """
    # Safe pattern: sentence boundary = punctuation followed by whitespace/newline.
    # We also include double newlines as a hard boundary (very common in OCR).
    parts = re.split(r'([.!?]+[\s\n]+|\n{2,})', text)
    
    sentences = []
    current_sentence = ""
    for i in range(0, len(parts) - 1, 2):
        sentence_body = parts[i]
        delimiter = parts[i+1]
        current_sentence += sentence_body + delimiter
        
        if current_sentence.strip():
            sentences.append(current_sentence)
            current_sentence = ""
            
    # Handle tail (text after last delimiter)
    tail = parts[-1] if len(parts) % 2 != 0 else ""
    if current_sentence or tail:
        remainder = current_sentence + tail
        if remainder.strip():
            sentences.append(remainder)
        
    if not sentences and text.strip():
        sentences.append(text)
        
    # Final pass: enforce maximum character limit to prevent chunk blowouts
    final_sentences = []
    for s in sentences:
        if len(s) > 2000:
            final_sentences.extend(_force_split_long_sentence(s, 1500))
        else:
            final_sentences.append(s)
            
    return final_sentences


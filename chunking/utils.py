import re
import tiktoken

# Use a global encoder to avoid loading it multiple times
_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Returns the number of tokens in a text string."""
    if not text:
        return 0
    return len(_encoder.encode(text))



def format_embed_text(title: str, heading_path: list[str], chunk_text: str) -> str:
    """Constructs the embedding string according to the specification."""
    parts = []
    if title:
        parts.append(title)
    if heading_path:
        parts.append(" > ".join(heading_path))
    parts.append(chunk_text.strip())
    
    return "\n\n".join(parts)

import zipfile
from pathlib import Path
from typing import Literal

def detect_file_type(file_path: str | Path) -> Literal["pdf", "docx", "doc", "txt", "zip", "unknown"]:
    """Detects file type based on magic bytes and content, ignoring extensions."""
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return "unknown"
        
    try:
        with open(path, 'rb') as f:
            header = f.read(8)
            
        if header.startswith(b"%PDF"):
            return "pdf"
            
        if header.startswith(b"PK\x03\x04"):
            # It's a ZIP archive. Check if it's a DOCX by looking inside.
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    file_list = zf.namelist()
                    if "[Content_Types].xml" in file_list and any(f.startswith("word/") for f in file_list):
                        return "docx"
                    return "zip"
            except zipfile.BadZipFile:
                # If it's a corrupted zip, we might just fail here, or return unknown
                return "unknown"
                
        # Legacy DOC format
        if header == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            return "doc"
            
        # If not any of the above binaries, let's treat it as txt
        # TXT parser will handle charset detection
        return "txt"
        
    except Exception:
        return "unknown"

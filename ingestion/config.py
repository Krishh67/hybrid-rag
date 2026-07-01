import os
from typing import Set

class Config:
    # Feature flags
    OCR_ENABLED: bool = os.getenv("RAG_OCR_ENABLED", "true").lower() == "true"
    OCR_LANGUAGE: str = os.getenv("RAG_OCR_LANGUAGE", "eng")
    TESSERACT_CMD: str = os.getenv("RAG_TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    
    # Limits
    MAX_FILE_SIZE_BYTES: int = int(os.getenv("RAG_MAX_FILE_SIZE", str(50 * 1024 * 1024))) # 50 MB
    MAX_UNCOMPRESSED_ZIP_SIZE: int = int(os.getenv("RAG_MAX_UNCOMPRESSED_ZIP_SIZE", str(500 * 1024 * 1024))) # 500 MB
    MAX_ZIP_DEPTH: int = int(os.getenv("RAG_MAX_ZIP_DEPTH", "3"))
    MAX_ZIP_FILES: int = int(os.getenv("RAG_MAX_ZIP_FILES", "1000"))
    
    # Thresholds
    LOW_TEXT_THRESHOLD: int = int(os.getenv("RAG_LOW_TEXT_THRESHOLD", "100")) # chars
    
    # Supported types
    SUPPORTED_EXTENSIONS: Set[str] = {
        ".pdf",
        ".doc", ".docx",
        ".txt",
        ".zip"
    }

config = Config()








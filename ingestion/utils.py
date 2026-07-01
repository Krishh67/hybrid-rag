import hashlib
import logging
import json
import sys
from pathlib import Path

def setup_logger() -> logging.Logger:
    """Sets up a structured JSON logger for the ingestion module."""
    logger = logging.getLogger("ingestion")
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_record = {
                    "level": record.levelname,
                    "message": record.getMessage(),
                }
                # Include extra kwargs if provided
                if hasattr(record, "extra_data"):
                    log_record.update(record.extra_data) # type: ignore
                return json.dumps(log_record)
                
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()

def compute_file_hash(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file's raw bytes."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

def clean_text(text: str) -> str:
    """Standardizes text: trims whitespace, normalizes newlines."""
    if not text:
        return ""
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines]
    return "\n".join(cleaned_lines).strip()

from typing import List, Optional
from .schema import TableEntry

def extract_text_tables(text: str, page_num: Optional[int] = None) -> List[TableEntry]:
    """
    Lightweight fallback detector for plain text tables (pipe or tab delimited).
    Requires at least 2 delimiters, multiple consecutive rows, and consistent column counts.
    """
    lines = text.splitlines()
    tables = []
    
    current_table_lines = []
    current_delimiter = None
    current_col_count = 0
    
    def finalize_table():
        if len(current_table_lines) >= 3:
            md_lines = []
            for i, line in enumerate(current_table_lines):
                cells = [c.strip() for c in line.split(current_delimiter)]
                md_line = "| " + " | ".join(cells) + " |"
                md_lines.append(md_line)
                
                if i == 0:
                    separator = ["---"] * len(cells)
                    md_lines.append("| " + " | ".join(separator) + " |")
                    
            tables.append(TableEntry(page=page_num, markdown="\n".join(md_lines)))
        current_table_lines.clear()

    for line in lines:
        cleaned_line = line.strip()
        pipe_count = cleaned_line.count('|')
        tab_count = cleaned_line.count('\t')
        
        is_row = False
        delimiter = None
        col_count = 0
        
        if pipe_count >= 3:
            delimiter = '|'
            col_count = pipe_count + 1
            is_row = True
        elif tab_count >= 3:
            delimiter = '\t'
            col_count = tab_count + 1
            is_row = True
            
        if is_row:
            if not current_table_lines:
                current_delimiter = delimiter
                current_col_count = col_count
                current_table_lines.append(cleaned_line)
            else:
                if delimiter == current_delimiter and col_count == current_col_count:
                    current_table_lines.append(cleaned_line)
                else:
                    finalize_table()
                    current_delimiter = delimiter
                    current_col_count = col_count
                    current_table_lines.append(cleaned_line)
        else:
            if current_table_lines:
                finalize_table()
                
    if current_table_lines:
        finalize_table()
        
    return tables

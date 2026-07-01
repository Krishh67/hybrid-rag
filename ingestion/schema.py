from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

class HeadingEntry(BaseModel):
    level: int
    text: str
    char_offset: int

class TableEntry(BaseModel):
    page: Optional[int]
    markdown: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None

class ParsedDocument(BaseModel):
    doc_id: str
    source_path: str
    original_filename: str
    file_type: Literal["pdf", "docx", "txt"]
    content_hash: str               # SHA-256 of raw bytes
    full_text: str
    structure_tier: Literal[1, 2, 3]
    headings: list[HeadingEntry]
    title: Optional[str]
    title_source: Literal["native", "heuristic", "none"]
    page_count: Optional[int]
    language: Optional[str]
    ocr_used: bool
    parser_used: str
    
    tables: list[TableEntry] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    extraction_errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

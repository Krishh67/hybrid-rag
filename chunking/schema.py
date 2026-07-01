from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

class ChunkMetadata(BaseModel):
    doc_id: str
    original_filename: str
    title: Optional[str]
    heading_path: List[str]
    chunk_type: Literal["text", "table", "code", "figure"]

class Chunk(BaseModel):
    chunk_id: str
    parent_window_id: str
    segment_id: str
    chunk_text: str
    embed_text: str
    char_start: int
    char_end: int
    token_count: int
    metadata: ChunkMetadata

class ParentWindow(BaseModel):
    parent_id: str
    segment_id: str
    text: str
    char_start: int
    char_end: int
    token_count: int
    chunks: List[Chunk] = Field(default_factory=list)

class Segment(BaseModel):
    segment_id: str
    doc_id: str
    heading_path: List[str]
    text: str
    char_start: int
    char_end: int
    token_count: int
    parent_windows: List[ParentWindow] = Field(default_factory=list)

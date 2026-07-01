from typing import List, Dict, Any
from pydantic import BaseModel

class SourceReference(BaseModel):
    doc_id: str
    parent_window_id: str

class GenerationResult(BaseModel):
    answer: str
    sources: List[SourceReference]
    model_used: str
    usage_metadata: Dict[str, Any]

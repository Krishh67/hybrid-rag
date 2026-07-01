# RAG Ingestion & Parsing Module (Stage 1)

This module handles the ingestion and parsing of arbitrary file formats (`.pdf`, `.docx`, `.txt`, `.zip`, `.doc`) into a single, normalized `ParsedDocument` schema. It serves as Stage 1 of a 6-stage RAG pipeline.

## Features
- **Format Agnostic Output:** Converts everything into one predictable Pydantic v2 schema.
- **Structure Tier Detection:**
  - **Tier 1:** Native structure (e.g. DOCX heading styles, PDF bookmarks).
  - **Tier 2:** Visual heuristics (e.g. PDF font sizes).
  - **Tier 3:** No structure (e.g. TXT, Scanned PDFs without layout).
- **Graceful OCR Fallback:** Uses `pytesseract` automatically when little text is found in PDFs.
- **Robust ZIP Handling:** Guards against zip-bombs and path-traversal attacks while recursing.
- **Idempotency:** SHA-256 hash checking prevents re-processing.
- **Safe Isolation:** Batch processing wraps each file in try/catch to avoid full-batch crashes.

## Output Schema
Every parsed document conforms to `ParsedDocument` from `ingestion.schema`.

```python
class ParsedDocument(BaseModel):
    doc_id: str
    source_path: str
    original_filename: str
    file_type: Literal["pdf", "docx", "txt"]
    content_hash: str               
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
```

## Running the Module

```python
from ingestion.pipeline import ingest_batch

# Processes files, directories, and zip files safely
docs = ingest_batch(["path/to/my_file.pdf", "path/to/docs_dir/"])
for doc in docs:
    print(f"{doc.original_filename} (Tier {doc.structure_tier}) -> {len(doc.full_text)} chars")
```

## Testing

Ensure you have the required dependencies installed (including `tesseract` binary in your PATH if you wish to run full OCR integration, although tests will patch it).

To run the full suite covering all 12 acceptance cases:
```bash
$env:PYTHONPATH="."
pytest tests/ -v
```

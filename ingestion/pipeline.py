import uuid
import time
import tempfile
from pathlib import Path
from typing import List, Set, Optional, Callable

from .schema import ParsedDocument
from .detectors import detect_file_type
from .parsers.pdf_parser import parse_pdf
from .parsers.docx_parser import parse_docx
from .parsers.txt_parser import parse_txt
from .zip_handler import extract_zip_safe
from .utils import compute_file_hash, logger

def ingest_file(file_path: str | Path, already_processed_hashes: Optional[Set[str]] = None) -> Optional[ParsedDocument]:
    """
    Ingests a single file. 
    Routes to correct parser based on content detection.
    """
    path = Path(file_path)
    start_time = time.time()
    
    try:
        content_hash = compute_file_hash(path)
        if already_processed_hashes is not None:
            if content_hash in already_processed_hashes:
                logger.info(f"Skipping already processed file: {path.name} (hash: {content_hash})")
                return None
            already_processed_hashes.add(content_hash)
            
        file_type = detect_file_type(path)
        doc_id = str(uuid.uuid4())
        
        doc = None
        
        if file_type == "pdf":
            doc = parse_pdf(path, doc_id, path.name)
        elif file_type == "docx":
            doc = parse_docx(path, doc_id, path.name)
        elif file_type == "txt":
            doc = parse_txt(path, doc_id, path.name)
        elif file_type == "doc":
            # Document limitation as requested
            logger.error(f"Legacy .doc format not supported: {path.name}")
            doc = ParsedDocument(
                doc_id=doc_id,
                source_path=str(path),
                original_filename=path.name,
                file_type="txt", # Defaulting to txt for schema compatibility when unsupported
                content_hash=content_hash,
                full_text="",
                structure_tier=3,
                headings=[],
                title=None,
                title_source="none",
                page_count=None,
                language=None,
                ocr_used=False,
                parser_used="none",
                tables=[],
                extraction_warnings=[],
                extraction_errors=["Legacy .doc format requires external converter."],
                metadata={}
            )
        else:
            logger.warning(f"Unsupported or unknown file type for: {path.name}")
            return None
            
        duration = time.time() - start_time
        status = "success" if not doc.extraction_errors else "error"
        logger.info(
            f"Processed {path.name}", 
            extra={"extra_data": {"tier": doc.structure_tier, "status": status, "duration_s": round(duration, 2)}}
        )
        return doc
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unhandled error processing {path.name}: {e}", extra={"extra_data": {"status": "error", "duration_s": round(duration, 2)}})
        return None


def ingest_path(target_path: str | Path, already_processed_hashes: Optional[Set[str]] = None) -> List[ParsedDocument]:
    """
    Ingests a target path which could be a single file, a directory, or a ZIP archive.
    """
    path = Path(target_path)
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        return []
        
    docs = []
    
    if path.is_file():
        file_type = detect_file_type(path)
        if file_type == "zip":
            with tempfile.TemporaryDirectory() as temp_dir:
                extracted_files = extract_zip_safe(path, temp_dir)
                for extracted_file in extracted_files:
                    doc = ingest_file(extracted_file, already_processed_hashes)
                    if doc:
                        # Restore original source_path context if desired, or keep temp path. 
                        # We'll keep temp path for debuggability but note original filename.
                        doc.original_filename = f"{path.name}::{extracted_file.name}"
                        docs.append(doc)
        else:
            doc = ingest_file(path, already_processed_hashes)
            if doc:
                docs.append(doc)
                
    elif path.is_dir():
        for file_in_dir in path.rglob("*"):
            if file_in_dir.is_file():
                # For directories, if a file is a zip, we can handle it or skip.
                # To keep it simple, we process it normally.
                file_type = detect_file_type(file_in_dir)
                if file_type == "zip":
                    with tempfile.TemporaryDirectory() as temp_dir:
                        extracted_files = extract_zip_safe(file_in_dir, temp_dir)
                        for extracted_file in extracted_files:
                            doc = ingest_file(extracted_file, already_processed_hashes)
                            if doc:
                                doc.original_filename = f"{file_in_dir.name}::{extracted_file.name}"
                                docs.append(doc)
                else:
                    doc = ingest_file(file_in_dir, already_processed_hashes)
                    if doc:
                        docs.append(doc)
                        
    return docs

def ingest_batch(paths: List[str | Path], already_processed_hashes: Optional[Set[str]] = None) -> List[ParsedDocument]:
    """Ingests a batch of paths."""
    all_docs = []
    for path in paths:
        docs = ingest_path(path, already_processed_hashes)
        all_docs.extend(docs)
    return all_docs

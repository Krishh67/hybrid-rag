import zipfile
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from .config import config
from .utils import logger
from .detectors import detect_file_type

class ZipBombError(Exception):
    pass

class PathTraversalError(Exception):
    pass

def _is_safe_path(base_dir: Path, target_path: str) -> bool:
    """Checks for path traversal attempts."""
    resolved_target = (base_dir / target_path).resolve()
    return str(resolved_target).startswith(str(base_dir.resolve()))

def extract_zip_safe(zip_path: str | Path, extract_to: str | Path, max_depth: int = config.MAX_ZIP_DEPTH, current_depth: int = 0) -> List[Path]:
    """
    Safely extracts a ZIP file, preventing zip bombs and path traversal.
    Returns a list of extracted file paths.
    """
    if current_depth > max_depth:
        logger.warning(f"Max ZIP extraction depth ({max_depth}) exceeded for {zip_path}")
        return []

    extracted_files = []
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total_size = 0
            file_count = 0

            for zip_info in zf.infolist():
                if zip_info.is_dir():
                    continue

                if not _is_safe_path(extract_to, zip_info.filename):
                    raise PathTraversalError(f"Path traversal detected: {zip_info.filename}")

                total_size += zip_info.file_size
                if total_size > config.MAX_UNCOMPRESSED_ZIP_SIZE:
                    raise ZipBombError("Total uncompressed size exceeds limit.")

                file_count += 1
                if file_count > config.MAX_ZIP_FILES:
                    raise ZipBombError("Total file count exceeds limit.")

                # Extract safely
                target_path = extract_to / zip_info.filename
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                with zf.open(zip_info) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

                # Check if nested ZIP
                file_type = detect_file_type(target_path)
                if file_type == "zip":
                    nested_extract_to = extract_to / f"{zip_info.filename}_extracted"
                    nested_files = extract_zip_safe(target_path, nested_extract_to, max_depth, current_depth + 1)
                    extracted_files.extend(nested_files)
                    # We can optionally remove the intermediate zip to save space
                    target_path.unlink()
                elif file_type in ["pdf", "docx", "txt", "doc"]:
                    extracted_files.append(target_path)
                else:
                    logger.info(f"Skipping unsupported file type in zip: {zip_info.filename}")
                    target_path.unlink() # Clean up unsupported file

    except zipfile.BadZipFile:
        logger.error(f"Bad or corrupted ZIP file: {zip_path}")
    except (ZipBombError, PathTraversalError) as e:
        logger.error(f"Security error extracting {zip_path}: {e}")

    return extracted_files

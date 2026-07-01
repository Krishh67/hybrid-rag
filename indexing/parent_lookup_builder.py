import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

PARENT_LOOKUP_FILENAME = "parent_lookup.pkl"

def save_parent_lookup(lookup: Dict[str, dict], out_dir: Path) -> Path:
    """Pickle the parent lookup dictionary to {out_dir}/parent_lookup.pkl.
    
    The dict format is:
    { "parent_window_id": {"doc_id": "...", "char_start": ..., "char_end": ...} }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PARENT_LOOKUP_FILENAME
    with path.open("wb") as fh:
        pickle.dump(lookup, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved parent lookup → %s  (%d records)", path, len(lookup))
    return path

def load_parent_lookup(index_dir: Path) -> Dict[str, dict]:
    """Load the parent lookup dictionary from pickle.
    
    Returns an empty dictionary if the file does not exist (for fresh index).
    """
    path = index_dir / PARENT_LOOKUP_FILENAME
    if not path.exists():
        return {}

    with path.open("rb") as fh:
        lookup = pickle.load(fh)

    if not isinstance(lookup, dict):
        raise TypeError(f"Expected dict but got {type(lookup).__name__}.")
        
    logger.info("Loaded parent lookup from %s  (%d records)", path, len(lookup))
    return lookup

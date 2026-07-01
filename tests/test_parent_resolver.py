import pytest
import json
from pathlib import Path
from retrieval.parent_resolver import ParentWindowResolver

def test_parent_window_resolver(tmp_path):
    kb_dir = tmp_path / "kb_test"
    parsed_dir = kb_dir / "parsed"
    parent_windows_dir = kb_dir / "parent_windows"
    
    parsed_dir.mkdir(parents=True)
    parent_windows_dir.mkdir(parents=True)
    
    # Create fake parsed document
    full_text = "This is a test document. It has several sentences. Here is the parent text. And more text."
    doc_id = "doc-123"
    
    with open(parsed_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
        json.dump({"doc_id": doc_id, "full_text": full_text}, f)
        
    # Create fake parent window record
    # Let's extract "Here is the parent text."
    target_text = "Here is the parent text."
    char_start = full_text.find(target_text)
    char_end = char_start + len(target_text)
    
    parent_id = "parent-456"
    
    index_dir = kb_dir / "index"
    index_dir.mkdir(parents=True)
    
    from indexing.parent_lookup_builder import save_parent_lookup
    save_parent_lookup({
        parent_id: {
            "doc_id": doc_id,
            "char_start": char_start,
            "char_end": char_end
        }
    }, index_dir)
        
    resolver = ParentWindowResolver(str(kb_dir))
    
    # Test valid resolution
    resolved_text = resolver.resolve(parent_id)
    assert resolved_text == target_text
    
    # Test invalid id
    assert resolver.resolve("invalid-id") is None
    
    # Test 'none' id
    assert resolver.resolve("none") is None

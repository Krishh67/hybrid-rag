# Stage 5 — Index & Storage

Consumes all Stage 4 embedding outputs in a knowledge base and builds a unified, retrieval-ready index consisting of four files.

---

## Output Files

All outputs are written to `knowledge_bases/{kb_id}/index/`:

| File | Description |
|---|---|
| `faiss.index` | `IndexFlatIP` (inner-product) covering all chunks in the KB — equivalent to cosine similarity because Stage 4 vectors are L2-normalized |
| `bm25.pkl` | `BM25State` pickle: fitted `BM25Okapi` model + tokenized corpus (needed for incremental rebuild) |
| `metadata.pkl` | `list[RetrievalRecord]` — KB-wide mapping from FAISS row / BM25 doc index → chunk metadata |
| `manifest.json` | Lightweight JSON tracking doc count, chunk count, and which doc IDs are indexed |

---

## Alignment Invariant

**The FAISS row index `i`, the BM25 doc index `i`, and `metadata[i]` always refer to the same chunk.**

This invariant is enforced by `_validate_alignment()` in `pipeline.py` after every build or update. A `RuntimeError` is raised and logged at CRITICAL level if the counts diverge.

---

## Schema

```python
class RetrievalRecord(BaseModel):
    global_chunk_id: int       # FAISS row == BM25 doc index
    chunk_id: str              # Original Stage 3 UUID
    doc_id: str
    title: Optional[str]
    heading_path: list[str]
    original_filename: str
    chunk_type: str            # 'text' | 'table' | 'code' | 'figure'
    chunk_text: str            # BM25 corpus document
    embed_text_hash: str       # SHA-256 for dedup
```

---

## Incremental Updates

When new documents are added to a KB, `index_kb()` skips already-indexed documents (tracked in `manifest.json`) and **appends** only new vectors:

- **FAISS**: `index.add(new_vectors)` — appends without rebuilding
- **BM25**: rebuilds from the combined cached tokenized corpus (fast, O(N) where N = total docs)
- **Metadata**: list extended, `global_chunk_id` re-sequenced
- **Manifest**: `indexed_doc_ids`, `total_documents`, `total_chunks` updated

To add a new document to an existing KB, just run:
```bash
python main.py
# Select the new file → select existing KB → pipeline runs Stage 1-5 automatically
```

---

## FAISS Index Choice

`IndexFlatIP` (inner product) was chosen because:
- Stage 4 `model_wrapper.py` L2-normalizes all dense vectors before storage.
- For unit-norm vectors: `dot(a, b) = cos(θ)` — inner product equals cosine similarity.
- `IndexFlatIP` is exact (no approximation), deterministic, and fast enough for the KB sizes this pipeline targets.

---

## Running the Pipeline

```python
from indexing.pipeline import index_kb

result = index_kb("knowledge_bases/kb_001")
print(result.total_chunks, result.total_documents)
```

---

## Running Tests

```bash
python -m pytest tests/test_indexing.py -v
```

All 16 tests run in ~1 second (no model loading required).

---

## Acceptance Criteria

| # | Criterion | Test |
|---|---|---|
| 1 | Single document indexes successfully | `TestSingleDocument::test_indexes_successfully` |
| 2 | Multiple documents merge into one index | `TestMultipleDocuments::test_merges_into_one_index` |
| 3 | FAISS row count == metadata count | `TestAlignmentInvariant` |
| 4 | Manifest counts are correct | `TestManifest::test_manifest_counts_are_correct` |
| 5 | Incremental update adds only new vectors | `TestIncrementalUpdate` |
| 6 | FAISS index reload works after restart | `TestFaissReload::test_index_reload_after_restart` |
| 7 | BM25 reload works after restart | `TestBm25Reload::test_bm25_reload_after_restart` |
| 8 | Empty KB handled gracefully | `TestEmptyKB::test_empty_kb_handled_gracefully` |
| 9 | Corrupted embedding files fail with clear errors | `TestCorruptedFiles` |
| 10 | All tests pass | ✅ 16/16 |

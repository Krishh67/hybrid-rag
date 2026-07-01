# Stage 4 — Embedding Generation

Converts `Chunk` objects (Stage 3) into `EmbeddedChunk` objects containing a dense vector and a sparse lexical-weight vector produced by [BGE-M3](https://huggingface.co/BAAI/bge-m3).

---

## Module Structure

```
embedding/
  __init__.py
  schema.py          # EmbeddedChunk Pydantic model
  config.py          # All tuneable settings and env-var overrides
  model_wrapper.py   # BGE-M3 lazy singleton + encode_batch()
  cache.py           # Per-document hash-based cache load/merge
  pipeline.py        # Main entry: embed_document()
  utils.py           # hash_text, normalize_l2, npy I/O
```

---

## Schema

```python
class EmbeddedChunk(BaseModel):
    chunk_id: str
    dense_vector: list[float]           # 1024-dim, L2-normalized
    sparse_vector: dict[str, float]     # token → weight (non-zero only)
    embedding_model: str                # "BAAI/bge-m3"
    embedding_model_version: str        # model revision for auditability
    embed_text_hash: str                # SHA-256 of the embedded text
    dim: int                            # always 1024
    generated_at: datetime
```

---

## Storage Layout

Per document, two files are written to `knowledge_bases/{kb_id}/embeddings/`:

| File | Content |
|---|---|
| `{doc_id}_dense.npy` | `np.float32` array of shape `[n_chunks, 1024]` — row *i* matches `meta.json` entry *i* |
| `{doc_id}_meta.json` | JSON list of `EmbeddedChunk` objects in the same order — includes sparse vectors |

> **Alignment guarantee**: The `.npy` row index and `meta.json` array index are always identical. This alignment is enforced by `cache.merge_and_save()` and must never drift.

---

## Normalization

BGE-M3's `encode()` (via FlagEmbedding) returns dense vectors that are **already L2-normalized** when `normalize_embeddings=True` (the library default). The pipeline explicitly re-runs `normalize_l2()` as a documented safety net. Downstream cosine-similarity search therefore needs no normalization step — it can compute `dot(a, b)` directly.

---

## Caching

Re-embedding unchanged chunks on a CPU machine is expensive. Before encoding, the pipeline computes `SHA-256(embed_text)` for each chunk and checks against an existing `{doc_id}_meta.json` file:

- **Cache hit**: chunk is taken from the existing file, no model call.
- **Cache miss**: chunk is sent to `encode_batch()`.
- Merged results are written back atomically, preserving original chunk order.

To bypass the cache and force re-embedding:
```bash
FORCE_REEMBED=1 python main.py
```

---

## Configuration

All defaults are in `embedding/config.py`. Override via environment variables:

| Env Var | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HuggingFace model identifier |
| `EMBEDDING_BATCH_SIZE` | `12` | Chunks per `encode()` call |
| `FORCE_REEMBED` | `0` | Set to `1` to bypass cache |

> **CPU-only**: `device="cpu"` and `use_fp16=False` are hard-pinned in code and cannot be overridden via env var — this ensures reproducible, GPU-free behavior regardless of the machine.

---

## PyTorch / CPU Build Check

This module requires the **CPU-only** PyTorch wheel. Verify with:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected output: `2.x.x+cpu  False`

If `cuda.is_available()` returns `True`, you have the CUDA wheel installed. Replace it with the CPU wheel:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Running Tests

### Fast suite (mocked — no model download, runs in ~2 seconds)
```bash
python -m pytest tests/test_model_wrapper.py tests/test_cache.py tests/test_pipeline.py -v -m "not live_model"
```

### Optional live-model integration test (downloads BGE-M3, ~30-60 s)
```bash
RUN_LIVE_MODEL_TESTS=1 python -m pytest tests/test_pipeline.py -v -m live_model
```

The live test (`TestLiveModel`) verifies:
- Near-duplicate texts score higher dense-cosine-similarity than unrelated texts.
- A chunk with a distinctive rare token produces a non-trivial sparse weight.
- A single chunk runs through the full pipeline and produces a valid `EmbeddedChunk`.

---

## Acceptance Criteria Coverage

| # | Criterion | Test |
|---|---|---|
| 1 | Single chunk → `dim=1024`, `len(dense_vector)==1024` | `test_pipeline.py::test_single_chunk_produces_embedded_chunk_with_correct_dim` |
| 2 | Full cache hit → zero `encode_batch` calls | `test_cache.py::test_full_cache_hit_zero_encode_calls` |
| 3 | One chunk changed → only that chunk re-embedded | `test_cache.py::test_partial_cache_miss_only_changed_chunk_reencoded` |
| 4 | Identical `embed_text` → cosine sim > 0.99 | `test_pipeline.py::test_identical_embed_text_produces_similar_vectors` |
| 5 | N chunks → N outputs, `.npy` rows aligned with `meta.json` | `test_pipeline.py::test_n_chunks_produce_n_outputs_with_aligned_npy` |
| 6 | Table + text chunks both succeed | `test_pipeline.py::test_table_and_text_chunks_both_succeed` |
| 7 | Empty/whitespace `embed_text` → skipped with warning, no crash | `test_pipeline.py::test_empty_embed_text_skipped_with_warning` |
| 8 | Model loaded exactly once across multi-document run | `test_model_wrapper.py::test_get_model_loads_once` |
| 9 | Round-trip `.npy` + `meta.json` — no reordering or data loss | `test_cache.py::test_npy_row_order_matches_meta_order` |
| 10 | Live BGE-M3: near-dup > unrelated similarity; rare token sparse weight | `test_pipeline.py::TestLiveModel` (gated behind `RUN_LIVE_MODEL_TESTS=1`) |

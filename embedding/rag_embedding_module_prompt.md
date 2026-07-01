
# RAG embedding generation module — build specification (Stage 4)

## Role & objective
Build **Stage 4 only** of a 6-stage production RAG pipeline: **embedding generation**. Input is the list of `Chunk` objects produced by Stage 3 (chunking) for a document. Output is one `EmbeddedChunk` per input chunk, containing both a dense vector and a sparse lexical-weight vector, generated from a single self-hosted BGE-M3 model running on **CPU only** — no GPU/CUDA assumed anywhere in this build.

## Scope
**In scope:** loading and wrapping the BGE-M3 model, batch encoding, dense + sparse vector generation, content-hash-based caching so re-runs never re-embed unchanged chunks, persistent per-document output files, integration with the existing `main.py` CLI orchestrator.

**Out of scope — do not implement:** building a FAISS/vector-DB index, similarity search, retrieval, reranking, multi-vector/ColBERT output (explicitly skipped — costly and only useful for a future reranking stage, not here).

## Defaults chosen for this build (all configurable — change freely)
- **Model:** `BAAI/bge-m3` via the `FlagEmbedding` library, loaded with `use_fp16=False` and `device="cpu"` — hard-pinned regardless of what hardware the code happens to run on, so behavior is reproducible and never silently tries to grab a GPU.
- **Batch size:** 12 chunks per `encode()` call by default — small enough to keep CPU memory and latency reasonable, configurable down further for low-RAM machines.
- **Multi-vector/ColBERT output:** disabled (`return_colbert_vecs=False`). Only dense + sparse are generated.
- **Sparse representation:** BGE-M3's native lexical-weight output (`dict[token_id, weight]`), converted to `dict[str, float]` via the tokenizer's vocabulary so it's human-readable and storage-agnostic.

## What gets embedded
Always embed `chunk.embed_text` (the version with the title/heading-path context prefix already applied in Stage 3) — never `chunk.chunk_text`. Both the dense and sparse outputs come from the same single forward pass over this same input string; there is no separate "what to feed the sparse head" decision to make.

## Model wrapper — load once, not per call
The BGE-M3 model must be loaded exactly once per process (lazy singleton), never reloaded per chunk or per batch — load time for a 550M-parameter model is expensive and must not be paid repeatedly. Expose a single `encode_batch(texts: list[str]) -> list[tuple[dense_vec, sparse_weights]]` function that internally manages the singleton.

## Idempotent caching — non-negotiable given CPU speed
Re-embedding unchanged chunks on a CPU machine is expensive and must never happen on a re-run. Before encoding, compute `embed_text_hash = sha256(chunk.embed_text)`. For each document being processed:
1. If `{doc_id}_meta.json` already exists from a previous run, load it and build a `hash -> EmbeddedChunk` lookup.
2. Only chunks whose `embed_text_hash` is **not** already present get sent to the model.
3. Merge newly-computed embeddings with the previously-cached ones, preserving original chunk order, and rewrite the output files.

This reuses the per-document output files as the cache — no separate cache database is needed for this stage.

## Output contract
```python
class EmbeddedChunk(BaseModel):
    chunk_id: str
    dense_vector: list[float]              # 1024-dim
    sparse_vector: dict[str, float]        # token -> weight, non-zero entries only
    embedding_model: str                    # "BAAI/bge-m3"
    embedding_model_version: str            # pin a model revision/hash so future swaps are detectable
    embed_text_hash: str
    dim: int
    generated_at: datetime
```

## Storage format
Per document, write two files into `knowledge_bases/{kb_id}/embeddings/`:
- `{doc_id}_dense.npy` — a `numpy` array of shape `[n_chunks, 1024]`, row `i` corresponding to chunk `i` in the metadata file (same order, strictly aligned — this alignment must never drift).
- `{doc_id}_meta.json` — a JSON list of `EmbeddedChunk` objects in the same order as the `.npy` rows, with `sparse_vector` included directly (sparse weights are naturally small and JSON-friendly; do not put them in the `.npy` file).

## Validation built into this stage
After generating each batch, run lightweight sanity checks before persisting:
- `dim` is exactly 1024 and `len(dense_vector) == dim`.
- `sparse_vector` has at least one non-zero entry — flag (don't crash) any chunk that produces a fully empty sparse vector, since that usually signals degenerate input (e.g. a near-empty chunk slipping through from an earlier stage).
- Confirm whether the library's dense output is already L2-normalized; if not, normalize it before storage and document this clearly in the README, since downstream cosine-similarity search assumes normalized vectors.

## Robustness requirements
- Per-chunk isolation within a batch: if one chunk's text causes an encoding failure (e.g. unexpected type, encoding error), log it, skip it, and continue the batch rather than aborting the whole document.
- Structured logging (the `logging` module) with running progress: batch number, chunks processed, chunks skipped via cache, elapsed time, estimated time remaining — CPU encoding is slow enough that silent multi-minute waits are a real usability problem.
- Configuration via env vars or a settings object: model name, batch size, output directory, cache-bypass flag (force re-embed).

## Integration with the existing CLI
`main.py` currently writes placeholder/empty `.npy` and `.pkl` files after the chunking step (per the prior implementation plan). Replace that placeholder step for `embeddings.npy` with a real call into this module's pipeline function. Leave `faiss.index` and `metadata.pkl` as placeholders — building the actual vector index is the next stage (index & store) and is out of scope here.

## Testing requirement — do not load the real model in the main test suite
The BGE-M3 model is too large/slow to load in every unit test run. The test suite must:
- Mock/stub `encode_batch` (dependency injection or monkeypatching) for all pipeline/cache/schema tests, so the suite runs in seconds without downloading or running real model weights.
- Provide exactly **one** separate, explicitly-marked integration test (e.g. `@pytest.mark.live_model`, skipped by default unless an env var like `RUN_LIVE_MODEL_TESTS=1` is set) that loads the real model once and performs a minimal real sanity check — this is the only place actual model inference happens during testing.

## Project structure
```
embedding/
  schema.py
  model_wrapper.py     # BGE-M3 singleton loader + encode_batch()
  cache.py              # per-document hash-based cache load/merge logic
  pipeline.py           # embed_document(chunks, output_dir) -> list[EmbeddedChunk]
  config.py
  utils.py              # hashing, npy/json IO helpers
tests/
  fixtures/
  test_model_wrapper.py
  test_cache.py
  test_pipeline.py
```

## Acceptance criteria — implement and pass tests for all of these
1. A single chunk produces an `EmbeddedChunk` with `dim=1024`, `len(dense_vector)==1024`, and a non-empty `sparse_vector`.
2. Re-running on a document whose chunks are unchanged from a prior run results in zero calls to `encode_batch` (verify via a call-count mock) — full cache hit.
3. Changing one chunk's `embed_text` changes its hash and triggers re-embedding for only that chunk; sibling chunks in the same document stay cached.
4. Two chunks with identical `embed_text` produce dense vectors with cosine similarity > 0.99 (determinism check, using a mocked deterministic encoder for the unit test).
5. Batch processing of N chunks produces exactly N `EmbeddedChunk` outputs, with `.npy` row order strictly matching `meta.json` order.
6. A table chunk and a text chunk both flow through the same encode path successfully — no type-based branching breaks either.
7. An empty or whitespace-only `embed_text` (defensive edge case) is handled without crashing — skipped with a logged warning, not silently embedded as a zero vector without a flag.
8. The model wrapper is confirmed to load exactly once across an entire multi-document pipeline run (call-count/timing assertion), not once per chunk or batch.
9. `dense_vector` and `sparse_vector` round-trip through the `.npy` + `meta.json` files without data loss or reordering.
10. The live-model integration test (skipped by default) confirms real BGE-M3 output: two near-duplicate texts score higher dense-cosine-similarity than two unrelated texts, and a chunk containing a distinctive rare token produces a non-trivial sparse weight on that token.

## Stack & quality bar
- Python 3.11+, full type hints, Pydantic v2.
- `FlagEmbedding` (BGE-M3), `numpy`, stdlib `hashlib`/`json`/`logging`.
- Verify `torch` installs as the CPU build on this machine — no CUDA-specific install steps needed for this task, but worth a one-line check in the README so it's obvious if the wrong wheel ever gets pulled in.
- Docstrings on public functions; README explaining the schema, caching behavior, storage layout, and how to run both the fast mocked suite and the optional live-model test.

## Process — follow this order
1. Finalize `schema.py` first.
2. Implement `model_wrapper.py` (singleton + `encode_batch`), with the live-model test as the only place it's actually exercised against real weights.
3. Implement `cache.py` (load/merge logic against existing `{doc_id}_meta.json` files).
4. Implement `pipeline.py` wiring batching, caching, validation, and file output together.
5. Wire the `main.py` integration change.
6. Build fixtures and tests (mocked) for all 10 acceptance cases.
7. Output a short README.

"""BGE-M3 model singleton wrapper.

Loads the model exactly once per process (lazy singleton).  All callers go
through ``encode_batch()`` — never construct or reload the model directly.

CPU-only: ``device="cpu"`` and ``use_fp16=False`` are hard-pinned here;
no GPU path exists in this build.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state — module-level, never exposed directly
# ---------------------------------------------------------------------------
_model = None
_model_revision: str = ""


def get_model():
    """Return the BGE-M3 singleton, loading it on the first call only.

    Subsequent calls return the already-loaded instance immediately — the
    model is *never* reloaded, regardless of how many documents are processed.
    """
    global _model, _model_revision

    if _model is not None:
        return _model

    logger.info(
        "Loading BGE-M3 model '%s' on CPU (first load — this will take ~30-60 s)…",
        config.MODEL_NAME,
    )

    # Import here so tests can monkeypatch before the first real load.
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FlagEmbedding is not installed. Run: pip install FlagEmbedding"
        ) from exc

    # Silence the transformers "fast tokenizer" UserWarning and any INFO-level
    # noise from the transformers / huggingface_hub libraries.
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    # Silence "Fetching N files" progress bar from huggingface_hub.
    import os as _os
    _hf_prev = _os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        _model = BGEM3FlagModel(
            config.MODEL_NAME,
            use_fp16=config.USE_FP16,
            device=config.DEVICE,
        )
        import time
        time.sleep(1)
    finally:
        # Restore the env var (don't permanently pollute the process env).
        if _hf_prev is None:
            _os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            _os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = _hf_prev
    # Stash the revision/tag so EmbeddedChunk can record it.
    _model_revision = getattr(_model, "model_name", config.MODEL_NAME)
    logger.info("BGE-M3 model loaded successfully.")
    return _model


def get_model_version() -> str:
    """Return the model revision string (populated after the first ``get_model()`` call)."""
    return _model_revision or config.MODEL_NAME

def free_model() -> None:
    """Frees the BGE-M3 model from RAM to prevent Out-Of-Memory crashes on CPU."""
    global _model, _model_revision
    if _model is not None:
        logger.info("Freeing BGE-M3 model from memory...")
        _model = None
        import gc
        gc.collect()

# ---------------------------------------------------------------------------
# Public encode API
# ---------------------------------------------------------------------------

def encode_batch(
    texts: List[str],
) -> List[Tuple[List[float], Dict[str, float]]]:
    """Encode a batch of texts and return dense + sparse vectors per text.

    Args:
        texts: List of ``embed_text`` strings (already include title/heading prefix).

    Returns:
        A list aligned with *texts*: each element is
        ``(dense_vector: List[float], sparse_vector: Dict[str, float])``.

        * ``dense_vector`` is L2-normalized (1024-dim).
        * ``sparse_vector`` maps token strings to weights; only non-zero entries
          are included.
        * If an individual text fails to encode, ``None`` is returned at that
          position — the caller must handle ``None`` entries.

    Note on normalization:
        BGE-M3's ``encode()`` returns dense vectors that are **already
        L2-normalized** when ``normalize_embeddings=True`` (the default in
        FlagEmbedding).  We still run ``normalize_l2`` explicitly as a
        safety net and to make the contract testable.
    """
    from .utils import normalize_l2

    model = get_model()
    tokenizer = model.tokenizer

    # Single forward pass — dense + sparse; ColBERT disabled as per spec.
    # Silence FlagEmbedding's tqdm bars by disabling tqdm globally for this call.
    # (show_progress_bar=False cannot be passed as a kwarg — it leaks into the
    # tokenizer via **kwargs and causes a TypeError.)
    try:
        import tqdm as _tqdm_module
        _tqdm_cls = _tqdm_module.tqdm
        _orig_disabled = _tqdm_cls.disable
        _tqdm_cls.disable = True
    except Exception:
        _tqdm_cls = None
        _orig_disabled = None

    try:
        outputs = model.encode(
            texts,
            batch_size=len(texts),
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
    except Exception as exc:
        logger.error("encode() call failed for entire batch: %s", exc, exc_info=True)
        return [None] * len(texts)  # type: ignore[list-item]
    finally:
        # Always restore tqdm state — even if encode() raised.
        if _tqdm_cls is not None and _orig_disabled is not None:
            _tqdm_cls.disable = _orig_disabled

    dense_vecs = outputs.get("dense_vecs", [])
    lexical_weights = outputs.get("lexical_weights", [])

    results: List[Tuple[List[float], Dict[str, float]] | None] = []

    for idx, text in enumerate(texts):
        try:
            # Dense vector — L2-normalize (safety net; library already normalizes).
            raw_dense = dense_vecs[idx]
            dense: List[float] = normalize_l2(raw_dense.tolist())

            # Sparse vector — convert token_id→weight to token_str→weight.
            raw_sparse: Dict[int, float] = lexical_weights[idx]
            sparse: Dict[str, float] = {}
            for token_id, weight in raw_sparse.items():
                if weight == 0.0:
                    continue
                token_str = tokenizer.decode(
                    [int(token_id)], skip_special_tokens=True
                ).strip()
                if token_str:
                    sparse[token_str] = float(weight)

            results.append((dense, sparse))

        except Exception as exc:
            logger.error(
                "Failed to process encoding result for text[%d] (preview: %r): %s",
                idx,
                text[:80],
                exc,
                exc_info=True,
            )
            results.append(None)  # type: ignore[arg-type]

    return results  # type: ignore[return-value]


def reset_singleton_for_testing() -> None:
    """Reset the module-level singleton.  For use in tests only — never call in production."""
    global _model, _model_revision
    _model = None
    _model_revision = ""

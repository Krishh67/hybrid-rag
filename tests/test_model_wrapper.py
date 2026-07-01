"""Tests for embedding.model_wrapper — all mocked (no real model loads).

Acceptance criterion covered:
    #8 — the model wrapper loads exactly once across multiple calls (singleton guarantee).
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_model_wrapper():
    """Reimport model_wrapper with a clean singleton state."""
    if "embedding.model_wrapper" in sys.modules:
        del sys.modules["embedding.model_wrapper"]
    import embedding.model_wrapper as mw
    return mw


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleton:
    """Criterion #8: model loaded exactly once across the entire pipeline run."""

    def test_get_model_loads_once(self, monkeypatch):
        """get_model() called N times must invoke BGEM3FlagModel.__init__ exactly once."""
        mw = _reload_model_wrapper()

        mock_model_instance = MagicMock()
        mock_model_instance.model_name = "BAAI/bge-m3"
        mock_cls = MagicMock(return_value=mock_model_instance)

        monkeypatch.setattr("embedding.model_wrapper._model", None)
        monkeypatch.setattr("embedding.model_wrapper._model_revision", "")

        with patch.dict("sys.modules", {"FlagEmbedding": MagicMock(BGEM3FlagModel=mock_cls)}):
            # Re-import so the patched sys.modules is used
            mw = _reload_model_wrapper()

            m1 = mw.get_model()
            m2 = mw.get_model()
            m3 = mw.get_model()

        assert m1 is m2 is m3, "get_model() must return the same singleton object"
        assert mock_cls.call_count == 1, (
            f"BGEM3FlagModel was constructed {mock_cls.call_count} times — expected exactly 1"
        )

    def test_reset_singleton_clears_state(self, monkeypatch):
        """reset_singleton_for_testing() clears state so the next call loads fresh."""
        mw = _reload_model_wrapper()

        mock_model_instance = MagicMock()
        mock_model_instance.model_name = "BAAI/bge-m3"
        mock_cls = MagicMock(return_value=mock_model_instance)

        with patch.dict("sys.modules", {"FlagEmbedding": MagicMock(BGEM3FlagModel=mock_cls)}):
            mw = _reload_model_wrapper()
            mw.get_model()  # first load
            mw.reset_singleton_for_testing()
            mw.get_model()  # second load after reset

        assert mock_cls.call_count == 2, "After reset, get_model() should trigger a fresh load"


class TestEncBatch:
    """encode_batch correctly transforms model outputs into (dense, sparse) tuples."""

    def _make_mock_model(self, dense_dim: int = 1024):
        import numpy as np

        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.tokenizer.decode.side_effect = lambda ids, **kw: f"token_{ids[0]}"

        dense_vec = np.ones(dense_dim, dtype=np.float32)
        mock_model.encode.return_value = {
            "dense_vecs": [dense_vec, dense_vec],
            "lexical_weights": [
                {42: 0.9, 7: 0.3},
                {42: 0.8},
            ],
        }
        return mock_model

    def test_returns_tuple_per_text(self, monkeypatch):
        mw = _reload_model_wrapper()
        mock_model = self._make_mock_model()
        monkeypatch.setattr(mw, "_model", mock_model)
        monkeypatch.setattr(mw, "_model_revision", "test")

        results = mw.encode_batch(["text one", "text two"])

        assert len(results) == 2
        for r in results:
            assert r is not None
            dense, sparse = r
            assert len(dense) == 1024
            assert isinstance(sparse, dict)
            assert len(sparse) > 0

    def test_zero_weight_tokens_excluded(self, monkeypatch):
        import numpy as np

        mw = _reload_model_wrapper()
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.tokenizer.decode.side_effect = lambda ids, **kw: f"token_{ids[0]}"
        mock_model.encode.return_value = {
            "dense_vecs": [np.ones(1024, dtype=np.float32)],
            "lexical_weights": [{5: 0.0, 10: 0.5}],  # token 5 has zero weight
        }
        monkeypatch.setattr(mw, "_model", mock_model)
        monkeypatch.setattr(mw, "_model_revision", "test")

        results = mw.encode_batch(["test"])
        _, sparse = results[0]
        assert "token_5" not in sparse, "Zero-weight tokens must be excluded from sparse vector"
        assert "token_10" in sparse

    def test_encode_failure_returns_none_per_chunk(self, monkeypatch):
        """If encode() raises, all results for that batch become None (no crash)."""
        mw = _reload_model_wrapper()
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_model.encode.side_effect = RuntimeError("GPU OOM")
        monkeypatch.setattr(mw, "_model", mock_model)
        monkeypatch.setattr(mw, "_model_revision", "test")

        results = mw.encode_batch(["text"])
        assert results == [None]

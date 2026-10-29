"""Embedding backend.

The default provider is a free local model (fastembed, BAAI/bge-small-en-v1.5),
so the system needs no embeddings API key. ``ARAG_EMBEDDING_PROVIDER=openai``
selects OpenAI embeddings instead (optional; needs the ``openai`` package and
OPENAI_API_KEY).

Passages (chunks, sentences) and queries are embedded through separate paths
because BGE-style models are trained with distinct query/passage handling.
All returned vectors are L2-normalised (the FAISS indexes use inner product).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from arag.runtime_ops import get_global_budget, get_global_rate_limiter

log = logging.getLogger(__name__)

DEFAULT_PROVIDER = "local"
DEFAULT_MODELS = {"local": "BAAI/bge-small-en-v1.5", "openai": "text-embedding-3-small"}
# OpenAI does not expose model dimensions locally, so list the known ones.
_OPENAI_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}

_model_lock = threading.Lock()
_local_models: Dict[str, Any] = {}


def get_provider() -> str:
    provider = (os.environ.get("ARAG_EMBEDDING_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown ARAG_EMBEDDING_PROVIDER={provider!r}; expected one of {sorted(DEFAULT_MODELS)}"
        )
    return provider


def get_model_name(provider: Optional[str] = None) -> str:
    provider = provider or get_provider()
    return os.environ.get("ARAG_EMBEDDING_MODEL") or DEFAULT_MODELS[provider]


def get_embedding_info() -> Dict[str, Any]:
    """Return {provider, model, dim} for the configured embedding model.

    ``dim`` is read from the model (fastembed's registry, no download needed);
    it is None only for an OpenAI model not in the known list.
    """
    provider = get_provider()
    model = get_model_name(provider)
    dim: Optional[int]
    if provider == "local":
        from fastembed import TextEmbedding

        try:
            dim = int(TextEmbedding.get_embedding_size(model))
        except ValueError:  # custom model not in the registry: measure it
            dim = len(_embed_local(["dimension probe"], "passage", model)[0])
    else:
        dim = _OPENAI_DIMS.get(model)
    return {"provider": provider, "model": model, "dim": dim}


def _load_local_model(model: str) -> Any:
    with _model_lock:
        if model not in _local_models:
            from fastembed import TextEmbedding

            log.info("Loading local embedding model %s", model)
            _local_models[model] = TextEmbedding(model_name=model)
        return _local_models[model]


def warm_up() -> None:
    """Load the configured local model now (used by the Streamlit app)."""
    if get_provider() == "local":
        _load_local_model(get_model_name("local"))


def _embed_local(texts: List[str], kind: str, model: str) -> List[List[float]]:
    import numpy as np

    if not texts:
        return []
    embedder = _load_local_model(model)
    embed = embedder.query_embed if kind == "query" else embedder.passage_embed
    arr = np.array(list(embed(list(texts))), dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0, 1.0, norms)
    return arr.tolist()


def _embed_openai(texts: List[str], model: str) -> List[List[float]]:
    # Enforce the spend caps before either backend is tried, so the
    # llama-index path is limited too (BudgetExceededError must not be
    # swallowed by the fallback handling below).
    rl = get_global_rate_limiter()
    budget = get_global_budget()
    if not rl.acquire(timeout=5.0):
        raise RuntimeError("Rate limiter timeout while attempting to obtain embeddings")
    budget.consume_api_call(1)

    try:
        from llama_index.embeddings.openai import OpenAIEmbedding

        embed_model = OpenAIEmbedding(model=model)
        log.debug("Using llama-index OpenAIEmbedding for embeddings")
        raw = embed_model.get_text_embedding_batch(texts)
        return [list(r) for r in raw]
    except Exception:
        log.debug("llama-index OpenAIEmbedding not available; falling back")

    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError(
            "No embedding backend available. Install/enable llama-index or the openai package and set OPENAI_API_KEY."
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment; cannot obtain embeddings")

    try:
        from arag.runtime_ops import inc_api_call_metric

        inc_api_call_metric(1)
    except Exception:
        pass
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=texts)
    embs: List[List[float]] = []
    for d in resp.data:
        if hasattr(d, "embedding"):
            emb_val = d.embedding
        elif isinstance(d, dict) and "embedding" in d:
            emb_val = d["embedding"]
        else:
            emb_val = list(d)  # type: ignore[arg-type]
        embs.append(list(emb_val))

    approx_tokens = sum(max(1, len(t.split())) for t in texts)
    try:
        budget.consume_tokens(approx_tokens)
        try:
            from arag.runtime_ops import inc_token_metric

            inc_token_metric(approx_tokens)
        except Exception:
            pass
    except Exception as e:
        log.warning("Budget monitor raised: %s", e)
    return embs


def _embed(texts: List[str], kind: str, model: Optional[str]) -> List[List[float]]:
    provider = get_provider()
    name = model or get_model_name(provider)
    if provider == "local":
        return _embed_local(texts, kind, name)
    return _embed_openai(texts, name)


def get_embeddings(texts: List[str], model: Optional[str] = None, kind: str = "passage") -> List[List[float]]:
    """Embed many texts. ``kind`` is "passage" (default) or "query"."""
    return _embed(texts, kind, model)


def get_embedding(text: str, model: Optional[str] = None, kind: str = "query") -> List[float]:
    """Embed one text. Defaults to the query path (a user question)."""
    return list(_embed([text], kind, model)[0])

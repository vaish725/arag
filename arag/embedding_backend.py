"""Embedding backend moved into arag package."""

from __future__ import annotations

import logging
import os
from typing import List

from arag.runtime_ops import get_global_budget, get_global_rate_limiter

log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
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

    rl = get_global_rate_limiter()
    budget = get_global_budget()

    if not rl.acquire(timeout=5.0):
        raise RuntimeError("Rate limiter timeout while attempting to obtain embeddings")
    budget.consume_api_call(1)
    try:
        from arag.runtime_ops import inc_api_call_metric, inc_token_metric

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


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    return list(get_embeddings([text], model=model)[0])

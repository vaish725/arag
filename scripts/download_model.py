#!/usr/bin/env python3
"""Download the local embedding model into the fastembed cache.

Needs network access to huggingface.co once. Afterwards everything (including
the test suite, run with HF_HUB_OFFLINE=1) works offline. Set
FASTEMBED_CACHE_PATH to choose where the model is stored.
"""
from arag.embedding_backend import get_embedding_info, get_embeddings, warm_up

if __name__ == "__main__":
    warm_up()
    vec = get_embeddings(["smoke test"])[0]
    info = get_embedding_info()
    assert len(vec) == info["dim"], (len(vec), info)
    print(f"OK: {info['provider']}/{info['model']} ready, dimension {info['dim']}")

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so tests can import local modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Import retrieval helpers inside tests after adjusting sys.path to avoid
# module-level imports that come after executable code (flake8 E402).


def test_bm25_non_empty():
    """BM25 should return at least one result for a simple query if index exists."""
    from arag import retrieval_tools
    from arag.retrieval_tools import keyword_search, load_indexes

    try:
        load_indexes()
    except FileNotFoundError:
        pytest.skip("BM25/chunks not built (bm25_index.pkl or chunks.pkl missing)")

    res = keyword_search("Scott Derrickson director", k=3)
    assert isinstance(res, list)
    assert len(res) > 0, "BM25 returned no results for a common query"


@pytest.mark.needs_model  # the index guard verifies the real model's files before searching
def test_semantic_search_when_faiss_exists(monkeypatch, tmp_path):
    """Semantic search should return results when FAISS index and
    embedding backend are available.

    This test skips if the FAISS index file is missing or if the
    embedding backend raises a RuntimeError (e.g., OPENAI_API_KEY not set).
    It asserts non-empty results otherwise.
    """
    # Ensure indexes are loaded (will set up internal handles)
    from arag.retrieval_tools import load_indexes, semantic_search

    try:
        load_indexes()
    except FileNotFoundError:
        pytest.skip("Indexes not built; skipping semantic search test")

    faiss_path = os.path.join("data", "faiss_index.bin")
    if not os.path.exists(faiss_path):
        pytest.skip("FAISS index not present; skip semantic search test")

    # Indexes are loaded; keep the query-embedding cache out of the repo.
    import arag.retrieval_tools as rt

    monkeypatch.setattr(rt, "DATA_DIR", str(tmp_path))

    # Use a fixed query vector of the index's dimension so the real FAISS
    # search runs with no model and no network.
    import numpy as np

    import arag.embedding_backend as eb
    import arag.retrieval_tools as rt_mod

    dim = rt_mod._sentence_faiss_index.d if rt_mod._sentence_faiss_index is not None else rt_mod._faiss_index.d
    vec = np.random.default_rng(0).normal(size=dim).tolist()
    monkeypatch.setattr(eb, "get_embedding", lambda text, model=None, kind="query": vec)
    monkeypatch.setattr(eb, "get_embeddings", lambda texts, model=None, kind="passage": [vec for _ in texts])

    res = semantic_search("Who directed Doctor Strange?", k=3)

    assert isinstance(res, list)
    assert (
        len(res) > 0
    ), "Semantic search returned no results despite FAISS index being present"

"""Exercise ``retrieval_tools.semantic_search`` without any API key.

The committed ``data/`` indexes hold real OpenAI vectors, so a fake query
embedding cannot be compared against them. Instead these tests build a tiny
index in ``tmp_path`` using the real builder functions, with a deterministic
hashed bag-of-words embedding patched in for ``arag.embedding_backend``. That
runs the real FAISS / BM25 / re-rank code in CI with zero network access.
"""

import hashlib
import re

import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

import numpy as np  # noqa: E402

DIM = 64

DOCS = [
    {
        "title": "Lighthouse Keeper",
        "text": (
            "Lighthouse Keeper is a drama about a lonely lighthouse on a rocky "
            "northern coast. The lighthouse guides ships through dangerous fog "
            "every single night."
        ),
    },
    {
        "title": "Desert Caravan",
        "text": (
            "Desert Caravan follows merchants crossing a vast sand desert. "
            "The caravan carries spices and silk toward a distant market town."
        ),
    },
    {
        "title": "Orbital Station",
        "text": (
            "Orbital Station is a thriller set aboard a failing space station. "
            "The crew races to repair the damaged oxygen system before dawn."
        ),
    },
]


def fake_embed(text):
    """Deterministic hashed bag-of-words vector (shared words => similar)."""
    vec = np.zeros(DIM, dtype="float32")
    for tok in re.findall(r"[a-z]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    return vec.tolist()


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    """Build chunk + sentence indexes with fake embeddings and load them."""
    import arag.embedding_backend as eb
    import arag.hierarchical_index_builder as hib
    import arag.retrieval_tools as rt

    calls = {"n": 0}

    def fake_get_embeddings(texts, model="text-embedding-3-small"):
        calls["n"] += len(texts)
        return [fake_embed(t) for t in texts]

    def fake_get_embedding(text, model="text-embedding-3-small"):
        calls["n"] += 1
        return fake_embed(text)

    monkeypatch.setattr(eb, "get_embeddings", fake_get_embeddings)
    monkeypatch.setattr(eb, "get_embedding", fake_get_embedding)
    # Relative paths inside retrieval_tools (sentence cache) must not touch data/.
    monkeypatch.chdir(tmp_path)

    chunks = hib.build_chunks(DOCS)
    paths = {
        "bm25_path": str(tmp_path / "bm25_index.pkl"),
        "chunks_path": str(tmp_path / "chunks.pkl"),
        "faiss_path": str(tmp_path / "faiss_index.bin"),
        "sentence_faiss_path": str(tmp_path / "sentence_faiss_index.bin"),
        "sentences_path": str(tmp_path / "sentences.pkl"),
    }
    hib.build_bm25(chunks, out_path=paths["bm25_path"])
    hib.build_faiss(chunks, out_index=paths["faiss_path"], out_chunks=paths["chunks_path"])
    hib.build_sentence_faiss(
        chunks, out_index=paths["sentence_faiss_path"], out_sentences=paths["sentences_path"]
    )

    # Register globals with monkeypatch so they are restored after the test.
    for name in ("_bm25", "_chunks", "_faiss_index", "_sentence_faiss_index", "_sentences"):
        monkeypatch.setattr(rt, name, None)
    rt.load_indexes(**paths)
    assert rt._sentence_faiss_index is not None and rt._faiss_index is not None
    calls["n"] = 0  # ignore build-time embeddings
    return rt, calls


def test_sentence_faiss_path_returns_relevant_sentence(indexed):
    rt, calls = indexed
    res = rt.semantic_search("lighthouse on a rocky coast guides ships through fog", k=3)
    assert res, "sentence-level FAISS search returned nothing"
    assert res[0]["chunk_id"].startswith("Lighthouse_Keeper")
    assert res[0]["faiss_score"] > 0
    # bm25_score is 0.0 only on the sentence-FAISS branch (not the re-rank fallback)
    assert res[0]["bm25_score"] == 0.0
    assert calls["n"] >= 1, "query embedding should have been requested"


def test_title_hint_short_circuits_without_embedding(indexed):
    rt, calls = indexed
    res = rt.semantic_search('Who directed "Desert Caravan" and did they win?', k=3)
    assert res and all(r["chunk_id"].startswith("Desert_Caravan") for r in res)
    assert calls["n"] == 0, "title-hinted queries must not call the embedding API"


def test_bm25_dense_rerank_path(indexed, monkeypatch):
    rt, _ = indexed
    # Disable sentence index so the BM25 -> dense re-rank branch runs.
    monkeypatch.setattr(rt, "_sentence_faiss_index", None)
    monkeypatch.setattr(rt, "_sentences", None)
    res = rt.semantic_search("crew repairs the oxygen system on the space station", k=3)
    assert res
    assert res[0]["chunk_id"].startswith("Orbital_Station")


def test_chunk_level_faiss_fallback_path(indexed, monkeypatch):
    rt, _ = indexed
    # No sentence index and no BM25 -> global chunk FAISS fallback branch.
    monkeypatch.setattr(rt, "_sentence_faiss_index", None)
    monkeypatch.setattr(rt, "_sentences", None)
    monkeypatch.setattr(rt, "_bm25", None)
    res = rt.semantic_search("merchants crossing the sand desert with spices", k=3)
    assert res
    assert res[0]["chunk_id"].startswith("Desert_Caravan")

"""Local embedding provider, tested with a fake fastembed (no model download)."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

DIM = 8


class FakeTextEmbedding:
    instances = 0
    calls = []

    def __init__(self, model_name=None, **kwargs):
        FakeTextEmbedding.instances += 1
        self.model_name = model_name

    @staticmethod
    def get_embedding_size(model_name):
        return DIM

    def _vec(self, text, offset):
        v = np.zeros(DIM, dtype="float32")
        v[len(text) % DIM] = 3.0  # deliberately NOT unit length
        v[(len(text) + offset) % DIM] += 4.0
        return v

    def query_embed(self, texts, **kw):
        FakeTextEmbedding.calls.append("query")
        return iter([self._vec(t, 1) for t in texts])

    def passage_embed(self, texts, **kw):
        FakeTextEmbedding.calls.append("passage")
        return iter([self._vec(t, 2) for t in texts])


@pytest.fixture
def eb(monkeypatch):
    import arag.embedding_backend as eb

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.delenv("ARAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("ARAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(eb, "_local_models", {})
    FakeTextEmbedding.instances = 0
    FakeTextEmbedding.calls = []
    return eb


def test_default_provider_is_local_bge_small(eb):
    info = eb.get_embedding_info()
    assert info == {"provider": "local", "model": "BAAI/bge-small-en-v1.5", "dim": DIM}


def test_vectors_have_model_dimension_and_unit_norm(eb):
    vecs = eb.get_embeddings(["alpha", "beta gamma", "x"])
    arr = np.array(vecs)
    assert arr.shape == (3, DIM)
    assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-5)
    q = np.array(eb.get_embedding("who directed it"))
    assert q.shape == (DIM,) and abs(np.linalg.norm(q) - 1.0) < 1e-5


def test_query_and_passage_use_different_paths(eb):
    eb.get_embedding("a question")  # default: query
    eb.get_embeddings(["a chunk", "another chunk"])  # default: passage
    eb.get_embedding("forced passage", kind="passage")
    assert FakeTextEmbedding.calls == ["query", "passage", "passage"]


def test_model_loaded_once_per_process(eb):
    eb.get_embedding("a")
    eb.get_embeddings(["b"])
    eb.warm_up()
    assert FakeTextEmbedding.instances == 1


def test_local_provider_spends_no_budget_and_needs_no_key(eb, monkeypatch):
    from arag import runtime_ops

    monkeypatch.setenv("ARAG_MAX_API_CALLS_PER_DAY", "0")  # any API call would raise
    runtime_ops.reset_global_budget()
    assert len(eb.get_embeddings(["x"])) == 1


def test_unknown_provider_is_rejected(eb, monkeypatch):
    monkeypatch.setenv("ARAG_EMBEDDING_PROVIDER", "carrier-pigeon")
    with pytest.raises(ValueError, match="ARAG_EMBEDDING_PROVIDER"):
        eb.get_embedding_info()


def test_openai_provider_is_opt_in_and_reports_its_dimension(eb, monkeypatch):
    monkeypatch.setenv("ARAG_EMBEDDING_PROVIDER", "openai")
    assert eb.get_embedding_info() == {"provider": "openai", "model": "text-embedding-3-small", "dim": 1536}


def test_semantic_search_works_with_no_openai_key(eb, tmp_path, monkeypatch):
    """End to end on the default (local) provider with both keys unset: build a
    small index, search it, and make sure nothing mentions OPENAI_API_KEY."""
    pytest.importorskip("faiss")
    pytest.importorskip("rank_bm25")
    import test_semantic_search as tss
    from arag import agent_loop
    from arag import hierarchical_index_builder as hib
    from arag import retrieval_tools as rt

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rt, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(agent_loop, "DATA_DIR", str(tmp_path))
    chunks = hib.build_chunks(tss.DOCS)
    paths = dict(
        bm25_path=str(tmp_path / "bm25_index.pkl"),
        chunks_path=str(tmp_path / "chunks.pkl"),
        faiss_path=str(tmp_path / "faiss_index.bin"),
        sentence_faiss_path=str(tmp_path / "sentence_faiss_index.bin"),
        sentences_path=str(tmp_path / "sentences.pkl"),
    )
    hib.build_bm25(chunks, out_path=paths["bm25_path"])
    hib.build_faiss(chunks, out_index=paths["faiss_path"], out_chunks=paths["chunks_path"])
    hib.build_sentence_faiss(chunks, out_index=paths["sentence_faiss_path"], out_sentences=paths["sentences_path"])
    for name in ("_bm25", "_chunks", "_faiss_index", "_sentence_faiss_index", "_sentences"):
        monkeypatch.setattr(rt, name, None)
    rt.load_indexes(**paths)

    assert rt.semantic_search("lighthouse on a rocky coast guides ships through fog", k=3)
    res = agent_loop.run_agent(
        'Who directed "Lighthouse Keeper" and did they win any Academy Awards?',
        allow_fallback=False,
        enable_director_flow=True,
    )
    assert "OPENAI_API_KEY" not in str(res["tool_trace"])
    assert "Failed to obtain embeddings" not in str(res["tool_trace"])


def test_reembed_script_changes_only_the_vector_indexes(eb, tmp_path, monkeypatch):
    """scripts/reembed_indexes.py: new FAISS indexes + meta, source files byte-identical."""
    pytest.importorskip("faiss")
    pytest.importorskip("rank_bm25")
    import hashlib
    import importlib.util
    import pickle
    from pathlib import Path

    import faiss
    import test_semantic_search as tss
    from arag import hierarchical_index_builder as hib
    from arag.index_meta import read_meta

    spec = importlib.util.spec_from_file_location(
        "reembed_indexes", Path(__file__).resolve().parents[1] / "scripts" / "reembed_indexes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    chunks = hib.build_chunks(tss.DOCS)
    chunks[0]["source"] = "Inception"  # pretend one of the 8 added articles is present
    hib.build_bm25(chunks, out_path=str(tmp_path / "bm25_index.pkl"))
    (tmp_path / "chunks.pkl").write_bytes(pickle.dumps(chunks))
    (tmp_path / "sentences.pkl").write_bytes(pickle.dumps(hib.extract_sentences(chunks)))
    stale = tmp_path / "sentences.pkl.embcache.pkl"
    stale.write_bytes(b"old openai vectors")
    sha = lambda n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()  # noqa: E731
    before = {n: sha(n) for n in mod.KEEP_UNCHANGED}

    assert mod.main(str(tmp_path)) == 0

    assert {n: sha(n) for n in mod.KEEP_UNCHANGED} == before
    assert not stale.exists()
    assert faiss.read_index(str(tmp_path / "faiss_index.bin")).ntotal == len(chunks)
    assert faiss.read_index(str(tmp_path / "sentence_faiss_index.bin")).d == DIM
    meta = read_meta(str(tmp_path))
    assert meta["dimension"] == DIM and meta["num_chunks"] == len(chunks)
    assert meta["wikipedia_revisions"] == {"Inception": mod.UNKNOWN_REVISION}

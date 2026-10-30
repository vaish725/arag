"""Model revision + file hashes in index_meta.json, and the failed-download path."""

import hashlib
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("faiss")
pytest.importorskip("rank_bm25")

import test_semantic_search as tss  # noqa: E402
from arag.index_meta import IndexMismatchError, read_meta  # noqa: E402

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _make_model_dir(tmp_path):
    d = tmp_path / "hub" / "models--x--y" / "snapshots" / REVISION
    d.mkdir(parents=True)
    (d / "model_optimized.onnx").write_bytes(b"onnx weights")
    (d / "tokenizer.json").write_bytes(b"{}")
    return d


class FakeModel:
    """Stand-in for fastembed.TextEmbedding that exposes a model dir like the real one."""

    model_dir = None
    fail = False

    def __init__(self, model_name=None, **kw):
        if FakeModel.fail:
            raise ValueError(f"Could not load model {model_name} from any source.")
        self.model = SimpleNamespace(_model_dir=FakeModel.model_dir)

    @staticmethod
    def get_embedding_size(name):
        return tss.DIM

    def _enc(self, texts):
        import numpy as np

        return iter([np.array(tss.fake_embed(t), dtype="float32") for t in texts])

    def query_embed(self, texts, **kw):
        return self._enc(texts)

    def passage_embed(self, texts, **kw):
        return self._enc(texts)


@pytest.fixture
def local_world(tmp_path, monkeypatch):
    """A tiny index built with the 'local' provider using FakeModel."""
    import arag.embedding_backend as eb
    from arag import hierarchical_index_builder as hib
    from arag import retrieval_tools as rt

    FakeModel.model_dir = _make_model_dir(tmp_path)
    FakeModel.fail = False
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeModel))
    monkeypatch.delenv("ARAG_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.setattr(eb, "_local_models", {})
    monkeypatch.setattr(eb, "_fingerprints", {})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rt, "DATA_DIR", str(tmp_path))
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
    return rt, paths, tmp_path


def test_meta_records_revision_and_file_hashes(local_world):
    _, _, tmp_path = local_world
    meta = read_meta(str(tmp_path))
    assert meta["model_revision"] == REVISION
    assert meta["model_files"]["model_optimized.onnx"] == hashlib.sha256(b"onnx weights").hexdigest()
    assert set(meta["model_files"]) == {"model_optimized.onnx", "tokenizer.json"}


def test_matching_model_files_pass(local_world):
    rt, _, _ = local_world
    assert rt.semantic_search("lighthouse on a rocky coast", k=2)


def test_changed_model_file_is_detected(local_world):
    import arag.embedding_backend as eb

    rt, _, _ = local_world
    (FakeModel.model_dir / "model_optimized.onnx").write_bytes(b"tampered weights")
    eb._fingerprints.clear()
    with pytest.raises(IndexMismatchError, match="sha256 mismatch: model_optimized.onnx"):
        rt.semantic_search("lighthouse", k=2)


def test_different_model_revision_is_detected(local_world, tmp_path):
    import shutil

    import arag.embedding_backend as eb

    rt, _, _ = local_world
    other = FakeModel.model_dir.parent / "ffffffffffffffffffffffffffffffffffffffff"
    shutil.copytree(FakeModel.model_dir, other)
    FakeModel.model_dir = other
    eb._local_models.clear()
    eb._fingerprints.clear()
    with pytest.raises(IndexMismatchError, match="revision differs"):
        rt.semantic_search("lighthouse", k=2)


def test_failed_model_download_disables_vector_search_but_not_bm25(local_world, caplog, monkeypatch):
    import arag.embedding_backend as eb
    from arag import agent_loop

    rt, paths, tmp_path = local_world
    FakeModel.fail = True  # model can't be downloaded / loaded any more
    eb._local_models.clear()
    eb._fingerprints.clear()

    with caplog.at_level("ERROR"):
        rt.load_indexes(**paths)  # must not raise
    assert "could not be loaded" in caplog.text and "Vector search disabled" in caplog.text
    assert rt.keyword_search("lighthouse keeper", k=2), "BM25 must keep working"
    with pytest.raises(IndexMismatchError, match="could not be loaded"):
        rt.semantic_search("lighthouse", k=2)

    monkeypatch.setattr(agent_loop, "DATA_DIR", str(tmp_path))
    res = agent_loop.run_agent(
        'Who directed "Lighthouse Keeper" and did they win any Academy Awards?',
        allow_fallback=False,
        enable_director_flow=True,
    )
    assert "could not be loaded" in str(res["tool_trace"])
    keyword = [t for t in res["tool_trace"] if t.get("tool") == "keyword_search"]
    assert keyword and not any("error" in t for t in keyword)

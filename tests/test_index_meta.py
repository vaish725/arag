"""index_meta.json and the embedding-mismatch guard."""

import json

import pytest

pytest.importorskip("faiss")

from arag.index_meta import IndexMismatchError, check_compat, read_meta  # noqa: E402

import test_semantic_search as tss  # noqa: E402  (reuse its tiny fake-embedding index)

indexed = tss.indexed  # fixture


def test_builders_write_meta(indexed, tmp_path):
    meta = read_meta(str(tmp_path))
    assert meta["provider"] == "fake" and meta["model"] == "hashed-bow"
    assert meta["dimension"] == tss.DIM
    assert meta["num_chunks"] == len(indexed[0]._chunks)
    assert meta["num_sentences"] == len(indexed[0]._sentences)
    assert meta["build_date"]


def test_matching_config_loads_and_searches(indexed):
    rt, _ = indexed
    assert rt.semantic_search("lighthouse on a rocky coast", k=2)


def test_different_model_raises_naming_both(indexed, monkeypatch):
    import arag.embedding_backend as eb

    rt, _ = indexed
    monkeypatch.setattr(eb, "get_embedding_info", lambda: {"provider": "local", "model": "BAAI/bge-small-en-v1.5", "dim": tss.DIM})
    with pytest.raises(IndexMismatchError) as e:
        rt.semantic_search("anything at all", k=2)
    msg = str(e.value)
    assert "fake/hashed-bow" in msg and "local/BAAI/bge-small-en-v1.5" in msg


def test_different_dimension_raises(indexed, monkeypatch):
    import arag.embedding_backend as eb

    rt, _ = indexed
    monkeypatch.setattr(eb, "get_embedding_info", lambda: {"provider": "fake", "model": "hashed-bow", "dim": 384})
    with pytest.raises(IndexMismatchError, match="dim"):
        rt.semantic_search("anything at all", k=2)


def test_missing_meta_raises(indexed, tmp_path):
    rt, _ = indexed
    (tmp_path / "index_meta.json").unlink()
    with pytest.raises(IndexMismatchError, match="index_meta.json"):
        rt.semantic_search("anything at all", k=2)


def test_stale_vector_count_raises(indexed, tmp_path):
    rt, _ = indexed
    path = tmp_path / "index_meta.json"
    meta = json.loads(path.read_text())
    meta["num_chunks"] += 5
    path.write_text(json.dumps(meta))
    with pytest.raises(IndexMismatchError, match="stale"):
        rt.semantic_search("anything at all", k=2)


def _paths(tmp_path):
    return dict(
        bm25_path=str(tmp_path / "bm25_index.pkl"),
        chunks_path=str(tmp_path / "chunks.pkl"),
        faiss_path=str(tmp_path / "faiss_index.bin"),
        sentence_faiss_path=str(tmp_path / "sentence_faiss_index.bin"),
        sentences_path=str(tmp_path / "sentences.pkl"),
    )


def test_mismatch_at_load_disables_vector_search_only(indexed, monkeypatch, tmp_path, caplog):
    import arag.embedding_backend as eb

    rt, _ = indexed
    monkeypatch.setattr(eb, "get_embedding_info", lambda: {"provider": "local", "model": "m", "dim": 384})
    with caplog.at_level("ERROR"):
        rt.load_indexes(**_paths(tmp_path))  # must not raise
    assert "Vector search disabled" in caplog.text and "fake/hashed-bow" in caplog.text
    assert rt.keyword_search("lighthouse keeper", k=2), "BM25 must keep working"
    with pytest.raises(IndexMismatchError):
        rt.semantic_search("lighthouse", k=2)


def test_mismatch_shows_in_tool_trace_while_keyword_search_still_answers(indexed, monkeypatch):
    import arag.embedding_backend as eb
    from arag import agent_loop

    rt, _ = indexed
    monkeypatch.setattr(eb, "get_embedding_info", lambda: {"provider": "local", "model": "m", "dim": 384})
    monkeypatch.setattr(agent_loop, "DATA_DIR", rt.DATA_DIR)
    res = agent_loop.run_agent(
        'Who directed "Lighthouse Keeper" and did they win any Academy Awards?',
        allow_fallback=False,
        enable_director_flow=True,
    )
    trace = str(res["tool_trace"])
    assert "Embedding mismatch" in trace, "the specific error must appear in tool_trace"
    keyword = [t for t in res["tool_trace"] if t.get("tool") == "keyword_search"]
    assert keyword and not any("error" in t for t in keyword), "keyword search must keep running"


def test_check_compat_ignores_missing_indexes(tmp_path):
    check_compat(str(tmp_path), {"provider": "x", "model": "y", "dim": 1})  # no index loaded: nothing to compare


def test_rebuilding_one_index_with_a_new_model_marks_the_other_stale(tmp_path):
    from arag.index_meta import record_build

    record_build(str(tmp_path), {"provider": "a", "model": "m1"}, 4, num_chunks=10, num_sentences=20)
    record_build(str(tmp_path), {"provider": "a", "model": "m2"}, 6, num_chunks=10)
    meta = read_meta(str(tmp_path))
    assert meta["model"] == "m2" and meta["num_chunks"] == 10 and meta["num_sentences"] is None

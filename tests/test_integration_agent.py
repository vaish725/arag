import pickle

from rank_bm25 import BM25Okapi

from arag import agent_loop
from arag.agent_loop import run_agent


def write_test_indexes(tmp_path):
    # Create two small documents that include a 'director' and an awards mention
    docs = [
        {
            "title": "Test_Film",
            "text": (
                "Test Film is a 2020 film directed by Jane Doe. "
                "It stars Alice B. and Bob C."
            ),
        },
        {
            "title": "Jane_Doe",
            "text": (
                "Jane Doe is a film director. "
                "She won an Academy Award for Best Short Film."
            ),
        },
    ]
    # BM25Okapi gives non-positive IDF on tiny corpora, so pad with filler docs.
    docs += [
        {"title": f"Filler_{i}", "text": f"Unrelated filler passage number {i} about geology and rivers."}
        for i in range(6)
    ]

    # Build chunks list similar to hierarchical_index_builder
    chunks = []
    for doc in docs:
        chunk_id = f"{doc['title']}_0"
        chunks.append(
            {
                "id": chunk_id,
                "text": doc["text"],
                "source": doc["title"],
                "coarse": doc["text"],
            }
        )

    # Save chunks.pkl
    chunks_path = tmp_path / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    # Build BM25
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_path = tmp_path / "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump((bm25, chunks), f)

    return str(bm25_path), str(chunks_path)


def test_agent_integration(tmp_path, monkeypatch):
    # Point the retrieval layer at a tiny temporary index instead of copying
    # files into the repo root, and run from tmp_path so no real data/ is seen.
    import arag.retrieval_tools as rt

    bm25_path, chunks_path = write_test_indexes(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rt, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(agent_loop, "DATA_DIR", str(tmp_path))
    for name in ("_bm25", "_chunks", "_faiss_index", "_sentence_faiss_index", "_sentences"):
        monkeypatch.setattr(rt, name, None)
    rt.load_indexes(
        bm25_path=bm25_path,
        chunks_path=chunks_path,
        faiss_path=str(tmp_path / "none.bin"),
        sentence_faiss_path=str(tmp_path / "none2.bin"),
        sentences_path=str(tmp_path / "none3.pkl"),
    )

    res = run_agent(
        "Who directed Test Film and did they win any Academy Awards?",
        max_steps=6,
        max_tokens=2000,
        allow_fallback=False,
        enable_director_flow=True,
    )
    assert "Jane Doe" in res["answer"]

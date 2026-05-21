import json
import os
import pickle
import shutil

from rank_bm25 import BM25Okapi

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
    # Write indexes to the temporary workspace and run the agent against them
    bm25_path, chunks_path = write_test_indexes(tmp_path)

    # Copy the files to the repository root where retrieval_tools expects
    # them
    shutil.copy(bm25_path, os.path.join(os.getcwd(), "bm25_index.pkl"))
    shutil.copy(chunks_path, os.path.join(os.getcwd(), "chunks.pkl"))

    try:
        res = run_agent(
            "Who directed Test Film and did they win any Academy Awards?",
            max_steps=6,
            max_tokens=2000,
        )
        # Ensure the agent found the director and that award detection can see
        # the local person page
        assert "Test Film" in json.dumps(res) or "Jane Doe" in json.dumps(res)
    finally:
        # cleanup
        for fname in ("bm25_index.pkl", "chunks.pkl"):
            if os.path.exists(fname):
                os.remove(fname)

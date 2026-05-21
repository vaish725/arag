"""Retrieval tools moved into arag package."""

from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

_bm25: Optional[Any] = None
_chunks: Optional[List[Dict[str, Any]]] = None
_faiss_index: Optional[Any] = None

GetEmbeddingsFn = Callable[[List[str], str], List[List[float]]]


def load_indexes(
    bm25_path: str = os.path.join("data", "bm25_index.pkl"),
    chunks_path: str = os.path.join("data", "chunks.pkl"),
    faiss_path: str = os.path.join("data", "faiss_index.bin"),
) -> None:
    global _bm25, _chunks, _faiss_index

    if os.path.exists(bm25_path):
        try:
            with open(bm25_path, "rb") as f:
                _bm25, _chunks = pickle.load(f)
            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    _chunks = pickle.load(f)
            print(f"Loaded BM25 index from {bm25_path} with {len(_chunks)} chunks")
        except Exception as e:
            raise RuntimeError(f"Failed to load BM25 index: {e}")
    elif os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            _chunks = pickle.load(f)
        print(f"Loaded chunks metadata from {chunks_path} (no BM25 index found)")
    else:
        raise FileNotFoundError(
            "No index files found. Run hierarchical_index_builder.py to build indexes first."
        )

    if os.path.exists(faiss_path):
        try:
            import faiss

            _faiss_index = faiss.read_index(faiss_path)
            print(f"Loaded FAISS index from {faiss_path}")
        except Exception as e:
            print(
                f"FAISS index found but failed to load: {e}. Semantic search will be unavailable."
            )
            _faiss_index = None
    else:
        _faiss_index = None


def keyword_search(query: str, k: int = 5) -> List[Dict]:
    global _bm25, _chunks
    if _bm25 is None or _chunks is None:
        load_indexes()
    if _bm25 is None:
        raise RuntimeError("BM25 index is not loaded")
    if _chunks is None:
        raise RuntimeError("Chunks metadata is not loaded")
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_k = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
    results = []
    for i, s in top_k:
        if s <= 0:
            continue
        c = _chunks[i]
        results.append({"chunk_id": c["id"], "text": c["text"][:400], "score": float(s)})
    return results


def semantic_search(query: str, k: int = 5) -> List[Dict]:
    global _faiss_index, _chunks
    if _chunks is None:
        load_indexes()
    if _chunks is None:
        raise RuntimeError("Chunks metadata is not loaded")
    assert _chunks is not None
    if _faiss_index is None:
        raise RuntimeError(
            "FAISS index not available. Build with --build_faiss or run hierarchical_index_builder.py with embeddings configured."
        )
    assert _faiss_index is not None
    try:
        from arag.embedding_backend import get_embedding

        q_vec_raw = get_embedding(query, model="text-embedding-3-small")
    except Exception as e:
        raise RuntimeError(f"Failed to obtain embeddings for semantic search: {e}")

    try:
        import faiss
        import numpy as np

        q_vec_arr = np.array(q_vec_raw).astype("float32").reshape(1, -1)
        faiss.normalize_L2(q_vec_arr)
        # Run the FAISS search inside a try/except to catch backend errors
        try:
            scores, indices = _faiss_index.search(q_vec_arr, k)
        except Exception as e:
            raise RuntimeError(f"FAISS search failed: {e}")

        # Normalize numpy arrays for consistent indexing
        scores_arr = np.array(scores)
        indices_arr = np.array(indices)
    except Exception as e:
        # Any import/np errors are surfaced as runtime errors
        raise RuntimeError(f"Error preparing FAISS search: {e}")

    out: List[Dict[str, Any]] = []

    try:
        from arag.embedding_backend import get_embeddings as _ge

        get_embeddings: Optional[GetEmbeddingsFn] = _ge  # type: ignore[assignment]
    except Exception:
        get_embeddings = None

    max_chunks_with_sentence_snippets = 2
    max_sentences_per_chunk = 3

    # Ensure indices_arr is 2-D: (n_queries, k)
    if indices_arr.ndim == 1:
        indices_arr = indices_arr.reshape(1, -1)
    if scores_arr.ndim == 1:
        scores_arr = scores_arr.reshape(1, -1)

    num_chunks = len(_chunks)

    for row_idx in range(indices_arr.shape[0]):
        for col_idx, idx in enumerate(indices_arr[row_idx]):
            try:
                idx_int = int(idx)
            except Exception:
                # Skip non-integer indices
                continue
            if idx_int < 0 or idx_int >= num_chunks:
                # Out-of-range index, skip
                continue

            c = _chunks[idx_int]
            snippet = c["text"][:400]

            # Optionally refine the snippet by finding the most relevant sentences
            try:
                if get_embeddings is not None and col_idx < max_chunks_with_sentence_snippets:
                    import re

                    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", c["text"]) if s.strip()]
                    max_sents = min(max_sentences_per_chunk, len(sents))
                    if max_sents > 0:
                        sent_batch = sents[:max_sents]
                        sent_embs = get_embeddings(sent_batch, "text-embedding-3-small")
                        q_emb = q_vec_arr.reshape(-1)
                        import numpy as _np

                        sent_matrix = _np.array(sent_embs).astype("float32")
                        sent_norm = sent_matrix / (_np.linalg.norm(sent_matrix, axis=1, keepdims=True) + 1e-10)
                        q_norm = q_emb / (_np.linalg.norm(q_emb) + 1e-10)
                        sims = (sent_norm @ q_norm).tolist()
                        top_idx = sorted(range(len(sims)), key=lambda x: -sims[x])[:3]
                        matched_sentences = [sent_batch[j] for j in top_idx if sims[j] > 0]
                        if matched_sentences:
                            snippet = " ... ".join(matched_sentences)
            except Exception:
                # If anything fails during refinement, fall back to the chunk snippet
                pass

            # obtain score safely
            try:
                score_val = float(scores_arr[row_idx][col_idx])
            except Exception:
                score_val = 0.0

            out.append({"chunk_id": c["id"], "text": snippet[:400], "score": score_val})

    return out


def chunk_read(chunk_id: str) -> Dict:
    global _chunks
    if _chunks is None:
        load_indexes()
    assert _chunks is not None
    chunk = next((c for c in _chunks if c["id"] == chunk_id), None)
    if not chunk:
        return {"error": f"chunk_id {chunk_id!r} not found"}
    return {"chunk_id": chunk_id, "source": chunk["source"], "full_context": chunk["coarse"]}


if __name__ == "__main__":
    try:
        load_indexes()
        print("Indexes loaded successfully. BM25 example:")
        for r in keyword_search("Scott Derrickson director", k=5):
            print(r)
    except Exception as e:
        print("Index load or search failed:", e)

"""Hierarchical index builder moved into arag package."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

def load_hotpot_corpus(max_examples: int = 1000) -> List[Dict[str, str]]:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("Please install the 'datasets' package to load HotpotQA: pip install datasets") from e

    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")

    paragraphs = []
    for ex in ds:
        titles = ex["context"]["title"]
        sents = ex["context"]["sentences"]
        for title, sent_list in zip(titles, sents):
            paragraphs.append({"title": title, "text": " ".join(sent_list)})

    seen = set()
    corpus = []
    for p in paragraphs:
        if p["title"] not in seen:
            seen.add(p["title"])
            corpus.append(p)
        if max_examples and len(corpus) >= max_examples:
            break

    print(f"Loaded {len(corpus)} unique paragraphs from HotpotQA (max_examples={max_examples})")
    return corpus


def simple_token_count(text: str) -> int:
    return max(1, len(text.split()))


def chunk_text(text: str, chunk_size_tokens: int = 200, chunk_overlap: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    n = len(words)
    while i < n:
        end = min(i + chunk_size_tokens, n)
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        i += chunk_size_tokens - chunk_overlap
    return chunks


def build_chunks(corpus: List[Dict[str, str]], fine_size: int = 200, fine_overlap: int = 30) -> List[Dict[str, str]]:
    chunks = []
    for doc in corpus:
        paragraph_text = doc["text"]
        nodes = chunk_text(paragraph_text, chunk_size_tokens=fine_size, chunk_overlap=fine_overlap)
        for i, node_text in enumerate(nodes):
            chunk_id = f"{doc['title'].replace(' ', '_')}_{i}"
            chunks.append({
                "id": chunk_id,
                "text": node_text,
                "source": doc["title"],
                "coarse": paragraph_text,
            })
    print(f"Created {len(chunks)} fine-grained chunks")
    return chunks


def build_bm25(chunks: List[Dict[str, str]], out_path: str = "bm25_index.pkl") -> None:
    try:
        from rank_bm25 import BM25Okapi
    except Exception as e:
        raise RuntimeError("Please install 'rank_bm25' to build BM25 index: pip install rank_bm25") from e

    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(out_path, "wb") as f:
        pickle.dump((bm25, chunks), f)
    print(f"Saved BM25 index to {out_path}")


def build_faiss(chunks: List[Dict[str, str]], out_index: str = "faiss_index.bin", out_chunks: Optional[str] = "chunks.pkl") -> None:
    try:
        import faiss
        import numpy as np
    except Exception as e:
        raise RuntimeError("Please install 'faiss-cpu' and 'numpy' to build FAISS index: pip install faiss-cpu numpy") from e

    from arag.embedding_backend import get_embedding_info, get_embeddings, get_model_fingerprint
    from arag.index_meta import record_build

    def embed_texts(texts: List[str]) -> List[List[float]]:
        return get_embeddings(texts, kind="passage")

    texts = [c["text"] for c in chunks]
    batch_size = 128
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"Embedding batch {i}..{i+len(batch)-1}")
        batch_embs = embed_texts(batch)
        embeddings.extend(batch_embs)

    embeddings_arr = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_arr)
    dim = embeddings_arr.shape[1]
    info = get_embedding_info()
    if info["dim"] is not None and info["dim"] != dim:
        raise RuntimeError(f"Embedding dimension {dim} does not match the configured model ({info})")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_arr)
    faiss.write_index(index, out_index)
    if out_chunks:  # None = leave an existing chunks.pkl untouched
        with open(out_chunks, "wb") as f:
            pickle.dump(chunks, f)
    record_build(os.path.dirname(out_index) or ".", info, dim, fingerprint=get_model_fingerprint(), num_chunks=len(chunks))
    print(f"Saved FAISS index to {out_index} and chunks to {out_chunks}")


def extract_sentences(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Split chunks into sentence records {'id', 'text', 'chunk_id', 'source'}."""
    import re

    sentences = []
    for c in chunks:
        chunk_id = c.get("id")
        text = c.get("text") or ""
        sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
        for i, s in enumerate(sents):
            if len(s) < 20:
                continue
            sid = f"{chunk_id}__sent_{i}"
            sentences.append({"id": sid, "text": s, "chunk_id": chunk_id, "source": c.get("coarse") or c.get("source") or ""})
    return sentences


def embed_sentence_index(sentences: List[Dict[str, str]], out_index: str) -> None:
    """Embed existing sentence records into a FAISS index (and update index_meta.json)."""
    try:
        import faiss
        import numpy as np
    except Exception as e:
        raise RuntimeError("Please install 'faiss-cpu' and 'numpy' to build FAISS index: pip install faiss-cpu numpy") from e

    from arag.embedding_backend import get_embedding_info, get_embeddings, get_model_fingerprint
    from arag.index_meta import record_build

    batch_size = 256
    embeddings: List[List[float]] = []
    for i in range(0, len(sentences), batch_size):
        batch = [s["text"] for s in sentences[i : i + batch_size]]
        print(f"Embedding sentence batch {i}..{i + len(batch) - 1}")
        embeddings.extend(get_embeddings(batch, kind="passage"))

    embeddings_arr = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_arr)
    dim = embeddings_arr.shape[1]
    info = get_embedding_info()
    if info["dim"] is not None and info["dim"] != dim:
        raise RuntimeError(f"Embedding dimension {dim} does not match the configured model ({info})")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_arr)
    faiss.write_index(index, out_index)
    record_build(os.path.dirname(out_index) or ".", info, dim, fingerprint=get_model_fingerprint(), num_sentences=len(sentences))
    print(f"Saved sentence FAISS index to {out_index}")


def build_sentence_faiss(chunks: List[Dict[str, str]], out_index: str = "sentence_faiss_index.bin", out_sentences: str = "sentences.pkl") -> None:
    """
    Build a sentence-level FAISS index. Splits each chunk into sentences, embeds
    them, and writes a FAISS index and sentences metadata. Each sentence record
    keeps a reference to the parent chunk id so retrieval can map back to chunk
    metadata and BM25 scores.
    """
    sentences = extract_sentences(chunks)
    if not sentences:
        print("No sentences extracted; skipping sentence FAISS build")
        return
    embed_sentence_index(sentences, out_index)
    with open(out_sentences, "wb") as f:
        pickle.dump(sentences, f)
    print(f"Saved sentences metadata to {out_sentences}")


def bm25_search_example(query: str, top_k: int = 5, bm25_path: str = "bm25_index.pkl") -> List[Dict]:
    with open(bm25_path, "rb") as f:
        bm25, chunks = pickle.load(f)
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_k_idx = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
    out = []
    for i, s in top_k_idx:
        if s <= 0:
            continue
        out.append({"chunk_id": chunks[i]["id"], "text": chunks[i]["text"][:300], "score": float(s)})
    return out


def semantic_search_example(query: str, k: int = 5, faiss_index_path: str = "faiss_index.bin", chunks_path: str = "chunks.pkl") -> List[Dict]:
    try:
        import faiss
        import numpy as np
    except Exception as e:
        raise RuntimeError("faiss and numpy required for semantic search") from e

    index = faiss.read_index(faiss_index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    from arag.embedding_backend import get_embedding

    vec_raw = get_embedding(query, kind="query")
    vec_arr = np.array(vec_raw).astype("float32").reshape(1, -1)
    faiss.normalize_L2(vec_arr)
    scores, indices = index.search(vec_arr, k)
    out = []
    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        out.append({"chunk_id": chunks[idx]["id"], "text": chunks[idx]["text"][:300], "score": float(scores[0][i])})
    return out


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and optional FAISS indexes for A-RAG")
    parser.add_argument("--max_examples", type=int, default=1000, help="Max unique paragraphs to load from HotpotQA")
    parser.add_argument("--build_faiss", type=lambda s: s.lower() in ("true", "1", "yes"), default=False, help="Whether to build FAISS index (local embedding model)")
    parser.add_argument("--build_sentence_faiss", type=lambda s: s.lower() in ("true", "1", "yes"), default=False, help="Whether to build sentence-level FAISS index (local embedding model)")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory for indexes")
    parser.add_argument("--quick_test", type=lambda s: s.lower() in ("true", "1", "yes"), default=True, help="Run quick validation searches after building")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    corpus = load_hotpot_corpus(max_examples=args.max_examples)
    chunks = build_chunks(corpus)

    chunks_path = os.path.join(args.out_dir, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks metadata to {chunks_path}")

    bm25_path = os.path.join(args.out_dir, "bm25_index.pkl")
    build_bm25(chunks, out_path=bm25_path)

    if args.build_faiss:
        print("Building FAISS index (local embedding model; may take a few minutes)...")
        out_index = os.path.join(args.out_dir, "faiss_index.bin")
        out_chunks = chunks_path
        build_faiss(chunks, out_index=out_index, out_chunks=out_chunks)

    if args.build_sentence_faiss:
        print("Building sentence-level FAISS index (local embedding model; may take a few minutes)...")
        out_sentence_index = os.path.join(args.out_dir, "sentence_faiss_index.bin")
        out_sentences = os.path.join(args.out_dir, "sentences.pkl")
        build_sentence_faiss(chunks, out_index=out_sentence_index, out_sentences=out_sentences)

    if args.quick_test:
        print("\nValidation checks:")
        q1 = "Scott Derrickson director"
        print(f"BM25 sample for query: '{q1}'")
        try:
            res1 = bm25_search_example(q1, top_k=5, bm25_path=bm25_path)
            for r in res1:
                print(r)
        except Exception as e:
            print("BM25 test failed:", e)

        if args.build_faiss:
            q2 = "Example semantic query: who directed a well-known film?"
            print(f"Semantic search sample for query: '{q2}'")
            try:
                res2 = semantic_search_example(q2, k=5, faiss_index_path=out_index, chunks_path=chunks_path)
                for r in res2:
                    print(r)
            except Exception as e:
                print("Semantic test failed:", e)

    print("\nPhase 1 complete (script finished).")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()

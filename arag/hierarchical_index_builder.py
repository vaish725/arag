"""Hierarchical index builder moved into arag package."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


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


def build_faiss(chunks: List[Dict[str, str]], out_index: str = "faiss_index.bin", out_chunks: str = "chunks.pkl") -> None:
    try:
        import faiss
        import numpy as np
    except Exception as e:
        raise RuntimeError("Please install 'faiss-cpu' and 'numpy' to build FAISS index: pip install faiss-cpu numpy") from e

    from arag.embedding_backend import get_embeddings

    def embed_texts(texts: List[str]) -> List[List[float]]:
        return get_embeddings(texts, model="text-embedding-3-small")

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
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_arr)
    faiss.write_index(index, out_index)
    with open(out_chunks, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved FAISS index to {out_index} and chunks to {out_chunks}")


def build_sentence_faiss(chunks: List[Dict[str, str]], out_index: str = "sentence_faiss_index.bin", out_sentences: str = "sentences.pkl") -> None:
    """
    Build a sentence-level FAISS index. Splits each chunk into sentences, embeds
    them, and writes a FAISS index and sentences metadata. Each sentence record
    keeps a reference to the parent chunk id so retrieval can map back to chunk
    metadata and BM25 scores.
    """
    try:
        import faiss
        import numpy as np
    except Exception as e:
        raise RuntimeError("Please install 'faiss-cpu' and 'numpy' to build FAISS index: pip install faiss-cpu numpy") from e

    from arag.embedding_backend import get_embeddings

    import re

    sentences = []  # list of {'id', 'text', 'chunk_id', 'source'}
    for c in chunks:
        chunk_id = c.get("id")
        text = c.get("text") or ""
        sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
        for i, s in enumerate(sents):
            if len(s) < 20:
                continue
            sid = f"{chunk_id}__sent_{i}"
            sentences.append({"id": sid, "text": s, "chunk_id": chunk_id, "source": c.get("coarse") or c.get("source") or ""})

    if not sentences:
        print("No sentences extracted; skipping sentence FAISS build")
        return

    texts = [s["text"] for s in sentences]
    # Embedding cache: if a cache file exists, load precomputed embeddings and
    # only compute embeddings for new sentences. Cache is a dict mapping sentence
    # id -> embedding list.
    cache_path = out_sentences + ".embcache.pkl"
    emb_cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                emb_cache = pickle.load(f)
            print(f"Loaded sentence embedding cache with {len(emb_cache)} entries")
        except Exception:
            emb_cache = {}

    batch_size = 256
    embeddings = []
    # For deterministic ordering, process sentences in list order
    to_compute_texts = []
    to_compute_ids = []
    for s in sentences:
        sid = s["id"]
        if sid in emb_cache:
            embeddings.append(emb_cache[sid])
        else:
            to_compute_ids.append(sid)
            to_compute_texts.append(s["text"])

    # Compute embeddings for sentences missing in cache
    if to_compute_texts:
        for i in range(0, len(to_compute_texts), batch_size):
            batch = to_compute_texts[i : i + batch_size]
            idx0 = i
            idx1 = i + len(batch) - 1
            print(f"Embedding sentence batch {idx0}..{idx1}")
            batch_embs = get_embeddings(batch, model="text-embedding-3-small")
            for sid, emb in zip(to_compute_ids[i : i + len(batch)], batch_embs):
                emb_cache[sid] = emb
                embeddings.append(emb)

    # At this point 'embeddings' is aligned with 'sentences' list order
    embeddings_arr = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_arr)
    dim = embeddings_arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_arr)
    faiss.write_index(index, out_index)
    with open(out_sentences, "wb") as f:
        pickle.dump(sentences, f)
    # Persist embedding cache for future incremental builds
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(emb_cache, f)
        print(f"Saved sentence FAISS index to {out_index}, sentences metadata to {out_sentences}, and embedding cache to {cache_path}")
    except Exception:
        print(f"Saved sentence FAISS index to {out_index} and sentences metadata to {out_sentences} (failed to write embedding cache)")


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

    vec_raw = get_embedding(query, model="text-embedding-3-small")
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
    parser.add_argument("--build_faiss", type=lambda s: s.lower() in ("true", "1", "yes"), default=False, help="Whether to build FAISS index (requires embeddings)")
    parser.add_argument("--build_sentence_faiss", type=lambda s: s.lower() in ("true", "1", "yes"), default=False, help="Whether to build sentence-level FAISS index (requires embeddings)")
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
        print("Building FAISS index (this may require an embedding API key and may take some time)...")
        out_index = os.path.join(args.out_dir, "faiss_index.bin")
        out_chunks = chunks_path
        build_faiss(chunks, out_index=out_index, out_chunks=out_chunks)

    if args.build_sentence_faiss:
        print("Building sentence-level FAISS index (this may require an embedding API key and may take some time)...")
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
    main()

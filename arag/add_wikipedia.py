"""Fetch a Wikipedia page, chunk it, and add to the local indexes under data/.

This utility does not require the `wikipedia` package; it uses the public
MediaWiki API via requests. It appends chunk entries to `data/chunks.pkl`
and rebuilds the BM25 index (`data/bm25_index.pkl`). Optionally you can also
rebuild the FAISS index but that requires embeddings and the --build-faiss flag
uses the embedding API.
"""

from __future__ import annotations

import argparse
import os
import pickle
import requests
from typing import List, Dict


def fetch_wikipedia_extract(title: str) -> str:
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "",
        "format": "json",
        "titles": title,
    }
    headers = {"User-Agent": "arag/1.0 (+https://github.com/vaish725/arag)"}
    r = S.get(URL, params=params, timeout=20, headers=headers)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if "extract" in page:
            return page["extract"]
    raise RuntimeError(f"No extract found for title: {title}")


def append_chunks_for_title(title: str, text: str, out_dir: str = "data") -> List[Dict]:
    # Import the chunking helper from the index builder
    from arag.hierarchical_index_builder import chunk_text

    nodes = chunk_text(text, chunk_size_tokens=200, chunk_overlap=30)
    new_chunks = []
    for i, node in enumerate(nodes):
        cid = f"{title.replace(' ', '_')}_{i}"
        new_chunks.append({
            "id": cid,
            "text": node,
            "source": title,
            "coarse": text,
        })

    os.makedirs(out_dir, exist_ok=True)
    chunks_path = os.path.join(out_dir, "chunks.pkl")
    chunks: List[Dict] = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
    chunks.extend(new_chunks)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Appended {len(new_chunks)} chunks for '{title}' to {chunks_path}")
    return chunks


def rebuild_bm25(chunks: List[Dict], out_dir: str = "data") -> None:
    from arag.hierarchical_index_builder import build_bm25

    bm25_path = os.path.join(out_dir, "bm25_index.pkl")
    build_bm25(chunks, out_path=bm25_path)


def main():
    parser = argparse.ArgumentParser(description="Add a Wikipedia page to local indexes")
    parser.add_argument("title", type=str, help="Wikipedia page title (e.g. 'Scott Derrickson')")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--rebuild_faiss", action="store_true", help="Also rebuild FAISS index (requires embeddings API and faiss)")
    args = parser.parse_args()

    text = fetch_wikipedia_extract(args.title)
    chunks = append_chunks_for_title(args.title, text, out_dir=args.out_dir)
    rebuild_bm25(chunks, out_dir=args.out_dir)

    if args.rebuild_faiss:
        print("Building FAISS index (this will call the embedding API and may take time)")
        from arag.hierarchical_index_builder import build_faiss

        out_index = os.path.join(args.out_dir, "faiss_index.bin")
        out_chunks = os.path.join(args.out_dir, "chunks.pkl")
        build_faiss(chunks, out_index=out_index, out_chunks=out_chunks)


if __name__ == "__main__":
    main()

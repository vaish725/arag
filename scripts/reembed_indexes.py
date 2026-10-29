#!/usr/bin/env python3
"""Re-embed the existing corpus into new FAISS indexes with the configured model.

Only the two FAISS indexes and index_meta.json change. chunks.pkl, sentences.pkl
and bm25_index.pkl are never rewritten (their hashes are checked before/after),
and nothing is downloaded from HotpotQA or Wikipedia. Embeddings are local, so
this makes no API calls.

Usage: PYTHONPATH=. python scripts/reembed_indexes.py [--data_dir data]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import pickle
import resource
import sys
import time

# The 8 articles added on top of the HotpotQA paragraphs (see scripts/fetch_and_index.py).
WIKIPEDIA_TITLES = [
    "The Godfather", "Titanic", "Titanic (1997 film)", "James Cameron",
    "Inception", "Christopher Nolan", "La La Land", "Damien Chazelle",
]
UNKNOWN_REVISION = "fetched before 2026-05-23, revid unknown"
KEEP_UNCHANGED = ("chunks.pkl", "sentences.pkl", "bm25_index.pkl")


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(data_dir: str = "data") -> int:
    import faiss

    from arag import hierarchical_index_builder as hib
    from arag.embedding_backend import get_embedding_info
    from arag.index_meta import read_meta, record_wikipedia_revisions

    info = get_embedding_info()
    print(f"Embedding with {info['provider']}/{info['model']} (dim {info['dim']})")
    if info["provider"] != "local":
        print("Refusing to run: this script is for the free local provider only.")
        return 2

    paths = {name: os.path.join(data_dir, name) for name in KEEP_UNCHANGED}
    before_hash = {n: _sha(p) for n, p in paths.items()}
    with open(paths["chunks.pkl"], "rb") as f:
        chunks = pickle.load(f)
    with open(paths["sentences.pkl"], "rb") as f:
        sentences = pickle.load(f)
    n_chunks, n_sentences = len(chunks), len(sentences)
    print(f"Before: {n_chunks} chunks, {n_sentences} sentences")
    if len(hib.extract_sentences(chunks)) != n_sentences:
        print("WARNING: sentences.pkl differs from a fresh split of chunks.pkl; keeping sentences.pkl as is.")

    # Stale caches hold OpenAI vectors; they must never be reused.
    for pattern in ("*.embcache.pkl", "sentence_embedding_cache*.pkl"):
        for stale in glob.glob(os.path.join(data_dir, pattern)):
            print(f"Deleting stale cache {stale}")
            os.remove(stale)

    t0 = time.time()
    hib.build_faiss(chunks, out_index=os.path.join(data_dir, "faiss_index.bin"), out_chunks=None)
    hib.embed_sentence_index(sentences, os.path.join(data_dir, "sentence_faiss_index.bin"))
    elapsed = time.time() - t0

    present = {c.get("source") for c in chunks}
    record_wikipedia_revisions(data_dir, {t: UNKNOWN_REVISION for t in WIKIPEDIA_TITLES if t in present})

    # Verification
    problems = []
    ci = faiss.read_index(os.path.join(data_dir, "faiss_index.bin"))
    si = faiss.read_index(os.path.join(data_dir, "sentence_faiss_index.bin"))
    if ci.ntotal != n_chunks:
        problems.append(f"chunk index ntotal {ci.ntotal} != {n_chunks} chunks")
    if si.ntotal != n_sentences:
        problems.append(f"sentence index ntotal {si.ntotal} != {n_sentences} sentences")
    if not (ci.d == si.d == info["dim"]):
        problems.append(f"dimensions differ: chunk {ci.d}, sentence {si.d}, model {info['dim']}")
    for n, p in paths.items():
        if _sha(p) != before_hash[n]:
            problems.append(f"{n} changed but must not")
    with open(paths["chunks.pkl"], "rb") as f:
        if len(pickle.load(f)) != n_chunks:
            problems.append("chunk count changed")
    meta = read_meta(data_dir)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    print(f"After: chunk index {ci.ntotal} x {ci.d}, sentence index {si.ntotal} x {si.d}")
    print(f"Embedding time {elapsed:.0f}s, peak memory {peak_mb:.0f} MB")
    print(f"index_meta.json: {meta}")
    if problems:
        print("VERIFICATION FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("Verification passed: counts unchanged, chunks/sentences/BM25 files byte-identical.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="data")
    raise SystemExit(main(parser.parse_args().data_dir))

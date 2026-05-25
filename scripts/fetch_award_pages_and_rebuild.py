#!/usr/bin/env python3
"""Fetch likely award-related Wikipedia pages for a small set of queries
and rebuild the sentence FAISS index.

This script tries a small set of title variants for each film/director and
uses `arag.add_wikipedia` utilities to append found pages to the local
`data/chunks.pkl` and rebuild BM25. After appending any new pages it calls
the hierarchical index builder to rebuild the sentence FAISS index.
"""

from __future__ import annotations

import sys
from typing import List


QUERIES = [
    {"film": "Doctor Strange (2016 film)", "director": "Scott Derrickson"},
    {"film": "Parasite (film)", "director": "Bong Joon-ho"},
    {"film": "La La Land (film)", "director": "Damien Chazelle"},
    {"film": "The Shape of Water", "director": "Guillermo del Toro"},
    {"film": "Inception", "director": "Christopher Nolan"},
    {"film": "Moonlight (film)", "director": "Barry Jenkins"},
]


def candidate_titles(entity: str) -> List[str]:
    # Variants to try for awards/accollades pages
    variants = [
        f"List of awards and nominations received by {entity}",
        f"List of accolades received by {entity}",
        f"Awards and nominations received by {entity}",
        f"{entity} awards and nominations",
        f"{entity} awards",
        f"Awards and nominations for {entity}",
        f"{entity} (awards and nominations)",
    ]
    # Also try the bare entity (some titles have dedicated awards pages under "List of awards and nominations received by X")
    variants.insert(0, entity)
    return variants


def try_fetch_and_append(title: str) -> bool:
    # Import utilities from add_wikipedia
    try:
        from arag.add_wikipedia import fetch_wikipedia_extract, append_chunks_for_title, rebuild_bm25
    except Exception as e:
        print("Failed to import add_wikipedia utilities:", e)
        return False

    try:
        text = fetch_wikipedia_extract(title)
    except Exception as e:
        # fetch failed (page not found or request error)
        return False

    try:
        append_chunks_for_title(title, text, out_dir="data")
        # rebuild BM25
        from arag.hierarchical_index_builder import build_bm25

        with open("data/chunks.pkl", "rb") as f:
            chunks = __import__("pickle").load(f)
        build_bm25(chunks, out_path="data/bm25_index.pkl")
        return True
    except Exception as e:
        print(f"Failed to append/rebuild for {title}:", e)
        return False


def main():
    added = []
    for q in QUERIES:
        film = q["film"]
        director = q["director"]
        print(f"\nChecking award pages for film: {film} and director: {director}")
        # try film variants first
        for t in candidate_titles(film):
            if try_fetch_and_append(t):
                print(f"Appended page for title: {t}")
                added.append(t)
                break
        # try director variants
        for t in candidate_titles(director):
            if try_fetch_and_append(t):
                print(f"Appended page for title: {t}")
                added.append(t)
                break

    if not added:
        print("No award pages were found/added for the provided queries.")
    else:
        print("Added pages:\n", "\n".join(added))
        # rebuild sentence FAISS index to include new sentences
        print("Rebuilding sentence-level FAISS index (this will use embeddings)...")
        import subprocess

        subprocess.run([sys.executable, "-m", "arag.hierarchical_index_builder", "--max_examples", "200", "--build_sentence_faiss", "true", "--out_dir", "data", "--quick_test", "false"], check=True)


if __name__ == "__main__":
    main()

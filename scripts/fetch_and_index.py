"""Fetch Wikipedia pages, append them to the local corpus, and rebuild indexes.

Usage examples:

# Append a single title and rebuild BM25 only
python3 scripts/fetch_and_index.py "The Godfather" --out_dir data

# Append multiple titles and rebuild BM25 + FAISS + sentence FAISS
python3 scripts/fetch_and_index.py "The Godfather" "Titanic" --out_dir data --rebuild-faiss --rebuild-sentence-faiss

# Read titles from a file (one per line)
python3 scripts/fetch_and_index.py --titles-file titles.txt --out_dir data --rebuild-faiss

Note: this script performs network requests to the Wikipedia API. Run it locally.
"""
from __future__ import annotations

import argparse
import sys
from typing import List


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Wikipedia titles and rebuild local indexes")
    parser.add_argument("titles", nargs="*", help="One or more Wikipedia page titles (e.g. 'The Godfather')")
    parser.add_argument("--titles-file", type=str, help="File with one title per line")
    parser.add_argument("--out_dir", type=str, default="data", help="Output data directory (default: data)")
    parser.add_argument("--rebuild-faiss", action="store_true", help="Also rebuild chunk-level FAISS index (requires embeddings and faiss)")
    parser.add_argument("--rebuild-sentence-faiss", action="store_true", help="Also rebuild sentence-level FAISS index (requires embeddings and faiss)")
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false", help="Do not skip appending if title already present in corpus")
    parser.set_defaults(dedupe=True)

    args = parser.parse_args(argv)

    titles: List[str] = list(args.titles or [])
    if args.titles_file:
        try:
            with open(args.titles_file, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        titles.append(t)
        except Exception as e:
            print(f"Failed to read titles file: {e}")
            return 2

    if not titles:
        print("No titles provided. Provide titles as arguments or via --titles-file.")
        parser.print_help()
        return 1

    try:
        from arag import add_wikipedia
        from arag import hierarchical_index_builder as hib
    except Exception as e:
        print(f"Failed to import repository helper modules: {e}")
        return 3

    all_chunks = None
    for title in titles:
        print(f"Fetching '{title}' from Wikipedia...")
        try:
            text = add_wikipedia.fetch_wikipedia_extract(title)
        except Exception as e:
            print(f"Failed to fetch '{title}': {e}")
            continue
        try:
            all_chunks = add_wikipedia.append_chunks_for_title(title, text, out_dir=args.out_dir, dedupe=args.dedupe)
        except Exception as e:
            print(f"Failed to append chunks for '{title}': {e}")
            continue

    if all_chunks is None:
        print("No chunks were added; aborting index rebuild.")
        return 4

    print("Rebuilding BM25 index...")
    try:
        add_wikipedia.rebuild_bm25(all_chunks, out_dir=args.out_dir)
    except Exception as e:
        print(f"Failed to rebuild BM25 index: {e}")
        return 5

    if args.rebuild_faiss:
        print("Rebuilding chunk-level FAISS index (may take time and require embeddings)...")
        try:
            out_index = f"{args.out_dir.rstrip('/')}" + "/faiss_index.bin"
            out_chunks = f"{args.out_dir.rstrip('/')}" + "/chunks.pkl"
            hib.build_faiss(all_chunks, out_index=out_index, out_chunks=out_chunks)
        except Exception as e:
            print(f"Failed to build chunk FAISS: {e}")
            return 6

    if args.rebuild_sentence_faiss:
        print("Rebuilding sentence-level FAISS index (may take time and require embeddings)...")
        try:
            out_index = f"{args.out_dir.rstrip('/')}" + "/sentence_faiss_index.bin"
            out_sentences = f"{args.out_dir.rstrip('/')}" + "/sentences.pkl"
            hib.build_sentence_faiss(all_chunks, out_index=out_index, out_sentences=out_sentences)
        except Exception as e:
            print(f"Failed to build sentence FAISS: {e}")
            return 7

    print("Done. Added titles and rebuilt requested indexes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

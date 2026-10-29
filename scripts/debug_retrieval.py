#!/usr/bin/env python3
"""Debug: what did each search return, and which passage did the award check use?

Runs one film through the director/award flow in two modes in the same process:
  old: vector search disabled (semantic_search raises), i.e. the BM25 fallback path
       that produced the baseline in scripts/regression_baseline.json
  new: the configured local embedding model (needs the re-embedded indexes)
Read-only; makes no API calls.

Usage: PYTHONPATH=. python scripts/debug_retrieval.py [--film "La La Land"] [--top 5]
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys


def corpus_win_sentences(film_prefix: str, person_prefix: str) -> dict:
    """chunk_id -> sentence, for corpus sentences about an Academy Award win."""
    chunks = pickle.load(open(os.path.join(os.environ.get("ARAG_DATA_DIR", "data"), "chunks.pkl"), "rb"))
    out = {}
    for c in chunks:
        cid = c["id"]
        if not (cid.startswith(film_prefix) or cid.startswith(person_prefix)):
            continue
        for s in re.split(r"(?<=[.?!])\s+", c["text"]):
            sl = s.lower()
            if "academy award" in sl and re.search(r"\b(won|win|wins|winning|winner)\b", sl):
                out.setdefault(cid, s.strip())
    return out


def one(s, n=200) -> str:
    return re.sub(r"\s+", " ", s or "")[:n]


def run_mode(mode: str, film: str, top: int) -> dict:
    from arag import agent_loop

    original = agent_loop.semantic_search
    if mode == "old":
        def _off(*a, **k):
            raise RuntimeError("vector search disabled (baseline = BM25 fallback path)")
        agent_loop.semantic_search = _off
    try:
        res = agent_loop.run_agent(
            f"Who directed {film} and did they win any Academy Awards?",
            allow_fallback=False,
            enable_director_flow=True,
        )
    finally:
        agent_loop.semantic_search = original

    print(f"\n{'=' * 78}\nMODE {mode.upper()}: {film}\n{'=' * 78}")
    print("answer   :", res["answer"])
    print("citations:", res["citations"])

    first_found = {}
    counts = {}
    for t in res["tool_trace"]:
        counts[t["tool"]] = counts.get(t["tool"], 0) + 1
        if t["tool"] in ("semantic_search", "keyword_search"):
            q = t["args"].get("query")
            print(f"\n[{t['tool']}] {q!r}" + (f"  ERROR: {t['error']}" if "error" in t else ""))
            for rank, r in enumerate((t.get("raw_results") or [])[:top], 1):
                scores = ", ".join(f"{k}={r[k]:.3f}" for k in ("score", "faiss_score", "bm25_score") if isinstance(r.get(k), (int, float)))
                print(f"   {rank}. {r['chunk_id']:28} {scores:44} {one(r.get('text'))}")
                first_found.setdefault(r["chunk_id"], (t["tool"], q, rank))
        elif t["tool"] == "chunk_read":
            first_found.setdefault(t["args"].get("chunk_id"), ("chunk_read", "direct", 0))
    print("\ntool calls:", counts)

    print("\nretrieved passages (citation numbers index this list):")
    for i, r in enumerate(res["retrieved_texts"], 1):
        how = first_found.get(r["chunk_id"], ("?", "", 0))
        mark = "  <== CITED" if i in res["citations"] else ""
        print(f"  [{i:2}] {r['chunk_id']:28} first via {how[0]}({how[1]!r}, rank {how[2]}){mark}")
        if i in res["citations"]:
            print(f"       {one(r['text'])}")
    return {"res": res, "found": first_found, "counts": counts}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default="La La Land")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    film_prefix = args.film.replace(" ", "_")

    # Director's name from the old run's answer, to find the win sentences in the corpus.
    runs = {}
    runs["old"] = run_mode("old", args.film, args.top)
    m = re.search(r"Director:\s*(.+?)\.\s", runs["old"]["res"]["answer"])
    person_prefix = (m.group(1) if m else "").replace(" ", "_")
    try:
        runs["new"] = run_mode("new", args.film, args.top)
    except Exception as e:  # keep the old-mode output even if the model is unavailable
        print(f"\nMODE NEW failed: {type(e).__name__}: {e}")

    wins = corpus_win_sentences(film_prefix, person_prefix) if person_prefix else {}
    print(f"\n{'=' * 78}\nCORPUS SENTENCES ABOUT A WIN (for {film_prefix}* and {person_prefix}*)\n{'=' * 78}")
    for cid, s in wins.items():
        print(f"  {cid:28} {one(s, 160)}")
    print("\nWere those chunks retrieved, and by what? (cited = used by the verdict)")
    for mode, r in runs.items():
        cited_ids = {r["res"]["retrieved_texts"][i - 1]["chunk_id"] for i in r["res"]["citations"] if 0 < i <= len(r["res"]["retrieved_texts"])}
        for cid in wins:
            how = r["found"].get(cid)
            print(f"  {mode:3} {cid:28} " + (f"retrieved via {how[0]}({how[1]!r}, rank {how[2]})" if how else "NOT RETRIEVED") + ("  [CITED]" if cid in cited_ids else ""))
    print("\nkeyword_search calls:", {m: r["counts"].get("keyword_search", 0) for m, r in runs.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""EXPERIMENT (not used by the product): what would hybrid retrieval do to the regression films?

Compares, on the same corpus and indexes:
  bm25   : semantic search off, keyword (BM25) only            (= the baseline)
  semantic: current behaviour with the local embedding model   (BM25 only runs as a fallback on error)
  hybrid : BM25 and semantic run on every search, merged with reciprocal rank fusion (k=60)
Verdict logic is untouched; only the retrieval function the agent sees is swapped.
Read-only; makes no API calls.

Usage: PYTHONPATH=. python scripts/prototype_hybrid.py [--baseline scripts/regression_baseline.json]
"""
from __future__ import annotations

import argparse
import json
import sys

from regression_report import run_all, score, tally

RRF_K = 60


def rrf_merge(semantic: list, keyword: list, k: int) -> list:
    """Reciprocal rank fusion of two ranked result lists, de-duplicated by chunk_id."""
    fused, info = {}, {}
    for source, results in (("semantic", semantic), ("keyword", keyword)):
        seen = set()
        rank = 0
        for r in results:
            cid = r["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            rank += 1
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            info.setdefault(cid, {"text": r.get("text", "")})[source] = r.get("faiss_score" if source == "semantic" else "score", 0.0)
    out = []
    for cid, s in sorted(fused.items(), key=lambda x: -x[1])[:k]:
        out.append({
            "chunk_id": cid,
            "text": info[cid]["text"],
            "score": s,
            "faiss_score": float(info[cid].get("semantic", 0.0) or 0.0),
            "bm25_score": float(info[cid].get("keyword", 0.0) or 0.0),
        })
    return out


def make_hybrid(rt, original_semantic):
    def hybrid(query: str, k: int = 5):
        pool = max(k * 4, 20)
        try:
            sem = original_semantic(query, k=pool)
        except Exception:
            sem = []  # model/index unavailable: degrade to BM25 only
        kw = rt.keyword_search(query, k=pool)
        return rrf_merge(sem, kw, k)

    return hybrid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="scripts/regression_baseline.json")
    args = ap.parse_args()
    from arag import agent_loop
    from arag import retrieval_tools as rt

    truth = {f: r.get("expected_truth", {}).get("verdict", "?") for f, r in json.load(open(args.baseline)).items()}
    original = agent_loop.semantic_search
    modes = {}

    def _off(*a, **k):
        raise RuntimeError("vector search disabled")

    agent_loop.semantic_search = _off
    modes["bm25"] = run_all()
    agent_loop.semantic_search = original
    modes["semantic"] = run_all()
    agent_loop.semantic_search = make_hybrid(rt, original)
    modes["hybrid"] = run_all()
    agent_loop.semantic_search = original

    def cell(m, f):
        r = modes[m][f]
        return f"{r['verdict']} {r['citations']}"

    print(f"{'film':22} | {'bm25 (baseline)':22} | {'semantic (current)':22} | {'hybrid (RRF)':22} | truth")
    for f in modes["bm25"]:
        print(f"{f:22} | {cell('bm25', f):22} | {cell('semantic', f):22} | {cell('hybrid', f):22} | {truth.get(f, '?')}")
    print()
    for m in modes:
        sc = [score(modes[m][f]["verdict"], truth.get(f, "?")) for f in modes[m]]
        print(f"{m:9}: {tally(sc)}  (of {len(sc)})")
    directors_same = all(modes["hybrid"][f]["director"] == modes["bm25"][f]["director"] for f in modes["bm25"])
    print(f"\ndirectors identical across bm25 and hybrid: {directors_same}")
    for m in ("semantic", "hybrid"):
        ch = [f for f in modes["bm25"] if modes[m][f]["verdict"] != modes["bm25"][f]["verdict"]]
        print(f"verdicts that differ from the baseline in {m}: {ch or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

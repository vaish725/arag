#!/usr/bin/env python3
"""Small evaluation harness: precision@k for award-retrieval on director queries.

This script runs `semantic_search` in two modes:
- chunk: force chunk-level FAISS search (disable sentence FAISS if present)
- sentence: use sentence-level FAISS when available

It reports precision@k where a retrieved item is considered relevant if its
text contains any award-related token (academy, oscar, nominated, won).
"""

from __future__ import annotations

import re
import statistics
from typing import List, Tuple

import arag.retrieval_tools as rt

import pickle
import os


QUERIES: List[Tuple[str, str]] = [
    ("Who directed Doctor Strange and did they win any Academy Awards?", "Scott Derrickson"),
    ("Who directed Parasite and did they win any Academy Awards?", "Bong Joon-ho"),
    ("Who directed La La Land and did they win any Academy Awards?", "Damien Chazelle"),
    ("Who directed The Shape of Water and did they win any Academy Awards?", "Guillermo del Toro"),
    ("Who directed Inception and did they win any Academy Awards?", "Christopher Nolan"),
    ("Who directed Moonlight and did they win any Academy Awards?", "Barry Jenkins"),
]

AWARD_TOKENS = ["academy", "oscar", "oscars", "academy award", "academy awards", "nominated", "nominated for", "won an oscar", "won an academy", "won the oscar", "won the academy"]


def is_award_mention(text: str) -> bool:
    tl = (text or "").lower()
    for t in AWARD_TOKENS:
        if t in tl:
            return True
    return False


def precision_at_k(results: List[dict], k: int = 5) -> float:
    if not results:
        return 0.0
    topk = results[:k]
    hits = sum(1 for r in topk if is_award_mention(r.get("text", "")))
    return hits / float(k)


def run_eval(k: int = 5):
    rt.load_indexes()

    # load chunks metadata so we can inspect parent chunk full text
    chunks_path = os.path.join("data", "chunks.pkl")
    chunks = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
    chunk_map = {c.get("id"): c for c in chunks}

    # load sentences metadata if present
    sentences_path = os.path.join("data", "sentences.pkl")
    sentences = []
    if os.path.exists(sentences_path):
        with open(sentences_path, "rb") as f:
            sentences = pickle.load(f)
    sentence_map = {s.get("id"): s for s in sentences}

    # build ground-truth sentence sets per query using award tokens + heuristics
    gt_per_query = {}
    for q, director in QUERIES:
        # simple heuristics: film/title tokens from query and director last name
        ql = q.lower()
        parts = [p for p in director.split() if p]
        last_name = parts[-1].lower() if parts else None
        # tokens from film title portion in the query (best-effort split)
        # remove question words
        candidates = re.findall(r"[A-Za-z0-9()'-]+", q)
        gt_texts = set()
        for s in sentences:
            txt = (s.get("text") or "").lower()
            source = (s.get("source") or "").lower()
            match_award = any(t in txt or t in source for t in AWARD_TOKENS)
            if not match_award:
                continue
            ok = False
            # match by director last name
            if last_name and last_name in txt:
                ok = True
            # match by film tokens in the source or in the sentence
            for tok in candidates:
                if len(tok) <= 3:
                    continue
                if tok.lower() in txt or tok.lower() in source:
                    ok = True
                    break
            if ok:
                gt_texts.add(s.get("text").strip())
        gt_per_query[q] = gt_texts

    modes = ["chunk", "sentence"]
    scores = {m: [] for m in modes}

    for m in modes:
        print(f"\nRunning mode: {m}")
        for q, _ in QUERIES:
            # For chunk mode, temporarily disable sentence FAISS in-memory
            saved_sentence_index = getattr(rt, "_sentence_faiss_index", None)
            saved_sentences = getattr(rt, "_sentences", None)
            try:
                if m == "chunk":
                    rt._sentence_faiss_index = None
                    rt._sentences = None

                res = rt.semantic_search(q, k=k)
                topk = res[:k]
                # precision@k by exact match against ground truth sentence texts
                gt_texts = gt_per_query.get(q, set())
                retrieved_texts = [r.get("text", "").strip() for r in topk]
                hits = sum(1 for t in retrieved_texts if t in gt_texts)
                p = hits / float(k)
                # recall@k: fraction of ground-truth sentences retrieved in top-k
                recall = 0.0
                if gt_texts:
                    recall = sum(1 for t in gt_texts if t in retrieved_texts) / float(len(gt_texts))
                scores[m].append(p)
                print(f"Query: {q!r} -> precision@{k}: {p:.3f}, recall@{k}: {recall:.3f} (returned {len(res)} items, GT count={len(gt_texts)})")
            except Exception as e:
                print(f"Query {q!r} failed in mode {m}: {e}")
                scores[m].append(0.0)
            finally:
                # restore
                rt._sentence_faiss_index = saved_sentence_index
                rt._sentences = saved_sentences

    print("\nSummary:\n")
    for m in modes:
        vals = scores[m]
        avg = statistics.mean(vals) if vals else 0.0
        print(f"Mode {m}: average precision@{k} = {avg:.3f} (per-query: {[round(v,3) for v in vals]})")


if __name__ == "__main__":
    run_eval(k=5)

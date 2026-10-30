#!/usr/bin/env python3
"""Run the director/award regression films and print director, verdict, citations.

Usage: PYTHONPATH=. python scripts/regression_report.py [--out file.json] [--baseline file.json]
With --baseline it prints baseline vs new vs expected truth and flags differences for review.
Makes no API calls (evidence-only mode, keys unused).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

FILMS = [
    "Inception",
    "Titanic (1997 film)",
    "Titanic",
    "La La Land",
    "Doctor Strange",
    "The Godfather",
    "The Godfather Part II",
]


def verdict_of(answer: str) -> str:
    a = answer.lower()
    if "won at least one academy award" in a:
        return "won"
    if "nominated" in a:
        return "nominated"
    return "unknown"


def score(verdict: str, truth: str) -> str:
    """correct / wrong / no answer. "unknown" is an honest 'no answer': in
    evidence-only mode absence of evidence cannot prove a negative, so declining
    to answer is never counted as wrong."""
    if verdict == "unknown":
        return "no answer"
    return "correct" if verdict == truth else "wrong"


def tally(scores) -> str:
    scores = list(scores)
    return ", ".join(f"{k} {scores.count(k)}" for k in ("correct", "wrong", "no answer"))


def evidence_line(r: dict) -> str:
    """One line saying WHICH passage produced the verdict, so a right answer for the
    wrong reason is visible."""
    ev = r.get("verdict_evidence")
    if not ev:
        return "(no award evidence found: verdict is 'unknown')"
    text = re.sub(r"\s+", " ", ev.get("matched_text") or "").strip()
    if len(text) > 170:
        text = text[:167] + "..."
    return f"{ev.get('chunk_id')} [citation {ev.get('citation')}] via {ev.get('type')} pattern: \"{text}\""


def career_line(r: dict) -> str:
    """Career wins found for OTHER works (never counted for the film asked about)."""
    c = r.get("career_wins") or []
    if not c:
        return ""
    return "career wins (other works): " + "; ".join(f"{x.get('work') or '?'} via {x.get('chunk_id')}" for x in c)


def run_all() -> dict:
    from arag import agent_loop

    out = {}
    for film in FILMS:
        res = agent_loop.run_agent(
            f"Who directed {film} and did they win any Academy Awards?",
            allow_fallback=False,
            enable_director_flow=True,
        )
        answer = res.get("answer") or ""
        m = re.search(r"Director:\s*(.+?)\.\s", answer)
        trace = res.get("tool_trace", [])
        out[film] = {
            "director": m.group(1) if m else None,
            "verdict": verdict_of(answer),
            "citations": res.get("citations", []),
            "verdict_evidence": res.get("verdict_evidence"),
            "career_wins": res.get("career_wins", []),
            "semantic_search_errors": [t["error"] for t in trace if t.get("tool") == "semantic_search" and t.get("error")][:1],
            "semantic_search_ok": any(t.get("tool") == "semantic_search" and "error" not in t for t in trace),
            "answer": answer,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    results = run_all()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
    if not args.baseline:
        for film, r in results.items():
            print(f"{film:24} director={r['director']!s:22} verdict={r['verdict']:9} citations={r['citations']} semantic_ok={r['semantic_search_ok']}")
            print(f"{'':24} verdict from: {evidence_line(r)}")
            if career_line(r):
                print(f"{'':24} {career_line(r)}")
        return 0

    with open(args.baseline) as f:
        base = json.load(f)
    # The baseline is the previous behaviour, NOT ground truth. Differences from it
    # are flagged for review, never treated as failures; the expected_truth column
    # (hand-checked Oscar outcome of the director for that film) is shown beside both.
    print(f"{'film':22} | {'baseline':34} | {'new':34} | {'truth':9} | {'baseline score':14} | {'new score':9} | flags")
    flagged = []
    base_scores, new_scores = [], []
    for film, r in results.items():
        b = base.get(film, {})
        truth = (b.get("expected_truth") or {}).get("verdict", "?")
        base_s = f"{b.get('director')}, {b.get('verdict')}, {b.get('citations')}"
        new_s = f"{r['director']}, {r['verdict']}, {r['citations']}"
        flags = []
        if b and r["director"] != b.get("director"):
            flags.append("DIRECTOR CHANGED")
        if b and r["verdict"] != b.get("verdict"):
            flags.append("VERDICT CHANGED")
        if b and r["citations"] != b.get("citations"):
            flags.append("citations changed")
        base_score = score(b.get("verdict", "unknown"), truth) if b else "-"
        new_score = score(r["verdict"], truth)
        if b:
            base_scores.append(base_score)
        new_scores.append(new_score)
        if b and r["verdict"] != b.get("verdict"):
            flags.append(f"{'IMPROVED' if (base_score, new_score) in (('wrong', 'correct'), ('wrong', 'no answer'), ('no answer', 'correct')) else 'WORSE' if (base_score, new_score) in (('correct', 'wrong'), ('correct', 'no answer'), ('no answer', 'wrong')) else 'moved'} vs truth")
        if any(f in ("DIRECTOR CHANGED", "VERDICT CHANGED") for f in flags):
            flagged.append(film)
        print(f"{film:22} | {base_s:34} | {new_s:34} | {truth:9} | {base_score:14} | {new_score:9} | {'; '.join(flags) or '-'}")
        print(f"{'':22}   verdict from: {evidence_line(r)}")
        if career_line(r):
            print(f"{'':22}   {career_line(r)}")
    print()
    print(f"Baseline: {tally(base_scores)}  (of {len(base_scores)})")
    print(f"New:      {tally(new_scores)}  (of {len(new_scores)})")
    print()
    if flagged:
        print("FLAG FOR REVIEW (director or verdict differs from the baseline): " + ", ".join(flagged))
    else:
        print("No director or verdict differs from the baseline.")
    print("Truth notes:")
    for film in results:
        note = (base.get(film, {}).get("expected_truth") or {}).get("note")
        if note:
            print(f"  {film}: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

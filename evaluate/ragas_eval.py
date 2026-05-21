"""Simple evaluation harness (lightweight RAGAS-like) for A-RAG answers.

Usage: python3 evaluate/ragas_eval.py --preds preds.jsonl --gold gold.jsonl
Each line: {"id": "<id>", "question": "...", "answer": "..."}
"""

from __future__ import annotations

import argparse
import json
from typing import List


def load_lines(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def simple_score(pred: str, gold: str) -> int:
    """Very simple exact-match-ish scorer.

    Returns 1 if overlap of non-stop tokens > threshold.
    """
    p_tokens = set([t.lower() for t in pred.split() if len(t) > 2])
    g_tokens = set([t.lower() for t in gold.split() if len(t) > 2])
    if not g_tokens:
        return 0
    overlap = p_tokens & g_tokens
    return 1 if len(overlap) / max(1, len(g_tokens)) > 0.5 else 0


def main(preds_path: str, gold_path: str):
    preds = {p["id"]: p for p in load_lines(preds_path)}
    gold = {g["id"]: g for g in load_lines(gold_path)}

    ids = set(preds.keys()) & set(gold.keys())
    if not ids:
        print("No overlapping ids between preds and gold")
        return
    total = 0
    correct = 0
    for i in ids:
        total += 1
        correct += simple_score(preds[i]["answer"], gold[i]["answer"])
    print(
        f"Evaluated {total} examples. Simple-score: {correct}/{total} = "
        f"{correct/total:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", dest="preds", required=True)
    parser.add_argument("--gold", dest="gold", required=True)
    args = parser.parse_args()
    main(args.preds, args.gold)

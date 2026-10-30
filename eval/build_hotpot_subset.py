#!/usr/bin/env python3
"""Build eval/data/hotpot_subset.json: HotpotQA validation questions whose gold supporting
paragraphs are all in our 1,000-document corpus, with gold at CHUNK level.

HotpotQA's own gold supporting facts are (title, sentence index). A gold chunk is any chunk of
that title (data/chunks.pkl) that contains the gold sentence. Questions where a gold sentence
cannot be found inside a chunk are dropped and counted. Needs `datasets` and network access once
(the dataset may be cached); the output is committed so run_eval.py needs neither.

Usage: PYTHONPATH=. python eval/build_hotpot_subset.py
"""
from __future__ import annotations

import collections
import json
import pickle
import re

from datasets import load_dataset

OUT = "eval/data/hotpot_subset.json"
CORPUS_SIZE = 1000  # must match arag.hierarchical_index_builder (--max_examples 1000)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
    chunks = pickle.load(open("data/chunks.pkl", "rb"))
    by_title = collections.defaultdict(list)
    for c in chunks:
        by_title[c["source"]].append(c)

    # Rebuild the HotpotQA part of the corpus exactly as the builder did: first 1,000 unique titles.
    titles, seen = [], set()
    for ex in ds:
        for t in ex["context"]["title"]:
            if t not in seen:
                seen.add(t)
                titles.append(t)
        if len(titles) >= CORPUS_SIZE:
            break
    corpus_titles = set(titles[:CORPUS_SIZE])
    assert corpus_titles <= set(by_title), "corpus titles missing from chunks.pkl"

    kept, dropped = [], collections.Counter()
    for ex in ds:
        gold = list(zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]))
        if not {t for t, _ in gold} <= corpus_titles:
            continue
        ctx = dict(zip(ex["context"]["title"], ex["context"]["sentences"]))
        facts, ok = [], True
        for title, sid in gold:
            sents = ctx.get(title)
            if sents is None or sid >= len(sents):
                dropped["gold sentence index missing from the question's context"] += 1
                ok = False
                break
            sentence = norm(sents[sid])
            ids = [c["id"] for c in by_title[title] if sentence in norm(c["text"])]
            if not ids:
                dropped["gold sentence not inside any chunk"] += 1
                ok = False
                break
            facts.append({"title": title, "sent_id": sid, "sentence": sentence, "chunk_ids": ids})
        if ok:
            kept.append({
                "id": ex["id"], "question": ex["question"], "answer": ex["answer"],
                "type": ex["type"], "level": ex["level"], "gold": facts,
                "gold_chunk_ids": sorted({i for f in facts for i in f["chunk_ids"]}),
            })
    out = {
        "source": "HotpotQA fullwiki validation (hotpot_qa)",
        "corpus_titles": CORPUS_SIZE,
        "questions_with_all_gold_titles_in_corpus": len(kept) + sum(dropped.values()),
        "dropped": dict(dropped),
        "questions": kept,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT}: {len(kept)} usable questions, dropped {dict(dropped)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Free evaluation of retrieval and director->award answers. No LLM or API calls.

Track A (retrieval): HotpotQA questions whose gold supporting paragraphs are in our corpus
  (eval/data/hotpot_subset.json). recall@5 / recall@10 at CHUNK level against HotpotQA's gold
  supporting facts, for BM25, semantic (the shipped search) and hybrid (reciprocal rank fusion,
  scripts/prototype_hybrid.py).
Track B (awards): hand-checked films whose pages are in the corpus (eval/data/award_set.json),
  scored correct / wrong / no answer, with the verdict passage per row.

Both tracks are split into dev and test by eval/data/splits.json (seeded, committed). Nothing here
tunes anything; retrieval and verdict code are used exactly as shipped.

Usage: PYTHONPATH=. python eval/run_eval.py [--out eval/results/after_fixes]
       PYTHONPATH=. python eval/run_eval.py --reaggregate eval/results/before_fixes   # no run, recompute the summary
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from prototype_hybrid import make_hybrid, rrf_merge  # noqa: E402
from regression_report import score, verdict_of  # noqa: E402

MODES = ("bm25", "semantic", "hybrid")
POOL = 100  # candidates fetched from each retriever before de-duplicating to chunks
KS = (5, 10)


def wilson(successes: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(centre - half, 3), round(centre + half, 3))


def unique_chunk_ids(results):
    seen, out = set(), []
    for r in results:
        if r["chunk_id"] not in seen:
            seen.add(r["chunk_id"])
            out.append(r["chunk_id"])
    return out


# ---------------------------------------------------------------- track A: retrieval
def retrieve(mode: str, question: str, rt, sem_fn):
    if mode == "bm25":
        return unique_chunk_ids(rt.keyword_search(question, k=POOL))
    if mode == "semantic":
        return unique_chunk_ids(sem_fn(question, k=POOL))
    kw = rt.keyword_search(question, k=POOL)
    return unique_chunk_ids(rrf_merge(sem_fn(question, k=POOL), kw, POOL))


def run_retrieval(subset, splits, rt, sem_fn):
    split_of = {i: "dev" for i in splits["retrieval"]["dev"]}
    split_of.update({i: "test" for i in splits["retrieval"]["test"]})
    rows = []
    for q in subset["questions"]:
        gold = set(q["gold_chunk_ids"])
        row = {"id": q["id"], "split": split_of[q["id"]], "type": q["type"], "question": q["question"],
               "n_gold_chunks": len(gold), "gold_chunk_ids": sorted(gold), "modes": {}}
        for mode in MODES:
            ranked = retrieve(mode, q["question"], rt, sem_fn)
            ranks = [i + 1 for i, c in enumerate(ranked) if c in gold]
            m = {"top10": ranked[:10], "first_gold_rank": ranks[0] if ranks else None}
            for k in KS:
                hit = len(gold & set(ranked[:k]))
                m[f"recall@{k}"] = hit / len(gold)
                m[f"all_gold@{k}"] = hit == len(gold)
            row["modes"][mode] = m
        rows.append(row)
    return rows


def aggregate_retrieval(rows):
    out = {}
    for split in ("dev", "test", "all"):
        sel = [r for r in rows if split == "all" or r["split"] == split]
        agg = {"n_questions": len(sel)}
        for mode in MODES:
            ms = [r["modes"][mode] for r in sel]
            a = {}
            for k in KS:
                a[f"recall@{k}"] = round(sum(m[f"recall@{k}"] for m in ms) / len(ms), 3) if ms else None
                n_all = sum(m[f"all_gold@{k}"] for m in ms)
                a[f"all_gold@{k}"] = {"count": n_all, "of": len(ms), "wilson95": wilson(n_all, len(ms))}
            a["mrr"] = round(sum(1 / m["first_gold_rank"] for m in ms if m["first_gold_rank"]) / len(ms), 3) if ms else None
            agg[mode] = a
        # paired: on how many questions is A strictly better than B at recall@10 (ties ignored)
        agg["paired_recall@10"] = {}
        for a_, b_ in (("hybrid", "bm25"), ("hybrid", "semantic"), ("semantic", "bm25")):
            better = sum(r["modes"][a_]["recall@10"] > r["modes"][b_]["recall@10"] for r in sel)
            worse = sum(r["modes"][a_]["recall@10"] < r["modes"][b_]["recall@10"] for r in sel)
            agg["paired_recall@10"][f"{a_}_vs_{b_}"] = {"better": better, "worse": worse, "tied": len(sel) - better - worse}
        out[split] = agg
    return out


# ---------------------------------------------------------------- track B: awards
def _norm_name(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("\u2019", "'")).strip(" .,")


def director_status(m: dict, expected: str) -> str:
    """The director as its own column, scored STRICTLY on the string the user sees.

    correct   : exactly the expected name(s) (co-directors may be joined with "and"), nothing else
    malformed : contains the expected name(s) but with extra text, e.g. "Joe Dante and produced by Michael Finnell"
    wrong     : another name, or a co-director missing
    no answer : "Unknown"
    """
    found = m.get("director_found")
    if not found or found.lower().startswith("unknown"):
        return "no answer"
    names = [_norm_name(n) for n in re.split(r"\s+and\s+", expected)]
    rest = _norm_name(found)
    if not all(n in rest for n in names):
        return "wrong"  # another name, or a co-director missing
    for n in names:
        rest = rest.replace(n, "")
    return "correct" if re.sub(r"\b(and)\b|[,\s]", "", rest) == "" else "malformed"


def overall_status(m: dict, expected: str, accept_malformed: bool = False) -> str:
    """Wrong overall if the director (wrong OR malformed) or the verdict is wrong; correct only if both are correct.
    accept_malformed=True is the lenient reading, reported alongside for transparency."""
    d, v = director_status(m, expected), m["score"]
    if accept_malformed and d == "malformed":
        d = "correct"
    if d in ("wrong", "malformed") or v == "wrong":
        return "wrong"
    return "correct" if d == "correct" and v == "correct" else "no answer"


def tally3(statuses) -> dict:
    statuses = list(statuses)
    return {"correct": statuses.count("correct"), "wrong": statuses.count("wrong"), "no_answer": statuses.count("no answer")}


def run_awards(award_set, splits, agent_loop, rt):
    split_of = {f: "dev" for f in splits["awards"]["dev"]}
    split_of.update({f: "test" for f in splits["awards"]["test"]})
    original = agent_loop.semantic_search

    def off(*a, **k):
        raise RuntimeError("vector search disabled for the bm25 mode")

    patches = {"bm25": off, "semantic": original, "hybrid": make_hybrid(rt, original)}
    rows = []
    for f in award_set["films"]:
        row = {"film": f["film"], "split": split_of[f["film"]], "expected_director": f["director"],
               "truth": f["truth"], "basis": f["basis"], "modes": {}}
        for mode in MODES:
            agent_loop.semantic_search = patches[mode]
            try:
                res = agent_loop.run_agent(f"Who directed {f['film']} and did they win any Academy Awards?",
                                           allow_fallback=False, enable_director_flow=True)
            finally:
                agent_loop.semantic_search = original
            answer = res.get("answer") or ""
            verdict = verdict_of(answer)
            m = re.search(r"Director:\s*(.+?)\.\s", answer)
            found = m.group(1) if m else None
            surnames = [p.split()[-1].lower() for p in re.split(r"\s+and\s+", f["director"])]
            ev = res.get("verdict_evidence")
            row["modes"][mode] = {
                "verdict": verdict,
                "score": score(verdict, f["truth"]),
                "director_found": found,
                "director_ok": bool(found) and any(s in found.lower() for s in surnames),
                "director_status": None,  # filled below
                "overall": None,
                "verdict_passage": None if not ev else {"chunk_id": ev["chunk_id"], "type": ev["type"], "matched_text": ev.get("matched_text")},
                "career_wins": [{"work": c.get("work"), "chunk_id": c.get("chunk_id")} for c in res.get("career_wins", [])],
                "answer": answer,
            }
        for m in row["modes"].values():
            m["director_status"] = director_status(m, f["director"])
            m["overall"] = overall_status(m, f["director"])
        rows.append(row)
    return rows


def aggregate_awards(rows):
    out = {}
    for split in ("dev", "test", "all"):
        sel = [r for r in rows if split == "all" or r["split"] == split]
        agg = {"n_films": len(sel), "truth_mix": {t: sum(r["truth"] == t for r in sel) for t in ("won", "nominated", "no_win")}}
        answerable = [r for r in sel if r["truth"] != "no_win"]
        for mode in MODES:
            ms = {r["film"]: r["modes"][mode] for r in sel}
            expected = {r["film"]: r["expected_director"] for r in sel}
            verdict = tally3(m["score"] for m in ms.values())
            dstat = [director_status(m, expected[f]) for f, m in ms.items()]
            agg[mode] = {
                **verdict,  # verdict only (kept for compatibility)
                "verdict": verdict,
                # "wrong" here includes "malformed"; the malformed count is also given on its own
                "director": {**tally3("wrong" if d == "malformed" else d for d in dstat), "malformed": dstat.count("malformed")},
                "overall": tally3(overall_status(m, expected[f]) for f, m in ms.items()),
                "overall_if_malformed_accepted": tally3(overall_status(m, expected[f], accept_malformed=True) for f, m in ms.items()),
                "by_truth": {
                    t: {
                        "n": sum(1 for r in sel if r["truth"] == t),
                        "verdict": tally3(r["modes"][mode]["score"] for r in sel if r["truth"] == t),
                        "overall": tally3(overall_status(r["modes"][mode], r["expected_director"]) for r in sel if r["truth"] == t),
                    }
                    for t in ("won", "nominated", "no_win")
                },
                # coverage on answerable films = correct / (films whose truth is won or nominated)
                "coverage_on_answerable_verdict": {"correct": sum(1 for r in answerable if r["modes"][mode]["score"] == "correct"), "of": len(answerable)},
                "coverage_on_answerable_overall": {"correct": sum(1 for r in answerable if overall_status(r["modes"][mode], r["expected_director"]) == "correct"), "of": len(answerable)},
                "wrong_overall_films": [r["film"] for r in sel if overall_status(r["modes"][mode], r["expected_director"]) == "wrong"],
            }
            for key in ("coverage_on_answerable_verdict", "coverage_on_answerable_overall"):
                c = agg[mode][key]
                c["ratio"] = round(c["correct"] / c["of"], 3) if c["of"] else None
        out[split] = agg
    return out


def reaggregate(directory: str) -> int:
    """Recompute summary.json from the saved per-row results (no retrieval or agent run)."""
    splits = json.load(open("eval/data/splits.json"))
    rows_r = json.load(open(os.path.join(directory, "retrieval_results.json")))["rows"]
    awards_doc = json.load(open(os.path.join(directory, "awards_results.json")))
    rows_a = awards_doc["rows"]
    for r in rows_a:  # refresh the stored statuses with the current scoring rules
        for m in r["modes"].values():
            m["director_status"] = director_status(m, r["expected_director"])
            m["overall"] = overall_status(m, r["expected_director"])
    with open(os.path.join(directory, "awards_results.json"), "w") as f:
        json.dump(awards_doc, f, indent=1, sort_keys=True)
        f.write("\n")
    summary = {"seed": splits["seed"], "pool_per_retriever": POOL, "retrieval": aggregate_retrieval(rows_r), "awards": aggregate_awards(rows_a)}
    with open(os.path.join(directory, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"re-aggregated {directory}/summary.json from the saved rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--reaggregate", metavar="DIR", help="only recompute DIR/summary.json from DIR's saved rows")
    args = ap.parse_args()
    if args.reaggregate:
        return reaggregate(args.reaggregate)
    os.makedirs(args.out, exist_ok=True)

    from arag import agent_loop
    from arag import retrieval_tools as rt

    rt.load_indexes()
    sem_fn = rt.semantic_search
    subset = json.load(open("eval/data/hotpot_subset.json"))
    award_set = json.load(open("eval/data/award_set.json"))
    splits = json.load(open("eval/data/splits.json"))

    retrieval_rows = run_retrieval(subset, splits, rt, sem_fn)
    award_rows = run_awards(award_set, splits, agent_loop, rt)
    summary = {
        "seed": splits["seed"],
        "pool_per_retriever": POOL,
        "retrieval": aggregate_retrieval(retrieval_rows),
        "awards": aggregate_awards(award_rows),
    }
    for name, obj in (("retrieval_results.json", {"rows": retrieval_rows}), ("awards_results.json", {"rows": award_rows}), ("summary.json", summary)):
        with open(os.path.join(args.out, name), "w") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
            f.write("\n")

    t = summary["retrieval"]["test"]
    print(f"RETRIEVAL test (n={t['n_questions']}): recall@5 / recall@10 / all-gold@10 / MRR")
    for m in MODES:
        print(f"  {m:9} {t[m]['recall@5']:.3f} / {t[m]['recall@10']:.3f} / {t[m]['all_gold@10']['count']}/{t[m]['all_gold@10']['of']} / {t[m]['mrr']:.3f}")
    a = summary["awards"]["test"]
    print(f"AWARDS test (n={a['n_films']}), overall (director AND verdict) correct / wrong / no answer:")
    for m in MODES:
        o = a[m]["overall"]
        print(f"  {m:9} {o['correct']} / {o['wrong']} / {o['no_answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

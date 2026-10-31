# Evaluation (free, no LLM calls)

Everything here runs on the committed indexes and the local embedding model. Reproduce with
`pip install -r requirements-eval.txt`, then `PYTHONPATH=. python eval/run_eval.py --out <dir>` and
`python eval/make_chart.py`. **These are small-sample results, not a general accuracy claim.**

## Method

**Track A: retrieval.** HotpotQA validation (fullwiki) questions whose gold supporting paragraphs are all in our
1,000-document corpus, using HotpotQA's own gold supporting facts. A *gold chunk* is any chunk of the gold
paragraph that contains the gold sentence (`eval/data/hotpot_subset.json`). Metric: **recall@5 / recall@10 at
chunk level**, plus "all gold chunks in the top k" with a Wilson 95% interval. Methods: **BM25**
(`keyword_search`), **semantic** (the shipped `semantic_search`, bge-small embeddings) and **hybrid** (BM25 +
semantic merged by reciprocal rank fusion, `scripts/prototype_hybrid.py`). Each retriever returns 100
candidates, de-duplicated to chunks.

**Track B: director-award answers.** 26 films whose pages are in the corpus (`eval/data/award_set.json`), asked
as "Who directed X and did they win any Academy Awards?" in evidence-only mode. Truth = the director's Oscar
outcome *for that film*: `won` / `nominated` / `no_win`. Scored **correct** (verdict equals truth), **wrong**
(a different positive claim) or **no answer** ("unknown"; absence of evidence cannot prove a negative, so this
is never counted as wrong). Rows with the verdict passage: `eval/results/*/awards_results.json`.

**Labels are agent-sourced, not owner-verified.** Every row carries `source_url`, `supporting_text` (verbatim,
at most 25 words) and `verified_by: "agent-sourced"`. `python eval/verify_award_labels.py` re-checks, against
the live Wikipedia pages, that each quote is on its page and that every `no_win` row's `absence_check` holds
(the director's full name appears on neither of the two ceremony pages after the film's release): 26 of 26
pass. No source contradicted a label, so no label was corrected. Two caveats: for the 15 `no_win` films with no
Oscar story at all, no sentence can show a negative, so their `supporting_text` is the ceremony's Best Directing
row and the claim rests on the absence check; and the Academy Awards database was not queried.

**Split.** Seed `20261030`, 40% dev / 60% test (`eval/data/splits.json`, committed). Retrieval 11 dev / 16 test;
awards 10 dev / 16 test (stratified by truth class). Only test-split numbers are reported below.

## Results, test split

![summary](eval/results/eval_summary.png)

### Held-out, before fixes
Results from `eval/results/before_fixes/`. Nothing had been tuned on the test split at this point.

| Retrieval (n=16) | recall@5 | recall@10 | all gold in top 10 (95% interval) | MRR |
|---|---|---|---|---|
| BM25 | 0.72 | 0.91 | 13/16 (0.57 to 0.93) | 0.87 |
| Semantic | 0.88 | 1.00 | 16/16 (0.81 to 1.00) | 0.92 |
| Hybrid (RRF) | 0.81 | 1.00 | 16/16 (0.81 to 1.00) | 0.91 |

Paired on recall@10: semantic beats BM25 on 3 of 16 questions and loses on 0; hybrid beats semantic on 0 and
loses on 0 (16 tied), and is lower at recall@5. **There is no evidence that hybrid retrieval helps at n=16.**

### After fixes: test split no longer held out
Two verdict faults were found on this test split (the film's own win counted as the director's, and a non-Oscar
nomination counted as an Oscar one) and fixed in PR #8. The numbers below, from `eval/results/after_fixes/`,
were produced **after** looking at this split, so they are **not held out** and say nothing about unseen films.
Retrieval is unchanged by the fixes (identical results, checked by `run_eval.py` output comparison).

| Awards (n=16 films: 2 won, 1 nominated, 13 no_win), correct / wrong / no answer | before fixes (held out) | after fixes (not held out) |
|---|---|---|
| BM25 | 1 / 3 / 12 | 1 / 1 / 14 |
| Semantic | 2 / 2 / 12 | 2 / 0 / 14 |
| Hybrid (RRF) | 1 / 3 / 12 | 1 / 1 / 14 |

What changed: *Innerspace* (the film's Best Visual Effects win, "the only film directed by Dante to do so") and
*Silent Light* (an Independent Spirit nomination) are now "no answer". The one remaining wrong answer, for BM25
and hybrid only, is *The Godfather* reported as nominated: the passage naming Coppola's win is not retrieved by
those methods, a retrieval effect and not one of the fixed faults.

## Limits

- **Small n.** 16 test questions and 16 test films: one question moves recall by about 6 points and the
  intervals are wide. Differences of a question or two are noise.
- **Retrieval is likely optimistic.** The corpus was built from the contexts of the first validation questions,
  so every usable question has its distractors in the corpus; only 27 of 7,405 questions qualify.
- Award rows are not independent (Derrickson 4 films, Cameron 2); most films have no Oscar story, so "no
  answer" dominates by design. Director detection misses 3 of 26 films.
- Chunks are 200-word windows, so chunk-level gold is usually the whole paragraph.

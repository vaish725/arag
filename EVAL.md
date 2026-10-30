# Evaluation (free, no LLM calls)

Everything here runs on the committed indexes and the local embedding model. Nothing was tuned: retrieval
and verdict code are used exactly as shipped. Reproduce with `PYTHONPATH=. python eval/run_eval.py` then
`python eval/make_chart.py` (results in `eval/results/`).

## Method

**Track A: retrieval.** HotpotQA validation (fullwiki) questions whose gold supporting paragraphs are all in
our 1,000-document corpus, using HotpotQA's own gold supporting facts. A *gold chunk* is any chunk of the gold
paragraph that contains the gold sentence (`eval/data/hotpot_subset.json`). Metric: **recall@5 / recall@10 at
chunk level** (share of gold chunks found in the top k unique chunks), plus "all gold chunks in top k" with a
Wilson 95% interval. Methods: **BM25** (`keyword_search`), **semantic** (the shipped `semantic_search`,
bge-small embeddings) and **hybrid** (BM25 + semantic merged by reciprocal rank fusion, k=60,
`scripts/prototype_hybrid.py`). Each retriever returns 100 candidates, de-duplicated to chunks.

**Track B: director-award answers.** 26 films whose pages are in the corpus (`eval/data/award_set.json`), asked
as "Who directed X and did they win any Academy Awards?" in evidence-only mode. Truth = the director's Oscar
outcome *for that film*: `won` / `nominated` / `no_win`. Each answer is scored **correct** (verdict equals
truth), **wrong** (a different positive claim) or **no answer** ("unknown"; absence of evidence cannot prove a
negative, so this is never counted as wrong). Per-film rows with the verdict passage are in
`eval/results/awards_results.json`. **Labels.** They were written by the assistant from the corpus and general knowledge (`basis` in the file),
then **corroborated against each film's Wikipedia article** (revision ids pinned in
`eval/data/award_label_evidence.json`; re-run `python eval/verify_award_labels.py`). Result: 26 of 26
consistent, 22 automatically and 4 (True Lies, Innerspace, Romeo + Juliet, The Family Man) after reading the
passages by hand; the reasons are recorded in the file. The five films shared with
`scripts/regression_baseline.json` use the truth sourced there. Limits: an article that does not mention a
nomination cannot prove there was none, and the Academy Awards database was not queried. `owner_verified` stays
false for every row until the owner checks them.

**Split.** Seed `20261030`, 40% dev / 60% test (`eval/data/splits.json`, committed, generated once by
`eval/build_splits.py`). Retrieval: ids sorted, shuffled, first 40% = dev. Awards: stratified by truth class.
*dev may be used for tuning; test is held out and must never be used to tune.* Retrieval 11 dev / 16 test;
awards 10 dev / 16 test.

## Results, test split (held out)

![summary](eval/results/eval_summary.png)

| Retrieval (n=16) | recall@5 | recall@10 | all gold in top 10 (95% interval) | MRR |
|---|---|---|---|---|
| BM25 | 0.72 | 0.91 | 13/16 (0.57 to 0.93) | 0.87 |
| Semantic | 0.88 | 1.00 | 16/16 (0.81 to 1.00) | 0.92 |
| Hybrid (RRF) | 0.81 | 1.00 | 16/16 (0.81 to 1.00) | 0.91 |

Paired on recall@10 (test): semantic beats BM25 on 3 questions, loses on 0; hybrid beats BM25 on 3, loses on 0;
**hybrid vs semantic: 0 better, 0 worse, 16 tied** (and hybrid is lower at recall@5: 0.81 vs 0.88).

| Awards (n=16 films: 2 won, 1 nominated, 13 no_win) | correct | wrong | no answer | director found |
|---|---|---|---|---|
| BM25 | 1 | 3 | 12 | 14/16 |
| Semantic | 2 | 2 | 12 | 14/16 |
| Hybrid (RRF) | 1 | 3 | 12 | 14/16 |

Only test-split numbers are reported here; per-question and per-film rows for both splits are in
`eval/results/*.json`. These are small-sample results on a held-out split, not a general accuracy claim.

## What the numbers say

- **Semantic beats BM25** in retrieval (recall@5 0.88 vs 0.72). **Hybrid does not help beyond semantic** on the
  held-out set, so per the plan it does not get its own PR yet. Re-check on a larger question set.
- **The award track is mostly "no answer"** (12 of 16), by design: 13 of the test films have no Oscar story, so
  the best possible answer is "unknown". The positive-truth films: Godfather and La La Land are correct (right
  passage, per `verdict_passage`); About a Boy (nominated) has no Oscar passage in its corpus page.
- **The two wrong claims are verdict-logic faults this set exposes** (no code changed here):
  1. *Innerspace*: "...and won an Oscar, the only f..." is the **film's** win (visual effects), counted as the
     director's because no category is listed. Truth: Joe Dante did not win; the article says the film "won an
     Academy Award for Best Visual Effects, the only film directed by Dante to do so".
  2. *Silent Light*: "The film was nominated for Best Foreign Film at the 24th **Independent Spirit Awards**"
     matches the generic `was nominated` pattern, so a non-Oscar nomination is reported as an Oscar nomination.
- Director detection misses 3 of 26 films (Ed Wood, Romeo + Juliet, Last Action Hero).
- A career-win case works: *True Lies* is "unknown" and the career line names Titanic.

## Limits

- **Small n.** 16 test questions and 16 test films: one question moves recall by about 6 points and the
  intervals are wide. Differences of a question or two are noise.
- **Retrieval is likely optimistic.** The 1,000-document corpus was built from the contexts of the first
  validation questions, so every usable question has its distractors in the corpus and only 27 of 7,405
  questions qualify (29 had all gold titles in the corpus; 2 were dropped because a gold sentence index was
  missing from the context).
- Question-set repeats: Derrickson (4 films) and Cameron (2) appear several times, so the awards rows are not
  independent. The award set covers only films with pages in the corpus; most are no-Oscar films.
- Chunks are 200-word windows, so chunk-level gold is usually the whole paragraph.

# Status

Last updated with the "Make degraded modes visible" change (see "What changed" below). This file describes
what is true of the code on `main`; it makes no accuracy claim. Held-out evaluation numbers will be added
only after that evaluation is merged (PR #6).

## What the system is today

A **deterministic, evidence-only two-hop baseline**: hand-written Python chooses the retrieval steps in a
fixed order (title lookup, keyword search, semantic search, chunk reads) and assembles a cited answer from the
retrieved passages. Its one specialised path is "Who directed *X* and did they win any Academy Awards?"
(director, then the Oscar outcome for that film). Other questions get a retrieval dump, or an LLM answer if
the optional LLM mode is on.

It is **not the paper's LLM agent**: no model decides which tool to call, there is no tool-calling loop and no
planner. The three retrieval tools (`keyword_search`, `semantic_search`, `chunk_read`) exist and a budget on
tool calls per question is enforced, but the choice of steps is code, not a model.

- **Retrieval:** BM25 (`rank_bm25`), plus FAISS vector search over chunks and sentences using a **free local
  embedding model** (`BAAI/bge-small-en-v1.5` via fastembed, downloaded once, about 65 MB). No embeddings API.
- **Corpus:** 1,000 HotpotQA validation paragraphs plus 8 hand-added Wikipedia articles (1,494 chunks, 8,283
  sentences). Committed in `data/` with `data/index_meta.json`, which records the model, dimension, counts and
  the model's file hashes.
- **Keys:** none needed. `ANTHROPIC_API_KEY` is optional and only enables the LLM-fallback answer mode.
- **Spend controls:** per-question and per-day caps on API calls and tokens (`arag/runtime_ops.py`), a
  per-session question limit in the Streamlit app, and a per-question tool-call budget (`max_steps`).
- **Degraded mode is visible:** if vector search is off (model missing or failed to download, index/model
  mismatch, no vector index), the Streamlit app shows "Meaning search unavailable: keyword-only mode. Reason: ...",
  the CLI prints the same line to stderr, and every `run_agent` result carries `vector_search`. A one-line
  startup health line is logged.

## What changed in PRs #1 to #5 and #8

| PR | Change |
|---|---|
| [#1](https://github.com/vaish725/arag/pull/1) | Fix plan phases 0-2: blank-answer bug in LLM mode, CLI flag pass-through, dead `find_award_mentions`, spend caps back (thread-safe), tests that cannot reach the network or real keys, offline `semantic_search` tests. |
| [#2](https://github.com/vaish725/arag/pull/2) | Local embeddings (fastembed) replace OpenAI; `index_meta.json` and a mismatch/download guard that turns vector search off while keyword search keeps working; indexes re-embedded; CI caches the model. |
| [#3](https://github.com/vaish725/arag/pull/3) | Manual `reembed.yml` workflow (workflow_dispatch only, no secrets) used to rebuild the indexes on a runner. |
| [#4](https://github.com/vaish725/arag/pull/4) | Award evidence: "winning N including <category>" counts as a win; "See also" / "List of" / reference text no longer counts; the verdict passage is reported. |
| [#5](https://github.com/vaish725/arag/pull/5) | Award attribution: generic "Oscar winner" needs the surname; wins are tied to the film asked about; career wins are reported separately and a work is named only when a passage states it; regression truth sourced from Wikipedia; tests read expectations from `scripts/regression_baseline.json`. |
| [#8](https://github.com/vaish725/arag/pull/8) | Regression labels sourced from Wikipedia (`source_url`, `supporting_text`); two general verdict fixes: a film's win counts for the director only if the sentence names Best Director or names the director as a recipient, and a nomination counts as an Oscar nomination only if "Academy Award(s)" or "Oscar" is in the same sentence. |

Open: [#6](https://github.com/vaish725/arag/pull/6), the evaluation (retrieval and award tracks, seeded dev/test split; award scores are reported by truth class).

## Known limits

- Only the director to Oscar question type has a specialised path; the verdict logic is regex heuristics over
  retrieved sentences. The evaluation (PR #6) found two faults, now fixed in #8 (a film's win counted as the
  director's; a non-Oscar nomination counted as an Oscar one). Other faults may remain: only 26 + 7 films were
  tested.
- **Director detection** now accepts "directed ... by" with up to four words in between ("directed and produced
  by", "directed, co-produced, and co-written by"), the name stops where it ends, and the director sentence must
  name the film or come from the film's own page. It was developed on the 26 + 7 evaluation films (all exact
  afterwards), so how it behaves on unseen films is unknown.
- The corpus is tiny and shaped around the demo films. Retrieval constants (boosts, thresholds, blend weights,
  top-k) are hand-picked and untuned.
- The recipe can use many tool calls per question (up to 40 observed), including `chunk_read` probes that do
  not appear in the trace; the default budget is 100.
- `llm_backend` still contains an OpenAI path that would activate if `OPENAI_API_KEY` is set in the
  environment. It is not used by default and is not documented as supported.
- The index loader reads pickles from `data/` (trusted files only). Paths resolve from the package, or from
  `ARAG_DATA_DIR`.
- CI lint, format and type checks (`black`, `isort`, `flake8`, `mypy`) are informational (`|| true`); they are not
  enforced yet.
- Hybrid retrieval exists only as a prototype (`scripts/prototype_hybrid.py`); it is not used by the app.

## Open work

1. Merge and review the evaluation (PR #6); its award labels are agent-sourced from Wikipedia, not owner-verified.
2. Trace completeness and the legacy OpenAI path in `llm_backend`.
3. Evaluate hybrid retrieval on a larger question set before deciding whether it ships.
4. Chunking tails and a single BM25 tokenizer (C-19); enforce formatting and type checks in CI (C-22).
5. The LLM tool-calling loop the paper describes (Phase 5).

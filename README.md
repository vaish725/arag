arag
====

A small, deterministic **evidence-only two-hop retrieval baseline** inspired by A-RAG (Retrieval-Augmented
Agent). It is **not** the paper's LLM agent: hand-written Python chooses the retrieval steps (title lookup,
keyword search, vector search, chunk reads) and assembles a cited answer from the retrieved passages. Its
specialised path answers "Who directed *X* and did they win any Academy Awards?" over a small corpus
(1,000 HotpotQA paragraphs plus 8 Wikipedia articles).

Retrieval uses BM25 and FAISS vector search with a **free local embedding model**
(`BAAI/bge-small-en-v1.5` via fastembed). No embeddings API and no OpenAI. The only API key the system can
use is `ANTHROPIC_API_KEY`, optional, for the LLM-fallback answer mode. See `docs/STATUS.md` for what works
today, what changed, and the known limits.

Key features
------------

- A deterministic director-to-award two-hop flow with cited, evidence-only answers
- BM25 keyword search and local-embedding vector search (chunk and sentence level)
- Optional LLM-fallback answers through Anthropic (off by default; needs `ANTHROPIC_API_KEY`)
- Visible degraded mode: if vector search is unavailable the app and CLI say "keyword-only mode" and why
- Spend caps, an enforced per-question tool-call budget, and a per-session limit in the demo
- A Streamlit demo and a CLI runner; pytest suite with offline tests

Repository layout
-----------------

- `arag/` — core Python package (agent loop, retrieval tools, index builder,
  embedding/LLM backends, runtime helpers)
- `data/` — the committed index artifacts (BM25, chunks, sentences, the two FAISS indexes and
  `index_meta.json`); only build-time caches are ignored
- `scripts/` — index rebuild, regression report and verification scripts
- `eval/`, `EVAL.md` — evaluation data and method (when merged)
- `docs/STATUS.md` — current status, what changed, known limits
- `tests/` — unit tests and fixtures
- `ARAG_IMPLMEMENTATION.MD` — design notes and implementation plan

Quickstart
----------

Prerequisites

- Python 3.10 to 3.14 (the versions CI tests; Streamlit Cloud runs 3.14)
- A virtual environment (recommended)
- Network access once, to download the embedding model (about 65 MB from Hugging Face)
- An Anthropic API key **only if** you want the optional LLM-fallback mode

Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/download_model.py   # one-time model download
```

Environment variables

No key is needed for keyword and vector search. For the optional LLM-fallback mode only:

```bash
export ANTHROPIC_API_KEY="..."
```

The vector indexes record the embedding model they were built with in `data/index_meta.json`. If the
configured model differs, or the model cannot be loaded, vector search is switched off: the app shows
"Meaning search unavailable: keyword-only mode. Reason: ...", the CLI prints the same line to stderr, and
keyword search keeps working.

Rebuild the vector indexes for the current model (local, free, no corpus download):

```bash
PYTHONPATH=. python scripts/reembed_indexes.py
```

Rebuild the whole corpus from scratch (needs the `datasets` package and network; downloads HotpotQA from
Hugging Face):

```bash
PYTHONPATH=. python -m arag.hierarchical_index_builder --max_examples 1000 --out_dir data \
    --build_faiss true --build_sentence_faiss true --quick_test false
```

Then re-add the hand-picked Wikipedia articles with `scripts/fetch_and_index.py`.

Run the agent (CLI)

```bash
python -m arag.agent_loop --question "Who directed Inception and did they win any Academy Awards?"
# flags: --max_steps N (most tool calls), --max_tokens N (LLM answer length), --allow-fallback /
#        --no-allow-fallback (LLM may answer from outside knowledge; needs ANTHROPIC_API_KEY),
#        --enable-director-flow / --no-enable-director-flow
```

Run the Streamlit demo (if included)

```bash
streamlit run streamlit_app.py
```

Development
-----------

Run tests

```bash
pytest -q
```

Rebuild & recommended commands
------------------------------

After changing the embedding model or the indexing code, rebuild the vector indexes:

```bash
# from repo root
PYTHONPATH=. python scripts/reembed_indexes.py   # re-embed the existing chunks and sentences
PYTHONPATH=. pytest -q
```

Run a quick deterministic director->award smoke check:

```bash
PYTHONPATH=. python -c "from arag import agent_loop as a; print(a.run_agent('Who directed Titanic (1997 film) and did they win any Academy Awards?', allow_fallback=False, enable_director_flow=True))"
```

Static checks

```bash
black --check .
isort --check .
flake8
mypy
```

Project notes and migration
---------------------------

This project was refactored to consolidate the core modules under the
`arag/` package. If you used older top-level module imports, update them to use
the package form, for example:

```py
from arag.agent_loop import run_agent
```

Built index artifacts live in `data/` and are committed so the Streamlit app has data to load.

Contributing
------------

Contributions are welcome. Please open issues for bugs or feature requests and
submit pull requests against `main`. Follow the existing code style and add
tests for any new behavior.

License
-------

This project is licensed under the MIT License. See the `LICENSE` file for
details and the full license text.

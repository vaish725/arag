arag
====

A Python reference implementation of a Retrieval-Augmented Agent (A-RAG).

This repository provides a small, extensible toolkit for building and running
retrieval-augmented generation (RAG) agents. It includes support for lightweight
BM25 ranking and optional FAISS vector search backends, helpers for building
document indexes, and wrappers around embedding and LLM providers (OpenAI and
Anthropic). The project also demonstrates a two-hop reasoning flow and a
reusable agent loop suitable for experimentation and integration.

Key features
------------

- Two-hop and single-hop retrieval-augmented agent flows
- BM25 ranking (rank_bm25) for quick, dependency-light retrieval
- Optional FAISS vector index support for fast similarity search
- Pluggable embedding and LLM backends (OpenAI and Anthropic wrappers)
- Utilities to build and persist indexes from document corpora
- Small Streamlit demo and a CLI agent runner for experimentation
- Test suite and development tooling (pytest, flake8, mypy)

Repository layout
-----------------

- `arag/` — core Python package (agent loop, retrieval tools, index builder,
  embedding/LLM backends, runtime helpers)
- `data/` — recommended place for index artifacts and other generated data
  (this directory is ignored by VCS via `.gitignore`) 
- `tests/` — unit tests and fixtures
- `ARAG_IMPLMEMENTATION.MD` — design notes and implementation plan

Quickstart
----------

Prerequisites

- Python 3.9+ (3.10/3.11 recommended)
- A virtual environment (recommended)
- OpenAI and/or Anthropic API credentials if you plan to run real LLM/embedding calls

Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables

Set provider credentials (example for OpenAI):

```bash
export OPENAI_API_KEY="sk-..."
# export ANTHROPIC_API_KEY="..."  # if using Anthropic
```

Build indexes (example)

Prepare a corpus under `data/corpus/` (plain text files or a simple CSV) and run:

```bash
python -m arag.hierarchical_index_builder --input data/corpus --build_faiss True
```

This produces index artifacts under `data/` (BM25 pickle, chunks pickle, and
an optional FAISS binary). The `data/` directory is intentionally git-ignored to
avoid committing large binaries.

Run the agent (CLI)

```bash
python -m arag.agent_loop --question "Who directed Doctor Strange?" --max_steps 6
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

After adding new pages or changing the indexing code, rebuild the indexes and the sentence FAISS index (if used):

```bash
# from repo root
PYTHONPATH=. python -m arag.hierarchical_index_builder --input data/corpus --build_faiss True --build_sentence_faiss True
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

The repository defaults to storing built index artifacts in `data/` and
ignoring that directory in version control. To preserve history of a previously
committed artifact, perform a `git mv` locally and commit the change.

Contributing
------------

Contributions are welcome. Please open issues for bugs or feature requests and
submit pull requests against `main`. Follow the existing code style and add
tests for any new behavior.

License
-------

This project is licensed under the MIT License. See the `LICENSE` file for
details and the full license text.

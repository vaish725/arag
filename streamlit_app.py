"""Simple Streamlit demo for A-RAG pipeline.

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # local .env; arag modules no longer load it at import time

# Bridge Streamlit Cloud secrets into the process environment, since the
# arag backends read credentials via os.environ (populated from .env locally).
for _key in ("ANTHROPIC_API_KEY",):  # the only key the system uses (optional, for LLM answers)
    try:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
    except Exception:
        pass

from arag.agent_loop import run_agent


# Make the startup health line ("arag.health") visible in the server log.
_arag_log = logging.getLogger("arag")
if not _arag_log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _arag_log.addHandler(_handler)
    _arag_log.setLevel(logging.INFO)


@st.cache_resource(show_spinner="Loading embedding model and indexes...")
def _startup() -> bool:
    """Once per server process: load the local model and the indexes (logs the health line).

    Failures are not raised: they put the app in keyword-only mode, which is shown in a banner.
    """
    from arag import retrieval_tools
    from arag.embedding_backend import warm_up

    try:
        warm_up()
    except Exception:
        pass  # surfaced through vector_search_status() below
    try:
        retrieval_tools.load_indexes()
    except Exception:
        pass
    return True


_startup()

from arag.health import degraded_line  # noqa: E402
from arag.retrieval_tools import vector_search_status  # noqa: E402

st.title("A-RAG Demo")
_vs_enabled, _vs_reason = vector_search_status()
_top_banner = st.empty()
if not _vs_enabled and st.session_state.get("result") is None:
    # Before the first answer: say so up front. Once an answer is shown, the banner sits above it.
    _top_banner.warning(degraded_line(_vs_reason))

mode = st.selectbox("Mode", ["evidence-only (no LLM)", "semantic + BM25 + optional LLM fallback"])
question = st.text_input(
    "Question", "",
    placeholder="Ask something like: Who directed Inception and did they win any Academy Awards?",
)
max_steps = st.slider(
    "Max steps (tool calls)", 1, 200, 100,
    help="Most retrieval tool calls for one question. When it is reached the run stops using tools and says so.",
)
max_tokens = st.slider(
    "Max tokens", 100, 20000, 2000,
    help="Caps the length of an LLM-written answer. Only used when the LLM is called (needs ANTHROPIC_API_KEY).",
)

MAX_QUESTIONS_PER_SESSION = int(os.environ.get("ARAG_MAX_QUESTIONS_PER_SESSION", "10"))
asked = st.session_state.get("questions_asked", 0)

# allow_fallback toggle: evidence-only mode disables LLM fallback so answers
# come strictly from retrieved corpus snippets. In the other mode the checkbox
# controls whether the LLM may fall back to external knowledge. It is rendered
# outside the Run block so toggling it does not clear the results.
if mode.startswith("evidence-only"):
    allow_fallback = False
    st.info("Evidence-only mode: answers are built strictly from retrieved snippets (no LLM fallback).")
else:
    allow_fallback = st.checkbox(
        "Allow LLM fallback to external knowledge if corpus lacks answer", value=True
    )

if st.button("Run"):
    if asked >= MAX_QUESTIONS_PER_SESSION:
        st.error(f"Session limit reached ({MAX_QUESTIONS_PER_SESSION} questions). Reload the page to start a new session.")
        st.stop()
    st.session_state["questions_asked"] = asked + 1
    with st.spinner("Running agent..."):
        # For the demo UI we enable the director/award flow so sample movie
        # queries work out-of-the-box. Programmatic callers default to
        # enable_director_flow=False unless they opt in.
        st.session_state["result"] = run_agent(
            question,
            max_steps=max_steps,
            max_tokens=max_tokens,
            allow_fallback=allow_fallback,
            enable_director_flow=True,
        )

# Results come from session state so they survive reruns (e.g. ticking the box).
res = st.session_state.get("result")
if res is not None:
    _top_banner.empty()  # the banner moves down, directly above the answer
    _vs = res.get("vector_search") or {"enabled": True, "reason": ""}
    if not _vs["enabled"]:
        st.warning(degraded_line(_vs["reason"]))  # never show a keyword-only answer without saying so
    if res.get("budget_exhausted"):
        st.warning(f"Step budget exhausted after {res.get('tool_calls')} tool calls: the answer may be incomplete. Raise Max steps to allow more.")
    st.subheader("Answer")
    st.write(res.get("answer"))
    # If the agent returned an explicit provenance note about fallback, make it visible
    if isinstance(res.get("answer"), str) and "external knowledge" in res.get("answer"):
        st.warning("This answer used external knowledge beyond the retrieved corpus.")

    st.subheader("Cited snippets")
    # Prefer explicit citations returned by the agent (1-based indices).
    citations = res.get("citations") or res.get("citations_list") or []
    retrieved = res.get("retrieved_texts", [])

    from urllib.parse import quote

    def build_source_url(entry: dict) -> str | None:
        # Prefer explicit URL fields if present
        for key in ("url", "source_url", "link"):
            if entry.get(key):
                return entry.get(key)
        # Try to build a Wikipedia link from the `source` or `chunk_id`
        candidate = entry.get("source") or entry.get("chunk_id")
        if not candidate:
            return None
        # Normalize to a wiki path: replace spaces with underscores
        path = candidate.replace(" ", "_")
        # Quote the path to be safe for parentheses and other chars
        path_quoted = quote(path, safe="()!$&'()*+,;=:@/[]~")
        return f"https://en.wikipedia.org/wiki/{path_quoted}"

    def render_entry_with_link(label: str, entry: dict, idx_str: str | None = None):
        url = build_source_url(entry)
        source = entry.get("source") or entry.get("chunk_id") or "source"
        label_text = f"[{idx_str}] {source}" if idx_str else source
        if url:
            # Show clickable link above the expander
            st.markdown(f"**[{label_text}]({url})**")
        else:
            st.markdown(f"**{label_text}**")
        with st.expander(source):
            st.write(entry.get("text") or entry.get("full_context") or "")

    if citations:
        # Render only the cited snippets in order
        for c in citations:
            try:
                idx = int(c) - 1
            except Exception:
                continue
            if idx < 0 or idx >= len(retrieved):
                continue
            entry = retrieved[idx]
            render_entry_with_link(str(c), entry, idx_str=str(c))
    else:
        # Fallback: show all retrieved_texts
        for i, entry in enumerate(retrieved, start=1):
                render_entry_with_link(str(i), entry, idx_str=str(i))
                # show a compact metadata line under each snippet
                meta = []
                if entry.get("chunk_id"):
                    meta.append(f"chunk_id={entry.get('chunk_id')}")
                if entry.get("source"):
                    meta.append(f"source={entry.get('source')}")
                if meta:
                    st.caption(" | ".join(meta))

    st.subheader("Tool trace")
    for step in res.get("tool_trace", []):
        st.json(step)
    st.write("Step count:", res.get("step_count"), "Token count:", res.get("token_count"))

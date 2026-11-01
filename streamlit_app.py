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

mode = st.selectbox("Mode", ["evidence-only (no LLM)", "evidence + LLM fallback (LLM only if the corpus gives no verdict)"])
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

LLM_ANSWER_LABEL = "LLM answer (model knowledge, not from the corpus)"

ROLE_LABEL = {
    "director": "supports: the director",
    "verdict": "supports: the Oscar verdict",
    "career": "supports: the career note (a win for another film)",
    "cited": "cited by the answer",
}


def md_escape(text: str) -> str:
    """Escape characters Streamlit would render as markdown or LaTeX (e.g. the $ in "$975 million")."""
    for ch in ("\\", "$", "*", "_", "`", "#", "[", "]", "<", ">"):
        text = text.replace(ch, "\\" + ch)
    return text


def build_source_url(entry: dict) -> str | None:
    from urllib.parse import quote

    for key in ("url", "source_url", "link"):  # an explicit URL wins
        if entry.get(key):
            return entry.get(key)
    candidate = entry.get("source") or entry.get("chunk_id")
    if not candidate:
        return None
    return "https://en.wikipedia.org/wiki/" + quote(candidate.replace(" ", "_"), safe="()!$&'()*+,;=:@/[]~")


def render_llm_status(res: dict) -> None:
    """Say what the LLM did for this answer: called, skipped (why), or failed (error)."""
    llm = res.get("llm") or {}
    decision = llm.get("decision")
    if decision == "called":
        usage = (
            f"{llm.get('input_tokens')} input / {llm.get('output_tokens')} output tokens"
            if llm.get("input_tokens") is not None
            else "token usage not reported"
        )
        st.info(f"LLM called ({llm.get('model') or 'model unknown'}; {usage}). The answer above is the LLM's and may use knowledge from outside the corpus.")
    elif decision == "failed":
        st.error(f"LLM call failed: {llm.get('error')}. Showing the corpus-evidence answer instead.")
    elif decision == "skipped":
        st.caption("LLM not called: " + md_escape(llm.get("reason") or "no reason recorded"))
    else:
        st.caption("LLM not considered for this question.")


def render_evidence(res: dict) -> None:
    st.subheader("Evidence")
    details = res.get("citation_details")
    if details:
        for d in details:
            title = d.get("source") or d.get("chunk_id") or "source"
            url = build_source_url({"source": d.get("source"), "chunk_id": d.get("chunk_id")})
            label = f"[{d['citation']}] {title}"
            st.markdown(f"**[{label}]({url})**" if url else f"**{label}**")
            for sp in d["supports"]:
                st.markdown(f"*{ROLE_LABEL.get(sp['role'], sp['role'])}*")
                if sp.get("sentence"):
                    if sp.get("before"):
                        st.caption(md_escape(sp["before"]))
                    st.markdown("> **" + md_escape(sp["sentence"]) + "**")
                    if sp.get("after"):
                        st.caption(md_escape(sp["after"]))
                else:
                    st.caption("(the answer cites this passage without naming a single sentence)")
            with st.expander("Show full passage", expanded=False):
                st.write(d.get("full_text") or "")
    else:
        for i, entry in enumerate(res.get("retrieved_texts", []), start=1):
            title = entry.get("source") or entry.get("chunk_id") or "source"
            url = build_source_url(entry)
            st.markdown(f"**[[{i}] {title}]({url})**" if url else f"**[{i}] {title}**")
            with st.expander("Show full passage", expanded=False):
                st.write(entry.get("text") or entry.get("full_context") or "")


# Results come from session state so they survive reruns (e.g. ticking the box).
res = st.session_state.get("result")
if res is not None:
    _top_banner.empty()  # the banner moves down, directly above the answer
    _vs = res.get("vector_search") or {"enabled": True, "reason": ""}
    if not _vs["enabled"]:
        st.warning(degraded_line(_vs["reason"]))  # never show a keyword-only answer without saying so
    if res.get("budget_exhausted"):
        st.warning(f"Step budget exhausted after {res.get('tool_calls')} tool calls: the answer may be incomplete. Raise Max steps to allow more.")
    if (res.get("llm") or {}).get("decision") == "called":
        # The LLM answered: keep its answer visibly apart from the answer built from the corpus.
        st.subheader("Evidence answer (from the corpus)")
        st.write(res.get("template_answer") or "No answer was built from the corpus for this question; see the passages under Evidence.")
        st.subheader(LLM_ANSWER_LABEL)
        with st.container(border=True):
            st.write(res.get("answer"))
        render_llm_status(res)
    else:
        st.subheader("Answer")
        st.write(res.get("answer"))
        render_llm_status(res)
    # If the agent returned an explicit provenance note about fallback, make it visible
    if isinstance(res.get("answer"), str) and "external knowledge" in res.get("answer"):
        st.warning("This answer used external knowledge beyond the retrieved corpus.")

    if res.get("no_specialised_route"):
        with st.expander("Related passages (not an answer)", expanded=False):
            for i, entry in enumerate(res.get("related_passages") or [], start=1):
                title = entry.get("source") or entry.get("chunk_id") or "source"
                url = build_source_url(entry)
                st.markdown(f"**[[{i}] {title}]({url})**" if url else f"**[{i}] {title}**")
                st.caption(md_escape(entry.get("text") or ""))
    else:
        render_evidence(res)

    with st.expander("Tool trace", expanded=False):
        for step in res.get("tool_trace", []):
            st.json(step)
    st.write("Step count:", res.get("step_count"), "Token count:", res.get("token_count"))

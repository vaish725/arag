"""Simple Streamlit demo for A-RAG pipeline.

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from arag.agent_loop import run_agent

st.title("A-RAG Demo")

mode = st.selectbox("Mode", ["evidence-only (no LLM)", "semantic + BM25 + optional LLM fallback"])
question = st.text_input(
    "Question", "",
    placeholder="Ask something like: Who directed Inception and did they win any Academy Awards?",
)
max_steps = st.slider("Max steps", 1, 20, 6)
max_tokens = st.slider("Max tokens", 100, 20000, 2000)

if st.button("Run"):
    # allow_fallback toggle: evidence-only mode disables LLM fallback so
    # answers come strictly from retrieved corpus snippets. In the other
    # mode the checkbox controls whether the LLM may fall back to external
    # knowledge when the corpus doesn't contain the answer.
    if mode.startswith("evidence-only"):
        allow_fallback = False
        st.info("Evidence-only mode: answers are built strictly from retrieved snippets (no LLM fallback).")
    else:
        allow_fallback = st.checkbox(
            "Allow LLM fallback to external knowledge if corpus lacks answer", value=True
        )
    with st.spinner("Running agent..."):
        # For the demo UI we enable the director/award flow so sample movie
        # queries work out-of-the-box. Programmatic callers default to
        # enable_director_flow=False unless they opt in.
        res = run_agent(
            question,
            max_steps=max_steps,
            max_tokens=max_tokens,
            allow_fallback=allow_fallback,
            enable_director_flow=True,
        )
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

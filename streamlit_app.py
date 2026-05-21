"""Simple Streamlit demo for A-RAG pipeline.

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from arag.agent_loop import run_agent

st.title("A-RAG Demo")

mode = st.selectbox("Mode", ["heuristic-only", "semantic+bm25+llm"])
question = st.text_input(
    "Question", "Who directed Doctor Strange and did they win any Academy Awards?"
)
max_steps = st.slider("Max steps", 1, 20, 6)
max_tokens = st.slider("Max tokens", 100, 20000, 2000)

if st.button("Run"):
    allow_fallback = mode == "heuristic-only"
    with st.spinner("Running agent..."):
        res = run_agent(
            question, max_steps=max_steps, max_tokens=max_tokens, allow_fallback=True
        )
    st.subheader("Answer")
    st.write(res.get("answer"))
    st.subheader("Tool trace")
    for step in res.get("tool_trace", []):
        st.json(step)
    st.write(
        "Step count:", res.get("step_count"), "Token count:", res.get("token_count")
    )

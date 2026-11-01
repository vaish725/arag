#!/usr/bin/env python3
"""Measure model load time and memory the way the Streamlit app uses the model.

Run each mode in its own fresh process:
  --mode manual     import streamlit, warm up the embedding model (as streamlit_app.py
                    does via st.cache_resource), embed one question, load the indexes,
                    run one full run_agent call
  --mode streamlit  run streamlit_app.py itself with streamlit's AppTest and click Run
Reports RSS after each phase and the process's peak RSS. Linux/macOS. Makes no API calls.

Usage: PYTHONPATH=. python scripts/measure_app_memory.py --mode manual
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

QUESTION = "Who directed Inception and did they win any Academy Awards?"


def rss_mb() -> float:
    try:  # Linux: current resident set size
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def peak_mb() -> float:
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def phase(name, fn):
    t = time.time()
    try:
        out = fn()
        status = "ok"
    except Exception as e:  # keep measuring later phases
        out, status = None, f"FAILED {type(e).__name__}: {e}"
    print(f"{name:42} {time.time() - t:7.2f}s   rss {rss_mb():7.0f} MB   peak {peak_mb():7.0f} MB   {status}", flush=True)
    return out


def manual() -> None:
    phase("import streamlit + arag", lambda: (__import__("streamlit"), __import__("arag.agent_loop")))
    from arag import embedding_backend as eb
    from arag import retrieval_tools as rt

    phase("load embedding model (warm_up)", eb.warm_up)
    phase("embed one question (query path)", lambda: eb.get_embedding(QUESTION))
    phase("load indexes (BM25, FAISS x2, pickles)", rt.load_indexes)
    phase("model fingerprint (sha256 of model files)", eb.get_model_fingerprint)

    def agent():
        from arag import agent_loop

        res = agent_loop.run_agent(QUESTION, allow_fallback=False, enable_director_flow=True)
        sem = [t for t in res["tool_trace"] if t["tool"] == "semantic_search"]
        assert sem and not any("error" in t for t in sem), "semantic_search did not succeed"
        return res

    phase("one full run_agent call (Inception)", agent)


def streamlit_mode() -> None:
    from streamlit.testing.v1 import AppTest

    at = phase("start app (imports, model warm-up)", lambda: AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=300).run())
    if at is None or at.exception:
        print("app raised:", at.exception if at is not None else "n/a")
        return
    at.selectbox[0].select("evidence + LLM fallback (LLM only if the corpus gives no verdict)")
    at.text_input[0].set_value(QUESTION)
    at = phase("click Run (one full question)", lambda: at.button[0].click().run())
    shown = " ".join(m.value for m in at.markdown) if at is not None else ""
    print("answer shown in app:", "Christopher Nolan" in shown, "| exceptions:", list(at.exception) if at is not None else "n/a")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["manual", "streamlit"], required=True)
    args = ap.parse_args()
    print(f"mode={args.mode}  python={sys.version.split()[0]}")
    (manual if args.mode == "manual" else streamlit_mode)()
    print(f"FINAL peak RSS: {peak_mb():.0f} MB")

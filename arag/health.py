"""Visible degraded mode and the startup health line.

When vector ("meaning") search is off, the app and the CLI must say so, next to every answer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

log = logging.getLogger("arag.health")

DEGRADED_PREFIX = "Meaning search unavailable: keyword-only mode. Reason: "


def degraded_line(reason: str) -> str:
    """The one wording used by the Streamlit banner and the CLI stderr line."""
    return DEGRADED_PREFIX + (reason or "unknown")


def short_reason(message: str, limit: int = 160) -> str:
    """First sentence of an error message, one line, at most ``limit`` characters."""
    text = re.sub(r"\s+", " ", message or "").strip()
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0] if text else ""
    return first if len(first) <= limit else first[: limit - 3].rstrip() + "..."


def runtime_status(trace: Any, static: tuple) -> Dict[str, Any]:
    """Vector-search status for one answer.

    ``static`` is (enabled, reason) from retrieval_tools.vector_search_status(). Even when the
    indexes and model look fine, a semantic_search call that failed during the run (visible in
    the tool trace) also counts as degraded.
    """
    enabled, reason = static
    if enabled:
        for step in trace or []:
            if isinstance(step, dict) and step.get("tool") == "semantic_search" and step.get("error") and not step.get("blocked"):
                return {"enabled": False, "reason": "semantic_search failed during this run: " + short_reason(str(step["error"]), 110)}
    return {"enabled": bool(enabled), "reason": reason}


def health_line() -> str:
    """One line: provider, model, revision, index dimensions, counts, vector search ON/OFF."""
    from arag import embedding_backend as eb
    from arag import retrieval_tools as rt

    info = eb.get_embedding_info()
    enabled, reason = rt.vector_search_status()
    revision = "n/a"
    try:
        fingerprint = eb.get_model_fingerprint()
        if fingerprint:
            revision = fingerprint["revision"][:8]
    except Exception:
        revision = "unavailable"
    chunk_dim = rt._faiss_index.d if rt._faiss_index is not None else "-"
    sent_dim = rt._sentence_faiss_index.d if rt._sentence_faiss_index is not None else "-"
    state = "ON" if enabled else f"OFF (reason: {reason})"
    return (
        f"A-RAG ready: embedding={info['provider']}/{info['model']} revision={revision} "
        f"dims(chunk/sentence)={chunk_dim}/{sent_dim} chunks={len(rt._chunks or [])} "
        f"sentences={len(rt._sentences or [])} vector_search={state}"
    )


_last_logged: Optional[str] = None


def log_health() -> Optional[str]:
    """Log the health line (INFO, logger "arag.health"); never raises.

    Indexes are re-read on every question, so an unchanged line is not logged again: one line at
    startup, and a new one only when something changed (for example vector search went OFF).
    """
    global _last_logged
    try:
        line = health_line()
    except Exception as e:  # a broken health line must not break loading
        line = f"A-RAG ready: health line unavailable ({type(e).__name__}: {e})"
    if line != _last_logged:
        log.info(line)
        _last_logged = line
    return line

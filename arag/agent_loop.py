"""Agent loop (Phase 3) moved into arag package.

This file is copied from the top-level `agent_loop.py` and adjusted to use
relative imports for package internal modules.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

# retrieval_tools symbols may not be available at import-time; declare
# typed optional placeholders which will be replaced at runtime if available.
chunk_read: Optional[Callable[[str], Dict[str, Any]]] = None
keyword_search: Optional[Callable[..., List[Dict[str, Any]]]] = None
semantic_search: Optional[Callable[..., List[Dict[str, Any]]]] = None
try:
    import importlib

    _rt = importlib.import_module("arag.retrieval_tools")
    chunk_read = getattr(_rt, "chunk_read", None)
    keyword_search = getattr(_rt, "keyword_search", None)
    semantic_search = getattr(_rt, "semantic_search", None)
except Exception:
    chunk_read = None
    keyword_search = None
    semantic_search = None


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+", text)))


def _safe_source(cr: Any) -> str:
    if not isinstance(cr, dict):
        return ""
    return str(cr.get("source") or "")


def _safe_text(cr: Any) -> str:
    if not isinstance(cr, dict):
        return ""
    return str(cr.get("full_context") or "")


def _call_chunk_read(chunk_id: Optional[str]) -> Dict[str, Any]:
    if not isinstance(chunk_id, str):
        return {}
    if not callable(chunk_read):
        return {}
    try:
        return chunk_read(chunk_id) or {}
    except Exception:
        return {}


def run_agent(
    question: str,
    max_steps: int = 10,
    max_tokens: int = 2000,
    allow_fallback: bool = True,
) -> Dict[str, Any]:
    q = question or ""
    if re.search(r"\bdirector\b|\bdirected\b", q, re.IGNORECASE):
        retrieval = two_hop_director_award_flow(q)
        contexts = [t.get("text", "") for t in retrieval.get("retrieved_texts", [])]
        answer = ""
        try:
            from . import llm_backend

            if llm_backend.llm_available():
                # Prefer context-first answers: only allow fallback when there are
                # no retrieved contexts available.
                allow_fallback_for_call = allow_fallback and not bool(contexts)
                res = llm_backend.synthesize_answer(
                    q, contexts, max_tokens=max_tokens, allow_fallback=allow_fallback_for_call
                )
                # synthesize_answer returns (answer, citations_list, used_fallback)
                if isinstance(res, tuple) and len(res) == 3:
                    answer_text, citations_list, used_fallback = res
                    provenance_note = (
                        "Note: this answer uses external knowledge outside the provided contexts."
                        if used_fallback
                        else ""
                    )
                    if provenance_note:
                        answer_text = f"{answer_text} {provenance_note}"
                    answer = answer_text
                else:
                    # Unexpected return format, fall back to raw string
                    answer = str(res)
            else:
                answer = "\n\n".join(contexts)
        except Exception:
            answer = "\n\n".join(contexts)

        out = {
            "answer": answer,
            "tool_trace": retrieval.get("tool_trace", []),
            "retrieved_texts": retrieval.get("retrieved_texts", []),
            "step_count": retrieval.get("step_count", 0),
            "token_count": retrieval.get("token_count", 0),
            "budget_exhausted": False,
        }
        return out

    return {
        "answer": "",
        "tool_trace": [],
        "retrieved_texts": [],
        "step_count": 0,
        "token_count": 0,
        "budget_exhausted": False,
    }


def two_hop_generic(
    question: str,
    first_hop_extract_regex: str | None = None,
    first_hop_queries: List[str] | None = None,
    second_hop_queries: List[str] | None = None,
    max_first_results: int = 6,
    max_second_results: int = 6,
) -> Dict[str, Any]:
    import re

    trace: List[Dict[str, Any]] = []
    retrieved_texts: List[Dict[str, str]] = []
    local_token_count = 0
    local_steps = 0

    work: Optional[str] = None
    if first_hop_extract_regex:
        m = re.search(first_hop_extract_regex, question, re.IGNORECASE)
        if m:
            work = m.group(1).strip()

    queries_first: List[str] = []
    if first_hop_queries:
        for t in first_hop_queries:
            if "{work}" in t:
                if work:
                    queries_first.append(t.format(work=work))
            else:
                queries_first.append(t)
    else:
        queries_first = [question]

    first_results: List[Dict[str, Any]] = []
    for q in queries_first:
        res: List[Dict[str, Any]] = []
        if semantic_search is not None:
            try:
                res = semantic_search(q, k=max_first_results)
                trace.append(
                    {
                        "tool": "semantic_search",
                        "args": {"query": q, "k": max_first_results},
                        "raw_results": res,
                    }
                )
            except Exception as ex:
                trace.append(
                    {"tool": "semantic_search", "args": {"query": q}, "error": str(ex)}
                )
                res = []
        if not res and keyword_search is not None:
            try:
                res = keyword_search(q, k=max_first_results)
                trace.append(
                    {
                        "tool": "keyword_search",
                        "args": {"query": q, "k": max_first_results},
                        "raw_results": res,
                    }
                )
            except Exception as ex:
                trace.append(
                    {"tool": "keyword_search", "args": {"query": q}, "error": str(ex)}
                )
                res = []
        if res:
            first_results.extend(res)

    seen_ids = set()
    deduped: List[Dict[str, Any]] = []
    for r in first_results:
        cid = r.get("chunk_id")
        if cid and cid not in seen_ids:
            deduped.append(r)
            seen_ids.add(cid)
    if not deduped:
        return {
            "tool_trace": trace,
            "retrieved_texts": retrieved_texts,
            "step_count": local_steps,
            "token_count": local_token_count,
        }

    name: Optional[str] = None
    for candidate in deduped[:max_first_results]:
        cid = candidate.get("chunk_id")
        if not isinstance(cid, str):
            continue
        cr = _call_chunk_read(cid)
        summary = _safe_text(cr)[:400]
        trace.append(
            {"tool": "chunk_read", "args": {"chunk_id": cid}, "result_summary": summary}
        )
        if isinstance(cr, dict) and "full_context" in cr:
            text = _safe_text(cr)
            retrieved_texts.append(
                {"chunk_id": cid, "source": _safe_source(cr), "text": text}
            )
            local_token_count += approx_token_count(text)
            local_steps += 1
            name = extract_person_name_from_text(text)
            if name:
                break

    if not name:
        return {
            "tool_trace": trace,
            "retrieved_texts": retrieved_texts,
            "step_count": local_steps,
            "token_count": local_token_count,
        }

    person = name.strip()
    safe = person.replace(" ", "_")
    person_page_ids = [
        safe,
        f"{safe}_0",
        f"{safe}_1",
        f"List_of_awards_and_nominations_received_by_{safe}",
        f"Awards_and_nominations_received_by_{safe}",
    ]
    for pid in person_page_ids:
        if not callable(chunk_read):
            continue
        cr = _call_chunk_read(pid)
        if not cr:
            continue
        if isinstance(cr, dict) and "full_context" in cr:
            trace.append(
                {
                    "tool": "chunk_read",
                    "args": {"chunk_id": pid},
                    "result_summary": (cr.get("full_context") or "")[:400],
                }
            )
            retrieved_texts.append(
                {"chunk_id": pid, "source": _safe_source(cr), "text": _safe_text(cr)}
            )
            local_token_count += approx_token_count(cr.get("full_context") or "")
            local_steps += 1
            return {
                "tool_trace": trace,
                "retrieved_texts": retrieved_texts,
                "step_count": local_steps,
                "token_count": local_token_count,
            }

    second_templates = second_hop_queries or ["{person} awards", "{person} Oscar"]
    for tmpl in second_templates:
        q = tmpl.format(person=person)
        res = []
        if semantic_search is not None:
            try:
                res = semantic_search(q, k=max_second_results)
                trace.append(
                    {
                        "tool": "semantic_search",
                        "args": {"query": q, "k": max_second_results},
                        "raw_results": res,
                    }
                )
            except Exception as ex:
                trace.append(
                    {"tool": "semantic_search", "args": {"query": q}, "error": str(ex)}
                )
                res = []
        if not res and keyword_search is not None:
            try:
                res = keyword_search(q, k=max_second_results)
                trace.append(
                    {
                        "tool": "keyword_search",
                        "args": {"query": q, "k": max_second_results},
                        "raw_results": res,
                    }
                )
            except Exception as ex:
                trace.append(
                    {"tool": "keyword_search", "args": {"query": q}, "error": str(ex)}
                )
                res = []

        if not res:
            continue

        parts = [p for p in person.split() if p]
        last_name = parts[-1].lower() if parts else None
        prioritized = []
        others = []
        for r in res:
            cid2 = (r.get("chunk_id") or "").lower()
            txt2 = (r.get("text") or "").lower()
            if last_name and (last_name in cid2 or last_name in txt2):
                prioritized.append(r)
            else:
                others.append(r)

        for r in (prioritized + others)[:max_second_results]:
            cid2 = r.get("chunk_id")
            if not isinstance(cid2, str):
                continue
            if not callable(chunk_read):
                continue
            cr2 = _call_chunk_read(cid2)
            summary2 = (
                (cr2.get("full_context") or "")[:400]
                if isinstance(cr2, dict)
                else str(cr2)[:400]
            )
            trace.append(
                {
                    "tool": "chunk_read",
                    "args": {"chunk_id": cid2},
                    "result_summary": summary2,
                }
            )
            if isinstance(cr2, dict) and "full_context" in cr2:
                retrieved_texts.append(
                    {
                        "chunk_id": cid2,
                        "source": _safe_source(cr2),
                        "text": _safe_text(cr2),
                    }
                )
                local_token_count += approx_token_count(cr2["full_context"])
                local_steps += 1
        if retrieved_texts:
            break

    return {
        "tool_trace": trace,
        "retrieved_texts": retrieved_texts,
        "step_count": local_steps,
        "token_count": local_token_count,
    }


def extract_person_name_from_text(text: str) -> str | None:
    import re

    m = re.search(r"directed by ([A-Z][A-Za-z\-]+(?: [A-Z][A-Za-z\-]+)*)", text)
    if m:
        return m.group(1)

    m = re.search(r"film directed by ([A-Z][A-Za-z\-]+(?: [A-Z][A-Za-z\-]+)*)", text)
    if m:
        return m.group(1)

    tokens = text.split()
    runs = []
    current = []
    for t in tokens:
        if t.istitle() and len(t) > 1:
            current.append(t.strip(".,"))
        else:
            if current:
                runs.append(" ".join(current))
                current = []
    if current:
        runs.append(" ".join(current))
    if not runs:
        return None
    return max(runs, key=lambda r: len(r.split()))


def detect_award_evidence(
    entries: List[Dict[str, str]], name_hint: str | None = None
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    import re

    patterns_win = [
        r"won (an |the )?academy award",
        r"academy award winner",
        r"won an oscar",
        r"oscar winner",
        r"received an academy award",
    ]
    patterns_nom = [
        r"nominated for (an |the )?academy award",
        r"nomination for (an |the )?academy award",
        r"nominated for an oscar",
    ]
    last_name = None
    if name_hint:
        parts = [p for p in name_hint.split() if p]
        if parts:
            last_name = parts[-1].lower()

    for idx, e in enumerate(entries):
        raw_text = e.get("text") or ""
        text = raw_text.lower()
        cid = (e.get("chunk_id") or "").lower()
        src = (e.get("source") or "").lower()

        def is_awards_page_identifier(cid_str: str, src_str: str) -> bool:
            if not cid_str and not src_str:
                return False
            checks = [
                "list_of_awards_and_nominations_received_by_",
                "awards_and_nominations_received_by_",
                "list_of_awards_and_nominations",
                "awards_and_nominations",
                "list_of_awards",
                "award",
            ]
            for p in checks:
                if p in cid_str or p in src_str:
                    return True
            return False

        if last_name:
            has_name_in_id_or_src = (last_name in cid) or (last_name in src)
            awards_page = is_awards_page_identifier(cid, src)
            if not (has_name_in_id_or_src or awards_page):
                continue
        else:
            if not is_awards_page_identifier(cid, src):
                continue

        matched = False
        for p in patterns_win:
            if re.search(p, text):
                evidence.append(
                    {
                        "entry_index": idx,
                        "chunk_id": e.get("chunk_id"),
                        "source": e.get("source"),
                        "text": raw_text,
                        "type": "win",
                        "pattern": p,
                    }
                )
                matched = True
                break
        if matched:
            continue
        for p in patterns_nom:
            if re.search(p, text):
                evidence.append(
                    {
                        "entry_index": idx,
                        "chunk_id": e.get("chunk_id"),
                        "source": e.get("source"),
                        "text": raw_text,
                        "type": "nom",
                        "pattern": p,
                    }
                )
                break
    return evidence


def two_hop_director_award_flow(question: str) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = []
    retrieved_texts: List[Dict[str, str]] = []
    local_token_count = 0
    local_steps = 0

    import re

    m = re.search(r"director of ([A-Za-z0-9\'\- ]+?)(\?|$)", question, re.IGNORECASE)
    work = None
    if m:
        work = m.group(1).strip()
    else:
        m = re.search(
            r"who directed ([A-Za-z0-9\'\- ]+?)(\?|$)", question, re.IGNORECASE
        )
        if m:
            work = m.group(1).strip()

    if not work:
        return {
            "tool_trace": trace,
            "retrieved_texts": retrieved_texts,
            "step_count": local_steps,
            "token_count": local_token_count,
        }

    director_query_variants = [
        f"director of {work}",
        f"who directed {work}",
        f"{work} (film) director",
        f"{work} director",
        f"{work} film director",
    ]

    first_results = []
    for q in director_query_variants:
        res = []
        if semantic_search is not None:
            try:
                res = semantic_search(q, k=5)
                trace.append(
                    {
                        "tool": "semantic_search",
                        "args": {"query": q, "k": 5},
                        "raw_results": res,
                    }
                )
            except Exception as e:
                trace.append(
                    {
                        "tool": "semantic_search",
                        "args": {"query": q, "k": 5},
                        "error": str(e),
                    }
                )
                res = []
        if not res and keyword_search is not None:
            try:
                res = keyword_search(q, k=5)
                trace.append(
                    {
                        "tool": "keyword_search",
                        "args": {"query": q, "k": 5},
                        "raw_results": res,
                    }
                )
            except Exception as e:
                trace.append(
                    {
                        "tool": "keyword_search",
                        "args": {"query": q, "k": 5},
                        "error": str(e),
                    }
                )
                res = []
        if res:
            first_results.extend(res)
    seen_ids = set()
    deduped = []
    for r in first_results:
        cid = r.get("chunk_id")
        if cid and cid not in seen_ids:
            deduped.append(r)
            seen_ids.add(cid)
    if not deduped:
        return {
            "tool_trace": trace,
            "retrieved_texts": retrieved_texts,
            "step_count": local_steps,
            "token_count": local_token_count,
        }

    name = None
    inspected = 0
    for candidate in deduped[:6]:
        cid = candidate.get("chunk_id")
        if not isinstance(cid, str):
            continue
        cr = _call_chunk_read(cid)
        summary = _safe_text(cr)[:400]
        trace.append(
            {"tool": "chunk_read", "args": {"chunk_id": cid}, "result_summary": summary}
        )
        if isinstance(cr, dict) and "full_context" in cr:
            text = _safe_text(cr)
            retrieved_texts.append(
                {"chunk_id": cid, "source": _safe_source(cr), "text": text}
            )
            local_token_count += approx_token_count(text)
            local_steps += 1
            name = extract_person_name_from_text(text)
            if name:
                break
        inspected += 1
    if not name:
        return {
            "tool_trace": trace,
            "retrieved_texts": retrieved_texts,
            "step_count": local_steps,
            "token_count": local_token_count,
        }

    person = name.strip()
    person_page_ids = [
        person.replace(" ", "_"),
        f"{person.replace(' ', '_')}_0",
        f"{person.replace(' ', '_')}_1",
        f"List_of_awards_and_nominations_received_by_{person.replace(' ', '_')}",
        f"Awards_and_nominations_received_by_{person.replace(' ', '_')}",
    ]
    for pid in person_page_ids:
        cr = _call_chunk_read(pid)
        if not cr:
            continue
        if isinstance(cr, dict) and "full_context" in cr:
            trace.append(
                {
                    "tool": "chunk_read",
                    "args": {"chunk_id": pid},
                    "result_summary": _safe_text(cr)[:400],
                }
            )
            retrieved_texts.append(
                {"chunk_id": pid, "source": _safe_source(cr), "text": _safe_text(cr)}
            )
            local_token_count += approx_token_count(_safe_text(cr))
            local_steps += 1
            return {
                "tool_trace": trace,
                "retrieved_texts": retrieved_texts,
                "step_count": local_steps,
                "token_count": local_token_count,
            }

    award_queries = [
        f"{person} Academy Awards",
        f"{person} Oscar",
        f"{person} awards",
        f"{person} award",
        f"List of awards and nominations received by {person}",
        f"{person} awards and nominations",
    ]
    for q in award_queries:
        res = []
        if semantic_search is not None:
            try:
                res = semantic_search(q, k=6)
                trace.append(
                    {
                        "tool": "semantic_search",
                        "args": {"query": q, "k": 6},
                        "raw_results": res,
                    }
                )
            except Exception as e:
                trace.append(
                    {
                        "tool": "semantic_search",
                        "args": {"query": q, "k": 6},
                        "error": str(e),
                    }
                )
                res = []
        if not res and keyword_search is not None:
            try:
                res = keyword_search(q, k=6)
                trace.append(
                    {
                        "tool": "keyword_search",
                        "args": {"query": q, "k": 6},
                        "raw_results": res,
                    }
                )
            except Exception as e:
                trace.append(
                    {
                        "tool": "keyword_search",
                        "args": {"query": q, "k": 6},
                        "error": str(e),
                    }
                )
                res = []

        if not res:
            continue
        last_name = None
        parts = [p for p in name.split() if p]
        if parts:
            last_name = parts[-1].lower()

        prioritized = []
        others = []
        for r in res:
            cid = (r.get("chunk_id") or "").lower()
            txt = (r.get("text") or "").lower()
            if last_name and (last_name in cid or last_name in txt):
                prioritized.append(r)
            else:
                others.append(r)

        for r in (prioritized + others)[:6]:
            cid2 = r.get("chunk_id")
            if not isinstance(cid2, str):
                continue
            cr2 = _call_chunk_read(cid2)
            if not cr2:
                continue
            summary2 = _safe_text(cr2)[:400]
            trace.append(
                {
                    "tool": "chunk_read",
                    "args": {"chunk_id": cid2},
                    "result_summary": summary2,
                }
            )
            if isinstance(cr2, dict) and "full_context" in cr2:
                retrieved_texts.append(
                    {
                        "chunk_id": cid2,
                        "source": _safe_source(cr2),
                        "text": _safe_text(cr2),
                    }
                )
                local_token_count += approx_token_count(_safe_text(cr2))
                local_steps += 1
        if retrieved_texts:
            break

    return {
        "tool_trace": trace,
        "retrieved_texts": retrieved_texts,
        "step_count": local_steps,
        "token_count": local_token_count,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the lightweight A-RAG agent (heuristic policy)"
    )
    parser.add_argument(
        "--question", type=str, default="Who directed Titanic?", help="Question to ask"
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=4000)
    args = parser.parse_args()

    result = run_agent(
        args.question,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        allow_fallback=True,
    )
    print(json.dumps(result, indent=2))

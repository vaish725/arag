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
    local_citations: List[int] = []
    if re.search(r"\bdirector\b|\bdirected\b", q, re.IGNORECASE):
        retrieval = two_hop_director_award_flow(q)
        contexts = [t.get("text", "") for t in retrieval.get("retrieved_texts", [])]

        # Targeted fallback: when the caller forbids LLM fallback (evidence-only)
        # and the question contains a work/title, try a focused semantic search
        # for "{work} director" and promote the returned film chunk into the
        # front of the retrieved contexts so the extractor sees the film page.
        try:
            work_hint = None
            # Capture the work title but stop before conjunctions like " and ...",
            # commas, or the end of the sentence so we don't grab trailing clauses
            m = re.search(r"who directed\s+([A-Za-z0-9'\- ]+?)(?:\s+and\b|,|\?|$)", q, re.IGNORECASE)
            if not m:
                m = re.search(r"director of\s+([A-Za-z0-9'\- ]+?)(?:\s+and\b|,|\?|$)", q, re.IGNORECASE)
            if m:
                work_hint = m.group(1).strip()

            if work_hint and not allow_fallback and semantic_search is not None and callable(chunk_read):
                # If none of the current contexts explicitly include both the
                # work title and an explicit "directed by" phrase, promote the
                # film chunk found by a targeted search so the extractor sees it.
                need_promote = True
                try:
                    for t in contexts:
                        if not t:
                            continue
                        lt = t.lower()
                        if work_hint.lower() in lt and "directed by" in lt:
                            need_promote = False
                            break
                except Exception:
                    need_promote = True
                if need_promote:
                    try:
                        # Try stronger targeted queries first to find the film page
                        targeted_queries = [
                            f"{work_hint} written and directed by",
                            f"{work_hint} written & directed by",
                            f"{work_hint} directed by",
                            f"{work_hint} director",
                            f"{work_hint} (film) director",
                        ]
                        sem_res = []
                        for tq in targeted_queries:
                            try:
                                sem_res = semantic_search(tq, k=6)
                                retrieval.setdefault("tool_trace", []).append({"tool": "semantic_search", "args": {"query": tq, "k": 6}, "raw_results": sem_res})
                            except Exception:
                                sem_res = []
                            if sem_res:
                                break
                        # attach trace for debugging
                        retrieval.setdefault("tool_trace", []).append({"tool": "semantic_search", "args": {"query": f"{work_hint} director", "k": 6}, "raw_results": sem_res})
                        for r in sem_res:
                            cid = r.get("chunk_id")
                            if isinstance(cid, str):
                                cr = _call_chunk_read(cid)
                                if isinstance(cr, dict) and "full_context" in cr:
                                    text_block = _safe_text(cr)
                                    # If this chunk contains an explicit 'directed by' sentence,
                                    # promote it immediately so the extractor can find the director.
                                    if re.search(r"directed by", text_block, re.IGNORECASE) or re.search(r"written and directed by", text_block, re.IGNORECASE):
                                        retrieval.setdefault("retrieved_texts", []).insert(0, {"chunk_id": cid, "source": _safe_source(cr), "text": text_block})
                                        contexts.insert(0, text_block)
                                        break
                                    # Otherwise still promote as a heuristic fallback
                                    retrieval.setdefault("retrieved_texts", []).insert(0, {"chunk_id": cid, "source": _safe_source(cr), "text": text_block})
                                    contexts.insert(0, text_block)
                                    break
                    except Exception:
                        pass
        except Exception:
            pass
        # Deterministic evidence-only assembly: when the caller forbids LLM
        # fallback and we have retrieved contexts, synthesize a concise
        # evidence-backed answer from the contexts without requiring an LLM.
        answer = ""
        try:
            # initialize local director vars used below
            director = None
            director_citation = None
            if contexts and not allow_fallback:
                retrieved = retrieval.get("retrieved_texts", [])
                # Quick deterministic check: if we've promoted a film chunk
                # to the front (contexts[0]) try extracting the director from
                # that chunk first. This avoids picking up unrelated "directed by"
                # mentions elsewhere in the retrieved set.
                if work_hint and retrieved:
                    try:
                        first_entry = retrieved[0]
                        first_txt = (first_entry.get("text") or first_entry.get("full_context") or "")
                        first_cid = (first_entry.get("chunk_id") or "")
                        if first_txt and (work_hint.lower() in first_txt.lower() or work_hint.replace(" ", "_").lower() in first_cid.lower()):
                            name0 = extract_person_name_from_text(first_txt, work_hint=work_hint)
                            if name0:
                                director = name0
                                director_citation = 1
                    except Exception:
                        pass
                # Deterministic lookup: try common chunk id variants for the work
                if not director and work_hint and callable(chunk_read):
                    try:
                        work_norm = work_hint.replace(" ", "_")
                        candidates_ids = [work_norm, f"{work_norm}_0", f"{work_norm}_1"]
                        for pid in candidates_ids:
                            cr = _call_chunk_read(pid)
                            if not cr:
                                continue
                            tb = _safe_text(cr)
                            if not tb:
                                continue
                            if work_hint.lower() in tb.lower() or work_norm.lower() in (pid or "").lower():
                                name_direct = extract_person_name_from_text(tb, work_hint=work_hint)
                                if name_direct:
                                    director = name_direct
                                    retrieval.setdefault("retrieved_texts", []).insert(0, {"chunk_id": pid, "source": _safe_source(cr), "text": tb})
                                    contexts.insert(0, tb)
                                    director_citation = 1
                                    break
                    except Exception:
                        pass
                # If still not found, scan tool_trace and retrieved_texts for candidate ids
                if not director and work_hint:
                    try:
                        work_norm = work_hint.replace(" ", "_").lower()
                        candidate_ids = []
                        for tr in retrieval.get("tool_trace", []):
                            raw = tr.get("raw_results") if isinstance(tr, dict) else None
                            if isinstance(raw, list):
                                for it in raw:
                                    cid = it.get("chunk_id") if isinstance(it, dict) else None
                                    if isinstance(cid, str) and work_norm in cid.lower():
                                        candidate_ids.append(cid)
                        for e in retrieval.get("retrieved_texts", []):
                            cid = e.get("chunk_id")
                            if isinstance(cid, str) and work_norm in cid.lower() and cid not in candidate_ids:
                                candidate_ids.append(cid)
                        for cid in candidate_ids:
                            cr = _call_chunk_read(cid)
                            if not cr:
                                continue
                            tb = _safe_text(cr)
                            if not tb:
                                continue
                            nm = extract_person_name_from_text(tb, work_hint=work_hint)
                            if nm:
                                director = nm
                                retrieval.setdefault("retrieved_texts", []).insert(0, {"chunk_id": cid, "source": _safe_source(cr), "text": tb})
                                contexts.insert(0, tb)
                                director_citation = 1
                                break
                    except Exception:
                        pass

                # If we still don't have a director, try scanning retrieved entries conservatively
                if not director:
                    # If we have a work hint, prefer entries that mention the work
                    if work_hint:
                        for idx, entry in enumerate(retrieved, start=1):
                            txt = (entry.get("text") or entry.get("full_context") or "")
                            cid = (entry.get("chunk_id") or "")
                            if work_hint.lower() in txt.lower() or work_hint.lower() in cid.lower():
                                name = extract_person_name_from_text(txt, work_hint=work_hint)
                                if name:
                                    director = name
                                    director_citation = idx
                                    break
                    else:
                        for idx, entry in enumerate(retrieved, start=1):
                            text = entry.get("text") or entry.get("full_context") or ""
                            name = extract_person_name_from_text(text)
                            if name:
                                director = name
                                director_citation = idx
                                break

                # Collect award evidence and assemble verdict
                award_evidence = []
                try:
                    award_evidence = detect_award_evidence(retrieval.get("retrieved_texts", []), name_hint=director)
                except Exception:
                    award_evidence = []

                award_verdict = "Unknown based on provided contexts."
                award_citations = []
                for ev in award_evidence:
                    t = (ev.get("type") or "").lower()
                    if t == "win":
                        award_verdict = "Won at least one Academy Award (supported by contexts)."
                        award_citations = ev.get("citations", [])
                        break
                    if t in ("nom", "nomination"):
                        if award_verdict.startswith("Unknown"):
                            award_verdict = "Nominated for Academy Awards (no win evidence in provided contexts)."
                            award_citations = ev.get("citations", [])

                parts = []
                # If director is a single token (last name only), try a simple
                # post-normalization: prefer a chunk 'source' that contains
                # the last name (e.g., 'James Cameron') and use that full
                # title for clearer answers.
                if director:
                    try:
                        dps = [p for p in director.split() if p]
                        if len(dps) == 1:
                            lname = dps[0].lower()
                            try:
                                import os, pickle
                                chunks_path = os.path.join("data", "chunks.pkl")
                                if os.path.exists(chunks_path):
                                    with open(chunks_path, "rb") as f:
                                        all_chunks = pickle.load(f)
                                else:
                                    all_chunks = []
                            except Exception:
                                all_chunks = []
                            for c in all_chunks:
                                src = (c.get("source") or "").strip()
                                if src and lname in src.lower():
                                    # prefer multi-token Titlecase source
                                    if " " in src:
                                        director = src
                                        break
                    except Exception:
                        pass
                if director:
                    parts.append(f"Director: {director}.")
                else:
                    parts.append("Director: Unknown based on provided contexts.")
                parts.append(award_verdict)

                citations = []
                if director_citation:
                    citations.append(director_citation)
                citations.extend([c for c in award_citations if isinstance(c, int)])
                seen = set()
                final_citations = []
                for c in citations:
                    if c not in seen:
                        seen.add(c)
                        final_citations.append(c)

                cite_str = f" Citations: {final_citations}." if final_citations else ""
                answer = " ".join(parts) + cite_str
                local_citations = final_citations
                out = {
                    "answer": answer,
                    "tool_trace": retrieval.get("tool_trace", []),
                    "retrieved_texts": retrieval.get("retrieved_texts", []),
                    "citations": local_citations,
                    "step_count": retrieval.get("step_count", 0),
                    "token_count": retrieval.get("token_count", 0),
                    "budget_exhausted": False,
                }
                return out
        except Exception:
            # Fallback: join contexts if deterministic assembly fails
            answer = "\n\n".join(contexts)

        out = {
            "answer": answer,
            "tool_trace": retrieval.get("tool_trace", []),
            "retrieved_texts": retrieval.get("retrieved_texts", []),
            "citations": local_citations,
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
            # Try to pass the original work hint (if present in the question)
            work_hint = None
            try:
                # same robust extraction for the per-question work hint
                mwork = re.search(r"who directed\s+([A-Za-z0-9'\- ]+?)(?:\s+and\b|,|\?|$)", question, re.IGNORECASE)
                if mwork:
                    work_hint = mwork.group(1).strip()
            except Exception:
                work_hint = None
            name = extract_person_name_from_text(text, work_hint=work_hint)
            if name:
                break

    # If we didn't find a director among the initial candidates, try an
    # explicit semantic search for "{work} director" and prefer results
    # that both mention the work and an explicit "directed by" pattern.
    if not name:
        try:
            qdir = f"{work} director"
            sem_res = []
            if semantic_search is not None:
                sem_res = semantic_search(qdir, k=8)
                trace.append({"tool": "semantic_search", "args": {"query": qdir, "k": 8}, "raw_results": sem_res})
            if not sem_res and keyword_search is not None:
                sem_res = keyword_search(qdir, k=8)
                trace.append({"tool": "keyword_search", "args": {"query": qdir, "k": 8}, "raw_results": sem_res})

            for r in sem_res:
                cid2 = r.get("chunk_id")
                txt2 = (r.get("text") or "")
                cr2 = None
                if isinstance(cid2, str):
                    cr2 = _call_chunk_read(cid2)
                    if isinstance(cr2, dict):
                        txt2 = _safe_text(cr2) or txt2
                if work.lower() in txt2.lower() and re.search(r"directed by", txt2, re.IGNORECASE):
                    name = extract_person_name_from_text(txt2, work_hint=work)
                    if name:
                        retrieved_texts.append({"chunk_id": cid2 or "", "source": _safe_source(cr2) if isinstance(cr2, dict) else "", "text": txt2})
                        local_token_count += approx_token_count(txt2)
                        local_steps += 1
                        break
        except Exception:
            pass

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


def extract_person_name_from_text(text: str, work_hint: str | None = None) -> str | None:
    import re
    # If a work hint is provided, prefer explicit sentences that mention the
    # work and one of the strong director phrases (e.g. "written and directed by").
    if work_hint:
        try:
            sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
            strong_phrases = [r"written and directed by", r"co-written and directed by", r"written, directed by", r"directed by"]
            # First look for the strongest patterns in sentences that also mention the work title
            for s in sents:
                s_low = s.lower()
                if work_hint.lower() in s_low:
                    for ph in strong_phrases:
                        if re.search(ph, s_low, re.IGNORECASE):
                            # capture the name following the phrase
                            m = re.search(rf"{ph}\s+([A-Z][\w\-\.'’]+(?:\s+[A-Z][\w\-\.'’]+)*)", s, re.IGNORECASE)
                            if m:
                                name = m.group(1).strip(" ,.")
                                return " ".join(name.split())
            # If none of the strong phrases matched within work-containing sentences,
            # scan any sentence with a weaker 'directed by' but still requiring the work hint
            for s in sents:
                s_low = s.lower()
                if work_hint.lower() in s_low and re.search(r"directed by", s, re.IGNORECASE):
                    m = re.search(r"directed by\s+([A-Z][\w\-\.'’]+(?:\s+[A-Z][\w\-\.'’]+)*)", s, re.IGNORECASE)
                    if m:
                        name = m.group(1).strip(" ,.")
                        return " ".join(name.split())
        except Exception:
            # fall through to the general approach
            pass

    # Try a few permissive regexes (case-insensitive) to capture patterns like
    # "directed by Scott Derrickson" or "was directed by Scott Derrickson".
    patterns = [
        r"(?:was )?directed by\s+([A-Z][\w\-\.'’]+(?:\s+[A-Z][\w\-\.'’]+)*)",
        r"film directed by\s+([A-Z][\w\-\.'’]+(?:\s+[A-Z][\w\-\.'’]+)*)",
        r"directed by\s+([A-Z][\w\-\.'’]+(?:\s+[A-Z][\w\-\.'’]+)*)",
    ]
    matched_directed = False
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip(" ,.")
            # Normalize whitespace
            name = " ".join(name.split())
            matched_directed = True
            return name

    # Extra safety: if a work_hint is provided, prefer extracting from any chunk
    # identifiers or lines that contain the normalized work title. This helps when
    # the promoted film chunk is present but generic 'directed by' matches appear
    # elsewhere in the article.
    if work_hint:
        try:
            # look for explicit "<Work> is a ... written and directed by <Name>" pattern
            m = re.search(rf"{re.escape(work_hint)}[\w\s,\-:\'\"]{{0,80}}written and directed by\s+([A-Z][\w\-\.'’]+(?:\s+[A-Z][\w\-\.'’]+)*)", text, re.IGNORECASE)
            if m:
                return " ".join(m.group(1).strip(" ,.").split())
        except Exception:
            pass

    # If a work hint was provided, be conservative: don't use the loose
    # Titlecase fallback unless we found an explicit 'directed by' match.
    if work_hint and not matched_directed:
        return None

    # Fallback: find the longest run of Titlecase tokens (2..4 words) which
    # often corresponds to person names. This is more robust than str.istitle()
    # which can fail on punctuation/quotes.
    cand_re = re.compile(r"\b([A-Z][a-z][\w\-\.'’]+(?:\s+[A-Z][a-z][\w\-\.'’]+){0,3})\b")
    matches = cand_re.findall(text)
    if not matches:
        return None
    # Choose the longest match (most words)
    best = max(matches, key=lambda s: len(s.split()))
    return best.strip(" ,.")


def detect_award_evidence(
    entries: List[Dict[str, str]], name_hint: str | None = None
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    import re

    # expanded award detection patterns (captures plural/numbered wins, "winner of", and variants)
    patterns_win = [
        r"won (an |the )?academy award",
        r"won \d+ (academy awards|oscars?)",
        r"academy award winner",
        r"winner of the academy award (for )?",
        r"winner of an oscar",
        r"won an oscar",
        r"oscar winner",
        r"received an academy award",
        r"received the academy award",
        r"awarded the academy award",
    ]
    patterns_nom = [
        r"nominated for (an |the )?academy award",
        r"nomination for (an |the )?academy award",
        r"nominated for an oscar",
        r"received (an |the )?academy award nomination",
        r"was nominated",
    ]
    last_name = None
    if name_hint:
        parts = [p for p in name_hint.split() if p]
        if parts:
            last_name = parts[-1].lower()

    candidates: List[Dict[str, Any]] = []

    # Pre-compute whether a person-page chunk is present in the entries
    person_page_present = False
    person_variants = set()
    if name_hint:
        safe = name_hint.replace(" ", "_")
        person_variants.update({safe.lower(), f"{safe.lower()}_0", f"{safe.lower()}_1", f"list_of_awards_and_nominations_received_by_{safe.lower()}", f"awards_and_nominations_received_by_{safe.lower()}"})
    for e in entries:
        cid = (e.get("chunk_id") or "").lower()
        src = (e.get("source") or "").lower()
        # If any chunk id or source matches a person variant, mark presence
        for pv in person_variants:
            if pv and (pv in cid or pv in src):
                person_page_present = True
                break
        if person_page_present:
            break

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

    for idx, e in enumerate(entries):
        raw_text = e.get("text") or ""
        text = raw_text.lower()
        cid = (e.get("chunk_id") or "").lower()
        src = (e.get("source") or "").lower()

        # Determine priority: high if awards page or contains last name or if a person page
        priority = 1
        awards_page = is_awards_page_identifier(cid, src)
        contains_name = last_name and (last_name in text or last_name in cid or last_name in src)
        if awards_page or contains_name or person_page_present:
            # treat person-page presence as high priority for award evidence
            priority = 0

        matched_type = None
        matched_pattern = None
        # Look for win patterns first
        for p in patterns_win:
            if re.search(p, text):
                matched_type = "win"
                matched_pattern = p
                break
        # If no win, look for nomination patterns
        if not matched_type:
            for p in patterns_nom:
                if re.search(p, text):
                    matched_type = "nom"
                    matched_pattern = p
                    break

        if matched_type:
            # If this chunk is clearly an awards page, accept the match even
            # if the chunk text does not include the person's full name.
            # Also accept if we earlier detected a person-page among the
            # retrieved entries (person_page_present). This relaxes strict
            # sentence-scoping so film-page award sentences can be used when a
            # person page context exists.
            if awards_page or contains_name or person_page_present:
                candidates.append(
                    {
                        "entry_index": idx,
                        "chunk_id": e.get("chunk_id"),
                        "source": e.get("source"),
                        "text": raw_text,
                        "type": matched_type,
                        "pattern": matched_pattern,
                        "priority": priority,
                        "citations": [idx + 1],
                    }
                )

    # Sort candidates by priority (0 first) and return
    candidates_sorted = sorted(candidates, key=lambda x: x.get("priority", 1))
    # Convert to expected output (remove priority)
    for c in candidates_sorted:
        entry = dict(c)
        entry.pop("priority", None)
        evidence.append(entry)
    return evidence
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

    # Strong deterministic pre-scan: look through the chunks metadata for
    # any chunk ids or source titles that look like the requested work and
    # promote the best match(s). This helps when indexes or PRF re-ranking
    # might not return the authoritative film chunk early enough.
    try:
        # load chunk metadata by probing the retrieval_tools loader if available
        try:
            from arag.retrieval_tools import load_indexes

            # ensure indexes/chunks are loaded into retrieval_tools globals
            load_indexes()
            from arag.retrieval_tools import _chunks as _all_chunks
        except Exception:
            _all_chunks = None

        if _all_chunks:
            # build normalized tokens
            import re as _re

            def _norm(t: str) -> str:
                return _re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")

            work_norm = _norm(work or "")
            candidates = []
            for c in _all_chunks:
                cid = (c.get("id") or "")
                src = (c.get("source") or "")
                if not cid and not src:
                    continue
                cid_norm = _norm(cid)
                src_norm = _norm(src)
                score = 0
                if work_norm and work_norm == cid_norm.split("_")[0]:
                    score += 10
                if work_norm and work_norm in cid_norm:
                    score += 8
                if work_norm and work_norm in src_norm:
                    score += 6
                # prefer chunk ids that equal work_norm or start with it
                if score > 0:
                    candidates.append((score, cid))
            if candidates:
                # choose top candidate and promote it
                candidates.sort(key=lambda x: -x[0])
                top_cid = candidates[0][1]
                cr_top = _call_chunk_read(top_cid)
                if isinstance(cr_top, dict) and "full_context" in cr_top:
                    txt_top = _safe_text(cr_top)
                    trace.append({"tool": "chunk_read", "args": {"chunk_id": top_cid}, "result_summary": txt_top[:400]})
                    retrieved_texts.insert(0, {"chunk_id": top_cid, "source": _safe_source(cr_top), "text": txt_top})
                    local_token_count += approx_token_count(txt_top)
                    local_steps += 1
                    # try extracting immediately
                    try:
                        nm = extract_person_name_from_text(txt_top, work_hint=work)
                        if nm:
                            name = nm
                    except Exception:
                        pass
    except Exception:
        pass

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
    # If none of the initial deduped candidates mention the work title,
    # run an additional targeted search for the work to gather more
    # candidate chunks (helps when the film page wasn't returned initially).
    try:
        work_norm = (work or "").replace(" ", "_").lower()
        has_work_mention = False
        for r in deduped:
            cid = (r.get("chunk_id") or "").lower()
            txt = (r.get("text") or "").lower()
            if work_norm and (work_norm in cid or work.lower() in txt):
                has_work_mention = True
                break
        if not has_work_mention:
            extra = []
            # Try a suite of targeted queries to increase chance of finding the film
            work_queries = []
            if work:
                work_queries = [
                    work,
                    f'"{work}"',
                    f"{work} directed by",
                    f"directed by {work}",
                    f"{work} director",
                    f"{work} (film)",
                    f"{work} (film) director",
                ]
            # semantic first with modest k
            if semantic_search is not None:
                try:
                    for q in work_queries:
                        try:
                            sem_extra = semantic_search(q, k=16)
                            trace.append({"tool": "semantic_search", "args": {"query": q, "k": 16}, "raw_results": sem_extra})
                            if sem_extra:
                                extra.extend(sem_extra)
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
            # keyword/BM25 with larger k as a last-resort sweep
            if keyword_search is not None:
                try:
                    for q in work_queries:
                        try:
                            kw_extra = keyword_search(q, k=200)
                            trace.append({"tool": "keyword_search", "args": {"query": q, "k": 200}, "raw_results": kw_extra})
                            if kw_extra:
                                extra.extend(kw_extra)
                                break
                        except Exception:
                            continue
                except Exception:
                    pass
            for r in extra:
                cid = r.get("chunk_id")
                if cid and cid not in seen_ids:
                    deduped.append(r)
                    seen_ids.add(cid)
    except Exception:
        pass

    # Aggressive promotion: if the corpus contains an explicit chunk whose id
    # matches the normalized work title (e.g., Titanic_0), promote it to the
    # front of retrieved_texts so extraction sees it first. This helps when
    # BM25/semantic ordering didn't return the film page early enough.
    try:
        if work:
            work_norm = work.replace(" ", "_")
            possible_ids = [work_norm, f"{work_norm}_0", f"{work_norm}_1", f"{work_norm}_00"]
            for pid in possible_ids:
                # If we already saw this id in deduped, prefer using that
                # candidate (the deduped entries will be processed below),
                # otherwise probe via chunk_read.
                found = False
                for r in deduped:
                    cid = r.get("chunk_id") if isinstance(r, dict) else None
                    if isinstance(cid, str) and cid.lower() == pid.lower():
                        # prefer this candidate's chunk data
                        cr_try = _call_chunk_read(cid)
                        if isinstance(cr_try, dict) and "full_context" in cr_try:
                            txt_try = _safe_text(cr_try)
                            trace.append({"tool": "chunk_read", "args": {"chunk_id": cid}, "result_summary": txt_try[:400]})
                            # promote to front
                            retrieved_texts.insert(0, {"chunk_id": cid, "source": _safe_source(cr_try), "text": txt_try})
                            local_token_count += approx_token_count(txt_try)
                            local_steps += 1
                            # try extracting right away
                            nm = extract_person_name_from_text(txt_try, work_hint=work)
                            if nm:
                                name = nm
                            found = True
                            break
                if found:
                    break
                # probe chunk_read directly if not found in deduped
                crp = _call_chunk_read(pid)
                if isinstance(crp, dict) and "full_context" in crp:
                    txtp = _safe_text(crp)
                    trace.append({"tool": "chunk_read", "args": {"chunk_id": pid}, "result_summary": txtp[:400]})
                    retrieved_texts.insert(0, {"chunk_id": pid, "source": _safe_source(crp), "text": txtp})
                    local_token_count += approx_token_count(txtp)
                    local_steps += 1
                    nm = extract_person_name_from_text(txtp, work_hint=work)
                    if nm:
                        name = nm
                    break
    except Exception:
        pass
    name = None
    inspected = 0

    # Pre-scan deduped first-hop results and prefer any chunk that
    # explicitly mentions the work (in id or text). This helps when the
    # film page (e.g., Inception_0) is present — we should use that to
    # extract the director rather than person-page snippets that may not
    # mention the film.
    work_norm = (work or "").replace(" ", "_").lower()
    # Deterministic quick-check: try explicit chunk-id variants for the work
    # before relying on semantic results. This enforces the paper's evidence-
    # first principle: if we have a chunk whose id starts with the normalized
    # title and it contains an explicit 'directed by' sentence, extract from it
    # and skip scanning other person pages.
    try:
        possible_ids = [work_norm, f"{work_norm}_0", f"{work_norm}_1", f"{work_norm}_00"]
        for pid in possible_ids:
            cr_try = _call_chunk_read(pid)
            if isinstance(cr_try, dict) and "full_context" in cr_try:
                txt_try = _safe_text(cr_try)
                trace.append({"tool": "chunk_read", "args": {"chunk_id": pid}, "result_summary": txt_try[:400]})
                # look for a strong directed-by sentence in this chunk
                try:
                    import re

                    sents_try = [s.strip() for s in re.split(r"(?<=[.?!])\s+", txt_try) if s.strip()]
                    for s in sents_try:
                        sl = s.lower()
                        if work.lower() in sl and ("written and directed by" in sl or "directed by" in sl):
                            retrieved_texts.append({"chunk_id": pid, "source": _safe_source(cr_try), "text": txt_try})
                            local_token_count += approx_token_count(txt_try)
                            local_steps += 1
                            name = extract_person_name_from_text(s, work_hint=work)
                            if name:
                                # found a definitive film-chunk director, return early
                                person = name.strip()
                                # proceed to collect award evidence below as usual
                                break
                    if name:
                        break
                except Exception:
                    pass
    except Exception:
        # non-fatal; fall back to existing behavior
        pass
    # If a deterministic film-chunk director was already found above, respect it
    # and do not scan other candidate chunks which might overwrite the extracted name.
    if not name:
        for cand in deduped:
            cid_c = (cand.get("chunk_id") or "").lower()
            txt_c = (cand.get("text") or "").lower()
            if work_norm and (work_norm in cid_c or work_norm in txt_c):
                cid = cand.get("chunk_id")
                if isinstance(cid, str):
                    cr = _call_chunk_read(cid)
                    summary = _safe_text(cr)[:400]
                    trace.append({"tool": "chunk_read", "args": {"chunk_id": cid}, "result_summary": summary})
                    if isinstance(cr, dict) and "full_context" in cr:
                        text = _safe_text(cr)
                        retrieved_texts.append({"chunk_id": cid, "source": _safe_source(cr), "text": text})
                        local_token_count += approx_token_count(text)
                        local_steps += 1
                        name = extract_person_name_from_text(text, work_hint=work)
                        if name:
                            break

    # If pre-scan didn't find it, fall back to scanning the top candidates
    if not name:
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
                name = extract_person_name_from_text(text, work_hint=work)
                if name:
                    break
            inspected += 1
    # If we still don't have a director, try an enhanced conservative
    # person-page fallback: look through candidate chunks (often person
    # pages) for sentences or small sentence-windows that mention the
    # work title and an explicit 'directed by' phrase. We build a few
    # normalized variants of the work title (strip parentheticals/years)
    # to improve matching on person pages that refer to the work in a
    # slightly different form.
    if not name:
        try:
            import re as _re

            def _work_variants(w: str) -> List[str]:
                variants = [w.strip()]
                # strip parenthetical e.g. 'Titanic (1997 film)' -> 'Titanic'
                v = _re.sub(r"\s*\(.*?\)", "", w).strip()
                if v and v not in variants:
                    variants.append(v)
                # strip trailing year tokens like 'Titanic 1997' or 'Titanic, 1997'
                v2 = _re.sub(r"[,\s]+\d{4}$", "", v).strip()
                if v2 and v2 not in variants:
                    variants.append(v2)
                # underscored normalized form
                unders = w.replace(" ", "_")
                if unders not in variants:
                    variants.append(unders)
                return [vv.lower() for vv in variants if vv]

            work_variants = _work_variants(work or "")

            for cand in deduped:
                cid = cand.get("chunk_id")
                if not isinstance(cid, str):
                    continue
                cr = _call_chunk_read(cid)
                if not isinstance(cr, dict) or "full_context" not in cr:
                    continue
                text_block = _safe_text(cr) or ""
                if not text_block:
                    continue
                # Split into sentences and search windows of up to 3 sentences
                sents = [s.strip() for s in _re.split(r"(?<=[.?!])\s+", text_block) if s.strip()]
                # Pre-lower sentences for quick checks
                sents_low = [s.lower() for s in sents]

                found = False
                # First pass: single-sentence must contain both work variant and directed-by phrase
                for i, s in enumerate(sents_low):
                    if any(wv in s for wv in work_variants) and _re.search(r"written and directed by|written & directed by|directed by", s, _re.IGNORECASE):
                        # prefer the original-cased sentence for extraction
                        orig = sents[i]
                        nm = extract_person_name_from_text(orig, work_hint=work)
                        if nm:
                            name = nm
                            retrieved_texts.insert(0, {"chunk_id": cid, "source": _safe_source(cr), "text": text_block})
                            local_token_count += approx_token_count(text_block)
                            local_steps += 1
                            trace.append({"tool": "chunk_read", "args": {"chunk_id": cid}, "result_summary": text_block[:400]})
                            found = True
                            break
                if found:
                    break

                # Second pass: sliding window of up to 3 sentences; if the window
                # contains both a work variant and a directed-by phrase, extract
                # from the sentence that contains the directed-by phrase (if any).
                n = len(sents)
                for start in range(0, n):
                    for end in range(start, min(n, start + 3)):
                        window_low = " ".join(sents_low[start : end + 1])
                        if any(wv in window_low for wv in work_variants) and _re.search(r"written and directed by|written & directed by|directed by", window_low, _re.IGNORECASE):
                            # find the sentence in the window with directed-by
                            directed_idx = None
                            for j in range(start, end + 1):
                                if _re.search(r"written and directed by|written & directed by|directed by", sents_low[j], _re.IGNORECASE):
                                    directed_idx = j
                                    break
                            if directed_idx is None:
                                directed_idx = start
                            orig = sents[directed_idx]
                            nm = extract_person_name_from_text(orig, work_hint=work)
                            if nm:
                                name = nm
                                retrieved_texts.insert(0, {"chunk_id": cid, "source": _safe_source(cr), "text": text_block})
                                local_token_count += approx_token_count(text_block)
                                local_steps += 1
                                trace.append({"tool": "chunk_read", "args": {"chunk_id": cid}, "result_summary": text_block[:400]})
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
        except Exception:
            pass

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
    # Person-page-first probe: if person page(s) exist in the corpus, fetch
    # them and prioritize their chunks before running award-search templates.
    person_probed = False
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
            # insert at front so detect_award_evidence and later logic see
            # the person page first (high priority)
            retrieved_texts.insert(0, {"chunk_id": pid, "source": _safe_source(cr), "text": _safe_text(cr)})
            local_token_count += approx_token_count(_safe_text(cr))
            local_steps += 1
            person_probed = True
    # If we found at least one person page, we can return early with the
    # person page(s) since they often contain awards lists. Otherwise continue
    # to run the award query templates below.
    if person_probed:
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
    # Stronger second-hop templates including nominations and award-list variants
    award_queries_extended = [
        f"{person} Academy Awards",
        f"{person} Oscar",
        f"{person} awards",
        f"{person} award",
        f"{person} Academy Award nominations",
        f"{person} Oscar nominations",
        f"{person} nominated for Academy Award",
        f"{person} nominated for an Academy Award",
        f"List of awards and nominations received by {person}",
        f"{person} awards and nominations",
        f"{person} accolades",
        f"{person} wins",
        f"{person} nominations",
        f"{person} awards list",
    ]

    # We'll collect candidate hits from both semantic_search and keyword_search
    seen_texts = set()
    for q in award_queries_extended:
        sem_res = []
        kw_res = []

        # semantic search (may be sentence-level if sentence FAISS exists)
        if semantic_search is not None:
            try:
                sem_res = semantic_search(q, k=8)
                trace.append(
                    {"tool": "semantic_search", "args": {"query": q, "k": 8}, "raw_results": sem_res}
                )
            except Exception as e:
                trace.append({"tool": "semantic_search", "args": {"query": q, "k": 8}, "error": str(e)})

        if keyword_search is not None:
            try:
                kw_res = keyword_search(q, k=8)
                trace.append({"tool": "keyword_search", "args": {"query": q, "k": 8}, "raw_results": kw_res})
            except Exception as e:
                trace.append({"tool": "keyword_search", "args": {"query": q, "k": 8}, "error": str(e)})

        # Normalize director/name: if we only extracted a last name, try to
        # find a person page in the chunks with that last name and prefer the
        # full name from that page for clarity. Use the local `name` variable
        # (the extracted director) rather than a non-local `director` symbol.
        try:
            if name:
                dparts = [p for p in name.split() if p]
                if len(dparts) == 1:
                    lname = dparts[0].lower()
                    # probe chunks metadata for a person page with this last name
                    try:
                        import os, pickle, re as _re
                        chunks_path = os.path.join("data", "chunks.pkl")
                        if os.path.exists(chunks_path):
                            with open(chunks_path, "rb") as f:
                                _all_chunks = pickle.load(f)
                        else:
                            _all_chunks = None
                    except Exception:
                        _all_chunks = None
                    if _all_chunks:
                        candidate_full = None
                        for c in _all_chunks:
                            src = (c.get("source") or "")
                            cid = (c.get("id") or "")
                            # prefer pages whose id or source contains the last name
                            if lname in src.lower() or lname in cid.lower():
                                # try to read the chunk to find Titlecase full name
                                try:
                                    crp = _call_chunk_read(c.get("id"))
                                except Exception:
                                    crp = None
                                if isinstance(crp, dict):
                                    textp = _safe_text(crp) or ""
                                    # look for a Titlecase full name (2-4 tokens) containing the last name
                                    m = _re.search(r"([A-Z][a-z][a-z]+(?:\s+[A-Z][a-z][a-z]+){0,3})", textp)
                                    if m and lname in m.group(1).lower():
                                        candidate_full = m.group(1).strip()
                                        break
                        if candidate_full:
                            name = candidate_full
                        else:
                            # fallback: prefer chunk 'source' that looks like a full name
                            for c in _all_chunks:
                                src = (c.get("source") or "").strip()
                                if src and lname in src.lower():
                                    # require multi-token Titlecase source
                                    if " " in src and any(ch.isupper() for ch in src[:1]):
                                        name = src
                                        break
        except Exception:
            pass

        # Merge semantic results first (they may be sentences)
        for r in sem_res:
            txt = (r.get("text") or "").strip()
            if not txt:
                continue
            if txt in seen_texts:
                continue
            # prefer sentence-level hits; if these are chunk-level, chunk_read below will expand
            if isinstance(r.get("chunk_id"), str):
                cid2 = r.get("chunk_id")
            else:
                cid2 = None
            # add to retrieved_texts directly if it's sentence-like
            retrieved_texts.append({"chunk_id": cid2 or "", "source": "", "text": txt})
            seen_texts.add(txt)
            local_token_count += approx_token_count(txt)
            local_steps += 1

        # Process keyword/BM25 hits by extracting award-containing sentences from the chunk
        for r in kw_res:
            cid2 = r.get("chunk_id")
            if not isinstance(cid2, str):
                continue
            cr2 = _call_chunk_read(cid2)
            summary2 = _safe_text(cr2)[:400]
            trace.append({"tool": "chunk_read", "args": {"chunk_id": cid2}, "result_summary": summary2})
            if not isinstance(cr2, dict):
                continue
            full = (cr2.get("full_context") or cr2.get("coarse") or cr2.get("text") or "")
            # split into sentences and extract those mentioning awards and person's last name
            import re as _re

            sents = [s.strip() for s in _re.split(r"(?<=[.?!])\s+", full) if s.strip()]
            parts = [p for p in person.split() if p]
            last_name = parts[-1].lower() if parts else None
            for s in sents:
                sl = s.lower()
                matched = False
                award_tokens = ["academy", "oscar", "nominated", "nomination", "won", "won an oscar", "academy award"]
                for tkn in award_tokens:
                    if tkn in sl:
                        # if we have a last name, prefer sentences that mention it
                        if last_name and last_name in sl:
                            matched = True
                            break
                        # otherwise accept award token alone
                        matched = True
                        break
                if not matched:
                    continue
                txt = s[:400]
                if txt in seen_texts:
                    continue
                retrieved_texts.append({"chunk_id": cid2, "source": _safe_source(cr2), "text": txt})
                seen_texts.add(txt)
                local_token_count += approx_token_count(txt)
                local_steps += 1

        # stop early if we have gathered enough candidate evidence
        if len(retrieved_texts) >= 6:
            break

    # If we didn't find explicit awards pages, try a heuristic scan across
    # chunks for sentences that mention awards and the person's last name.
    if not retrieved_texts:
        try:
            from .retrieval_tools import find_award_mentions

            fallback_hits = find_award_mentions(person, max_results=6)
            for fh in fallback_hits:
                retrieved_texts.append(
                    {
                        "chunk_id": fh.get("chunk_id"),
                        "source": fh.get("source") or "",
                        "text": fh.get("text") or "",
                    }
                )
                local_token_count += approx_token_count(fh.get("text") or "")
                local_steps += 1
        except Exception:
            pass

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
    parser.add_argument(
        "--allow-fallback",
        dest="allow_fallback",
        action="store_true",
        help="Allow LLM to fallback to external knowledge when contexts are empty",
    )
    parser.add_argument(
        "--no-allow-fallback",
        dest="allow_fallback",
        action="store_false",
        help="Do not allow LLM fallback (answer strictly from contexts)",
    )
    parser.set_defaults(allow_fallback=True)
    args = parser.parse_args()

    result = run_agent(
        args.question,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        allow_fallback=args.allow_fallback,
    )
    print(json.dumps(result, indent=2))

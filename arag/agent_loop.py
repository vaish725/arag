"""Agent loop (Phase 3) moved into arag package.

This file is copied from the top-level `agent_loop.py` and adjusted to use
relative imports for package internal modules.
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

# retrieval_tools symbols may not be available at import-time; declare
# typed optional placeholders which will be replaced at runtime if available.
DATA_DIR = "data"
chunk_read: Optional[Callable[[str], Dict[str, Any]]] = None
keyword_search: Optional[Callable[..., List[Dict[str, Any]]]] = None
semantic_search: Optional[Callable[..., List[Dict[str, Any]]]] = None
try:
    import importlib

    _rt = importlib.import_module("arag.retrieval_tools")
    DATA_DIR = _rt.DATA_DIR
    chunk_read = getattr(_rt, "chunk_read", None)
    keyword_search = getattr(_rt, "keyword_search", None)
    semantic_search = getattr(_rt, "semantic_search", None)
except Exception:
    _rt = None
    chunk_read = None
    keyword_search = None
    semantic_search = None


# --- Step budget and the complete tool trace.
# max_steps = the number of retrieval tool calls (keyword_search, semantic_search, chunk_read) one
# run_agent() call may make. The three tools are wrapped once; the wrapper (a) counts every call against
# the budget and (b) RECORDS every call, including the chunk_read probes the recipe makes internally.
# The record lives in a ContextVar so concurrent runs (Streamlit sessions run in threads) never share
# state. Once the budget is spent the tools return nothing and the attempt is recorded as "blocked", the
# recipe finishes with what it has, and the result carries budget_exhausted=True.
# run_agent() builds result["tool_trace"] from this record, so the number of tool entries equals the number
# of calls actually made (result["tool_calls"]).
DEFAULT_MAX_STEPS = 100  # generous: the longest path seen uses 59 calls (Last Action Hero)
TOOL_NAMES = ("semantic_search", "keyword_search", "chunk_read")


class _StepBudget:
    def __init__(self, limit: Optional[int]):
        self.limit = limit
        self.used = 0
        self.exhausted = False
        self.log: List[Dict[str, Any]] = []  # one entry per call (plus blocked attempts), in call order


_STEP_BUDGET: "contextvars.ContextVar[Optional[_StepBudget]]" = contextvars.ContextVar("arag_step_budget", default=None)


def _tool_args(name: str, args: tuple, kwargs: dict) -> Dict[str, Any]:
    if name == "chunk_read":
        return {"chunk_id": args[0] if args else kwargs.get("chunk_id")}
    return {"query": args[0] if args else kwargs.get("query"), "k": args[1] if len(args) > 1 else kwargs.get("k", 5)}


def _recorded_tool(name: str, empty: Callable[[], Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        fn = getattr(_rt, name)  # resolved at call time, so a substituted implementation is still counted and recorded
        budget = _STEP_BUDGET.get()
        entry: Dict[str, Any] = {"tool": name, "args": _tool_args(name, args, kwargs)}
        if budget is not None:
            if budget.limit is not None and budget.used >= budget.limit:
                budget.exhausted = True
                budget.log.append({**entry, "blocked": True, "error": "step budget exhausted"})
                return empty()
            budget.used += 1
            entry["step"] = budget.used
            budget.log.append(entry)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            entry["error"] = str(e)
            raise
        if name == "chunk_read":
            text = result.get("full_context") if isinstance(result, dict) else None
            entry["result_summary"] = (text or "")[:400]
        else:
            entry["raw_results"] = result
        return result

    wrapper.__name__ = name
    return wrapper


chunk_read = _recorded_tool("chunk_read", dict) if _rt is not None else None  # type: ignore[assignment]
keyword_search = _recorded_tool("keyword_search", list) if _rt is not None else None  # type: ignore[assignment]
semantic_search = _recorded_tool("semantic_search", list) if _rt is not None else None  # type: ignore[assignment]


def _complete_trace(recorded: List[Dict[str, Any]], flow_trace: Any) -> List[Dict[str, Any]]:
    """Every recorded call, in call order, plus the flow's own entries for anything not recorded.

    The flow's own entries for a tool are dropped when that tool was recorded (the record is complete and
    chronological); they are kept for a tool that bypassed the wrapper (an implementation substituted at
    module level) and for pseudo-tools such as "find_award_mentions".
    """
    recorded_tools = {e["tool"] for e in recorded}
    extra = [e for e in (flow_trace or []) if isinstance(e, dict) and e.get("tool") not in recorded_tools]
    return list(recorded) + extra


# "directed by", "directed and produced by", "directed, co-produced, and co-written by": the word
# "directed", up to four words, then "by". The words may not cross sentence punctuation.
DIRECTED_BY = r"(?:co-)?directed(?:[,\s]+[\w\-]+){0,4}?[,\s]+by\b"
_DIRECTED_BY_PHRASE_RE = re.compile(DIRECTED_BY, re.IGNORECASE)
# A person's name: Title-case tokens only (so it stops where the name ends: "Joe Dante and produced by ..." gives
# "Joe Dante"), with lower-case particles ("de", "van") and co-directors joined by "and" ("Chris Weitz and Paul Weitz").
_CAP = r"(?-i:[A-ZÀ-ÖØ-Þ][\w\-\.'’]*)"
_NAME_PAT = (
    r"(?:(?:brothers|sisters|siblings|the|duo|filmmakers?|directors?)\s+)?"
    rf"({_CAP}(?:\s+(?:(?:de|del|della|der|di|da|van|von|la|le|al|bin|ibn)\s+)?{_CAP})*"
    rf"(?:\s+and\s+{_CAP}(?:\s+{_CAP})*)?)"
)


def has_directed_by(text: str) -> bool:
    """True if the text contains "directed by" or a close variant ("directed and produced by")."""
    return bool(_DIRECTED_BY_PHRASE_RE.search(text or ""))


def _name_after_directed_by(sentence: str) -> Optional[str]:
    m = re.search(DIRECTED_BY + r"\s+" + _NAME_PAT, sentence, re.IGNORECASE)
    if not m:
        return None
    name = " ".join(m.group(1).strip(" ,.").split())
    return name or None


def _is_film_page(chunk_id: Optional[str], work_hint: Optional[str]) -> bool:
    """True if ``chunk_id`` is a chunk of the page for ``work_hint`` ("Ed_Wood_(film)_3" for "Ed Wood")."""
    if not chunk_id or not work_hint:
        return False
    return bool(re.match(r"^" + re.escape(work_hint.replace(" ", "_")) + r"(?:_\(.*?\))?_\d+$", chunk_id, re.IGNORECASE))


def approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.findall(r"\w+", text)))


def normalize_title(title: str) -> str:
    """Normalize a work title for chunk lookup and matching.

    - Strip surrounding whitespace
    - Remove trailing parenthetical qualifiers like '(1997 film)'
    - Remove trailing "Part N" or "Part II" style suffixes
    """
    if not title:
        return ""
    t = title.strip()
    try:
        # remove trailing parentheticals
        t = re.sub(r"\s*\(.*?\)\s*$", "", t)
        # remove trailing 'Part N' or roman numerals
        t = re.sub(r"\b[Pp]art\s+(?:[IVX]+|\d+)\b", "", t).strip()
    except Exception:
        pass
    return t


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


EVIDENCE_ONLY_REASON = "evidence-only mode: the LLM is never called (LLM fallback is off)"


def _consider_llm(
    question: str,
    contexts: List[str],
    max_tokens: int,
    allow_fallback: bool,
    trace: List[Dict[str, Any]],
    needed: bool,
    not_needed_reason: str = "",
) -> Optional[Dict[str, Any]]:
    """Decide whether to ask the LLM, ask it if so, and RECORD the decision in ``trace``.

    The entry is {"tool": "llm", "decision": "called" | "skipped" | "failed", ...}:
      called  -> model, input_tokens, output_tokens (None if the provider reports none)
      skipped -> reason (evidence-only mode, verdict already established, key or package missing)
      failed  -> error (the caller keeps its own answer)
    Returns the backend's result dict when the LLM answered, else None.
    """
    entry: Dict[str, Any] = {"tool": "llm", "decision": "skipped"}
    result: Optional[Dict[str, Any]] = None
    # This is the ONLY place the LLM is called. With fallback off (evidence-only mode) it is never
    # called, on any route, whatever else is true; that is checked first so the recorded reason is
    # always the mode.
    if not allow_fallback:
        entry["reason"] = EVIDENCE_ONLY_REASON
    elif not needed:
        entry["reason"] = not_needed_reason or "the LLM was not needed for this answer"
    else:
        try:
            import arag.llm_backend as llm_backend

            if not llm_backend.llm_available():
                entry["reason"] = f"LLM not available: {llm_backend.llm_unavailable_reason() or 'no LLM is configured'}"
            else:
                out = llm_backend.synthesize_answer(question, contexts, max_tokens=max_tokens, allow_fallback=allow_fallback)
                if isinstance(out, dict):
                    result = out
                elif isinstance(out, (tuple, list)):  # older mocks: (answer, citations, used_fallback)
                    result = {"answer": out[0] if out else "", "citations": out[1] if len(out) > 1 else []}
                else:
                    result = {"answer": str(out)}
                entry.update(
                    decision="called",
                    model=result.get("model"),
                    input_tokens=result.get("input_tokens"),
                    output_tokens=result.get("output_tokens"),
                )
        except Exception as e:  # a failing LLM must never hide the corpus-evidence answer
            entry.update(decision="failed", error=f"{type(e).__name__}: {e}")
            result = None
    trace.append(entry)
    return result if result and result.get("answer") else None


def _llm_summary(trace: Any) -> Dict[str, Any]:
    """The last LLM decision in a trace, for the result's top-level "llm" field."""
    for e in reversed(trace or []):
        if isinstance(e, dict) and e.get("tool") == "llm":
            return {k: v for k, v in e.items() if k != "tool"}
    return {"decision": "not considered", "reason": "this path made no LLM decision"}



def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


_TERMINAL = (".", "!", "?", '"', "\u201d", ")", "]")


_ABBREVIATIONS = {
    "bros.", "inc.", "co.", "ltd.", "corp.", "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "mt.", "jr.", "sr.", "vs.",
    "no.", "vol.", "gen.", "col.", "lt.", "sgt.", "rev.", "u.s.", "u.k.", "ft.",
}
_HEADING_TOKEN_RE = re.compile(r"={2,}\s*[^=]+?\s*={2,}\s*")


def split_sentences(text: str) -> List[str]:
    """Sentence split that does not break after abbreviations ("Warner Bros. Pictures") or initials ("C. Robert")."""
    pieces: List[str] = []
    start = 0
    for m in re.finditer(r"(?<=[.?!])\s+", text or ""):
        last_word = (text[start:m.start()].split() or [""])[-1].lower()
        if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]\.", last_word):
            continue
        pieces.append(text[start:m.start()])
        start = m.end()
    pieces.append((text or "")[start:])
    return [x.strip() for x in pieces if x.strip()]


def _without_headings(sentence: str) -> str:
    return _HEADING_TOKEN_RE.sub("", sentence).strip()


_CHUNK_ID_RE = re.compile(r"^(.*)_(\d+)$")


def _neighbour_text(chunk_id: str, delta: int) -> str:
    """Text of the previous (-1) / next (+1) chunk of the same article, "" if there is none.
    Display only: not a tool call, so it is never recorded or counted."""
    m = _CHUNK_ID_RE.match(chunk_id or "")
    if not m or int(m.group(2)) + delta < 0:
        return ""
    try:
        from arag import retrieval_tools

        return retrieval_tools.chunk_text(f"{m.group(1)}_{int(m.group(2)) + delta}")
    except Exception:
        return ""


def _join_chunks(first: str, second: str) -> str:
    """Join two consecutive chunks, dropping the words they share (chunks overlap)."""
    a, b = first.split(), second.split()
    for k in range(min(len(a), len(b), 80), 0, -1):
        if a[-k:] == b[:k]:
            return " ".join(a + b[k:])
    return " ".join(a + b)


def _ends_cleanly(sentence: str) -> bool:
    return sentence.rstrip().endswith(_TERMINAL)


def evidence_window(chunk_text: str, evidence: str, chunk_id: str = "", prev_text: str = "", next_text: str = "") -> Dict[str, Any]:
    """The evidence sentence with at most one neighbouring sentence on each side, all whole sentences.

    Chunks are 200-word windows that start and end mid-sentence. A sentence cut at a chunk edge is completed
    from the neighbouring chunk of the same article (``prev_text`` / ``next_text``). Only where there is no
    neighbour (page start or end) is a cut kept, with an ellipsis; neighbour sentences that are still
    fragments are dropped.
    """
    context = chunk_text
    if prev_text:
        context = _join_chunks(prev_text, context)
    if next_text:
        context = _join_chunks(context, next_text)
    sentences = split_sentences(context)
    target = _norm_ws(evidence)
    idx = next((i for i, x in enumerate(sentences) if _norm_ws(x) == target), None)
    if idx is None:
        idx = next((i for i, x in enumerate(sentences) if target and (target in _norm_ws(x) or _norm_ws(x) in target)), None)
    if idx is None:
        return {"sentence": target[:400], "before": "", "after": ""}
    last = len(sentences) - 1
    # the context starts mid-sentence unless it begins at the first chunk of the page
    first_id = (_CHUNK_ID_RE.sub(lambda m: f"{m.group(1)}_{max(int(m.group(2)) - 1, 0)}", chunk_id) if prev_text else chunk_id)
    start_is_page_start = first_id.endswith("_0")
    cut_start = idx == 0 and not start_is_page_start
    cut_end = idx == last and not _ends_cleanly(sentences[idx])
    before = sentences[idx - 1] if idx >= 1 and not (idx - 1 == 0 and not start_is_page_start) else ""
    after = sentences[idx + 1] if idx + 1 <= last and not (idx + 1 == last and not _ends_cleanly(sentences[idx + 1])) else ""
    sentence = ("\u2026 " if cut_start else "") + _without_headings(sentences[idx]) + (" \u2026" if cut_end else "")
    return {"sentence": sentence, "before": _without_headings(before), "after": _without_headings(after)}


def whole_sentence_passage(text: str, chunk_id: str = "") -> str:
    """A chunk for display: heading markup removed, the partial sentence dropped at each end, "\u2026" where cut."""
    sentences = [x for x in (_without_headings(x) for x in split_sentences(text)) if x]
    if not sentences:
        return ""
    cut_start = not chunk_id.endswith("_0")
    if cut_start and len(sentences) > 1:
        sentences = sentences[1:]
    cut_end = not _ends_cleanly(sentences[-1])
    if cut_end and len(sentences) > 1:
        sentences = sentences[:-1]
    return ("\u2026 " if cut_start else "") + " ".join(sentences) + (" \u2026" if cut_end else "")


def build_citation_details(result: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
    """For each citation: the source, the exact evidence sentence(s) with what each supports
    ("director", "verdict", "career"; "cited" when the answer gave no sentence), and the full passage."""
    retrieved = result.get("retrieved_texts") or []
    verdict = result.get("verdict_evidence") or {}
    careers = result.get("career_wins") or []
    m = re.match(r"Director:\s*(.+?)\.\s", result.get("template_answer") or result.get("answer") or "")
    director = m.group(1).strip() if m and not m.group(1).lower().startswith("unknown") else None
    title = extract_asked_title(question)
    details: List[Dict[str, Any]] = []
    for n in result.get("citations") or []:
        if not isinstance(n, int) or not 1 <= n <= len(retrieved):
            continue
        entry = retrieved[n - 1]
        text, cid = entry.get("text") or "", entry.get("chunk_id") or ""
        prev_text, next_text = _neighbour_text(cid, -1), _neighbour_text(cid, +1)
        supports: List[Dict[str, Any]] = []
        if director and title:
            for sent in split_sentences(text):
                if has_directed_by(sent) and director in sent and (re.search(re.escape(title), sent, re.IGNORECASE) or _is_film_page(cid, title)):
                    supports.append({"role": "director", "evidence": sent})
                    break
        if verdict.get("citation") == n and verdict.get("matched_text"):
            supports.append({"role": "verdict", "evidence": verdict["matched_text"]})
        for c in careers:
            if c.get("citation") == n and c.get("matched_text"):
                supports.append({"role": "career", "evidence": c["matched_text"]})
        if not supports:
            supports.append({"role": "cited", "evidence": ""})
        for sp in supports:
            sp.update(evidence_window(text, sp["evidence"], cid, prev_text, next_text) if sp["evidence"] else {"sentence": "", "before": "", "after": ""})
            sp.pop("evidence")
        details.append({"citation": n, "chunk_id": cid, "source": entry.get("source") or "", "supports": supports, "full_text": whole_sentence_passage(text, cid)})
    return details


def run_agent(
    question: str,
    max_steps: Optional[int] = DEFAULT_MAX_STEPS,
    max_tokens: int = 2000,
    allow_fallback: bool = True,
    enable_director_flow: bool = False,
) -> Dict[str, Any]:
    """Run one question.

    max_steps  : most retrieval tool calls this run may make (None = unlimited). When it is hit the
                 result has budget_exhausted=True and the answer is built from what was gathered.
    max_tokens : cap on the LLM answer length; only used when the LLM is called.
    The result also carries ``tool_calls`` (tool calls made) and ``vector_search``
    ({"enabled", "reason"}): vector search off or failing is never silent.
    """
    from arag.health import runtime_status
    from arag.retrieval_tools import vector_search_status

    budget = _StepBudget(max_steps)
    token = _STEP_BUDGET.set(budget)
    try:
        result = _run_agent(question, max_steps, max_tokens, allow_fallback, enable_director_flow)
    finally:
        _STEP_BUDGET.reset(token)
    result["budget_exhausted"] = budget.exhausted
    result["tool_calls"] = budget.used
    result["tool_trace"] = _complete_trace(budget.log, result.get("tool_trace"))
    result["step_count"] = budget.used  # one step = one recorded tool call (it used to be an internal counter)
    result["llm"] = _llm_summary(result["tool_trace"])
    result["citation_details"] = build_citation_details(result, question)
    result["vector_search"] = runtime_status(result.get("tool_trace"), vector_search_status())
    return result


def _run_agent(
    question: str,
    max_steps: Optional[int],
    max_tokens: int,
    allow_fallback: bool,
    enable_director_flow: bool,
) -> Dict[str, Any]:
    from arag.runtime_ops import start_run

    start_run()  # fresh per-question API budget
    q = question or ""
    local_citations: List[int] = []
    # If the specialized director flow is not enabled, and the caller forbids
    # any LLM fallback, then for unrelated (non-director) queries we should
    # return an empty answer. This prevents accidental routing of generic
    # queries into the director two-hop logic or the generic synthesis flow
    # when the caller explicitly requested evidence-only/no-fallback behavior.
    if not enable_director_flow and not allow_fallback:
        if not re.search(r"\bdirector\b|\bdirected\b", q, re.IGNORECASE):
            return {
                "answer": "",
                "tool_trace": [{"tool": "llm", "decision": "skipped", "reason": EVIDENCE_ONLY_REASON}],
                "retrieved_texts": [],
                "step_count": 0,
                "token_count": 0,
                "budget_exhausted": False,
            }
    # Only run the specialized director→award two-hop flow if explicitly enabled
    if enable_director_flow and re.search(r"\bdirector\b|\bdirected\b", q, re.IGNORECASE):
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
            # Capture the work title more flexibly: allow parentheses and
            # additional tokens. Use a non-greedy capture up to common
            # delimiters like ' and', comma, question mark or end of string.
            # Afterwards strip parenthetical qualifiers such as
            # '(1997 film)'. This fixes titles like 'Titanic (1997 film)'.
            m = re.search(r"who directed\s+(.+?)(?:\s+and\b|,|\?|$)", q, re.IGNORECASE)
            if not m:
                m = re.search(r"director of\s+(.+?)(?:\s+and\b|,|\?|$)", q, re.IGNORECASE)
            if m:
                work_hint = normalize_title(m.group(1))

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
                        if work_hint.lower() in lt and has_directed_by(lt):
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
                                    if has_directed_by(text_block):
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
            if contexts:
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
                            name0 = extract_person_name_from_text(first_txt, work_hint=work_hint, chunk_id=first_cid)
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
                                name_direct = extract_person_name_from_text(tb, work_hint=work_hint, chunk_id=pid)
                                if name_direct:
                                    director = name_direct
                                    retrieval.setdefault("retrieved_texts", []).insert(0, {"chunk_id": pid, "source": _safe_source(cr), "text": tb})
                                    contexts.insert(0, tb)
                                    director_citation = 1
                                    break
                    except Exception:
                        pass
                    # Also try a variant of the work title with trailing 'Part II/Part 2' removed
                    try:
                        alt_work = normalize_title(work_hint)
                        if alt_work and alt_work.lower() != work_hint.lower():
                            alt_norm = alt_work.replace(" ", "_")
                            candidates_ids = [alt_norm, f"{alt_norm}_0", f"{alt_norm}_1"]
                            for pid in candidates_ids:
                                cr = _call_chunk_read(pid)
                                if not cr:
                                    continue
                                tb = _safe_text(cr)
                                if not tb:
                                    continue
                                if alt_work.lower() in tb.lower() or alt_norm.lower() in (pid or "").lower():
                                    name_direct = extract_person_name_from_text(tb, work_hint=work_hint, chunk_id=pid)
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
                            nm = extract_person_name_from_text(tb, work_hint=work_hint, chunk_id=pid)
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
                                name = extract_person_name_from_text(txt, work_hint=work_hint, chunk_id=cid)
                                if name:
                                    director = name
                                    director_citation = idx
                                    break
                    # Final attempt: run a targeted semantic search for the work title
                    if not director and semantic_search is not None and callable(chunk_read):
                        try:
                            sem_res = semantic_search(work_hint, k=6)
                            retrieval.setdefault("tool_trace", []).append({"tool": "semantic_search", "args": {"query": work_hint, "k": 6}, "raw_results": sem_res})
                            for r in sem_res:
                                cid = r.get("chunk_id")
                                if isinstance(cid, str):
                                    cr = _call_chunk_read(cid)
                                    tb = _safe_text(cr)
                                    if not tb:
                                        continue
                                    name = extract_person_name_from_text(tb, work_hint=work_hint, chunk_id=cid)
                                    if name:
                                        director = name
                                        retrieval.setdefault("retrieved_texts", []).insert(0, {"chunk_id": cid, "source": _safe_source(cr), "text": tb})
                                        contexts.insert(0, tb)
                                        director_citation = 1
                                        break
                        except Exception:
                            pass
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
                    # If we found a director, but the current retrieved_texts
                    # don't include an authoritative person page, try a
                    # targeted semantic search for the person and promote their
                    # page so award detection can scan it.
                    if director and callable(chunk_read):
                        # Include the work title and explicit award keywords so
                        # the BM25 fallback (used when no embedding backend/API
                        # key is configured) is likely to surface the specific
                        # award-evidence sentence, not just the person's bio
                        # intro (which rarely states win/nomination verdicts).
                        person_query = f"{director} {work_hint} won Best Director Academy Award nominated" if work_hint else f"{director} academy award"
                        person_res: List[Dict[str, Any]] = []
                        if semantic_search is not None:
                            try:
                                person_res = semantic_search(person_query, k=6)
                                retrieval.setdefault("tool_trace", []).append({"tool": "semantic_search", "args": {"query": person_query, "k": 6}, "raw_results": person_res})
                            except Exception:
                                person_res = []
                        if not person_res and keyword_search is not None:
                            try:
                                person_res = keyword_search(person_query, k=6)
                                retrieval.setdefault("tool_trace", []).append({"tool": "keyword_search", "args": {"query": person_query, "k": 6}, "raw_results": person_res})
                            except Exception:
                                person_res = []
                        for r in person_res:
                            cid = r.get("chunk_id")
                            if isinstance(cid, str):
                                cr = _call_chunk_read(cid)
                                tb = _safe_text(cr)
                                if not tb:
                                    continue
                                # promote the first likely person/work page (heuristic)
                                src = _safe_source(cr) or ""
                                if (
                                    director.lower().split()[-1] in src.lower()
                                    or director.lower() in tb.lower()
                                    or (work_hint and work_hint.lower() in (cid or "").lower())
                                ):
                                    retrieval.setdefault("retrieved_texts", []).insert(0, {"chunk_id": cid, "source": src, "text": tb})
                                    contexts.insert(0, tb)
                                    break
                    award_evidence = detect_award_evidence(retrieval.get("retrieved_texts", []), name_hint=director)
                except Exception:
                    award_evidence = []

                attribution = attribute_award_evidence(award_evidence, extract_asked_title(q) or work_hint, director)
                award_verdict = attribution["verdict"]
                award_citations = attribution["citations"]
                verdict_evidence = attribution["evidence"]  # the passage that produced the verdict
                career_wins = attribution["career_wins"]

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
                                chunks_path = os.path.join(DATA_DIR, "chunks.pkl")
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
                if career_wins_sentence(career_wins):
                    parts.append(career_wins_sentence(career_wins))

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
                # The corpus-evidence template is always built first. The LLM is asked only when the
                # VERDICT is Unknown (whether or not the director was found) and fallback is allowed;
                # every decision, including a skip or a failure, is recorded in the trace.
                template_answer = answer
                llm_result = _consider_llm(
                    q, contexts, max_tokens, allow_fallback, retrieval.setdefault("tool_trace", []),
                    needed=award_verdict.startswith("Unknown"),
                    not_needed_reason=f"the corpus already gave a verdict ({award_verdict.split(' (')[0].rstrip('.')})",
                )
                if llm_result:
                    answer = str(llm_result["answer"])
                out = {
                    "answer": answer,
                    "template_answer": template_answer,
                    "tool_trace": retrieval.get("tool_trace", []),
                    "retrieved_texts": retrieval.get("retrieved_texts", []),
                    "citations": local_citations,
                    "verdict_evidence": (
                        {
                            "chunk_id": verdict_evidence.get("chunk_id"),
                            "type": verdict_evidence.get("type"),
                            "pattern": verdict_evidence.get("pattern"),
                            "matched_text": verdict_evidence.get("matched_text"),
                            "citation": (verdict_evidence.get("citations") or [None])[0],
                        }
                        if verdict_evidence
                        else None
                    ),
                    "career_wins": career_wins,
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

    # Non-director queries should fall through to the generic retrieval->LLM
    # flow below. The specialized director flow returned earlier when
    # enabled; otherwise continue to gathering contexts and synthesis.

    # Generic retrieval -> synthesis flow for non-specialized queries.
    # This path is used by tests and for general question answering when the
    # specialized director flow is not enabled.
    try:
        # Use the generic two-hop helper to gather contexts and trace
        retrieval = two_hop_generic(q)
        contexts = [t.get("text", "") for t in retrieval.get("retrieved_texts", [])]
        # Ask the LLM to write the answer from the contexts, but only in fallback mode: in evidence-only
        # mode (allow_fallback=False) it is never called. The decision is recorded in the trace.
        generic_trace = retrieval.setdefault("tool_trace", [])
        llm_result = _consider_llm(q, contexts, max_tokens, allow_fallback, generic_trace, needed=True)
        if llm_result:
            answer = str(llm_result["answer"])
            citations = [int(c) for c in llm_result.get("citations", []) if isinstance(c, int)]
        else:
            answer = "\n\n".join(contexts)  # no LLM answer: the retrieved passages themselves
            citations = []
        return {
            "answer": answer,
            "tool_trace": generic_trace,
            "retrieved_texts": retrieval.get("retrieved_texts", []),
            "citations": citations,
            "step_count": retrieval.get("step_count", 0),
            "token_count": retrieval.get("token_count", 0),
            "budget_exhausted": False,
        }
    except Exception:
        # Generic fallback: empty answer
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
            name = extract_person_name_from_text(text, work_hint=work_hint, chunk_id=cid)
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
                if work.lower() in txt2.lower() and has_directed_by(txt2):
                    name = extract_person_name_from_text(txt2, work_hint=work, chunk_id=cid2)
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


def extract_person_name_from_text(text: str, work_hint: str | None = None, chunk_id: str | None = None) -> str | None:
    """The director named in ``text``.

    With ``work_hint`` the director sentence must be about that film: it names the title, or ``chunk_id``
    is a chunk of the film's own page. A "directed by" sentence about anything else (for example a
    passage on another film's producer, "...Point Break (1991), directed by Kathryn Bigelow") never
    yields a director. Without ``work_hint`` the first "directed by" sentence is used.
    """
    sents = [x.strip() for x in re.split(r"(?<=[.?!])\s+", text or "") if x.strip()]
    if work_hint:
        title_re = re.compile(r"(?<!\w)" + re.escape(work_hint) + r"(?!\w)", re.IGNORECASE)
        on_film_page = _is_film_page(chunk_id, work_hint)
        for sentence in sents:
            if on_film_page or title_re.search(sentence):
                name = _name_after_directed_by(sentence)
                if name:
                    return name
        return None
    for sentence in sents:
        name = _name_after_directed_by(sentence)
        if name:
            return name
    # Last resort, only without a work hint: the longest run of Title-case tokens.
    matches = re.compile(r"\b([A-Z][a-z][\w\-\.'’]+(?:\s+[A-Z][a-z][\w\-\.'’]+){0,3})\b").findall(text or "")
    return max(matches, key=lambda m: len(m.split())).strip(" ,.") if matches else None


# Navigation / boilerplate sections of a Wikipedia plaintext chunk. Their text
# (link titles, reference lists) is not prose and must never count as award evidence.
_NAV_HEADINGS = {
    "see also", "references", "external links", "notes", "further reading",
    "bibliography", "sources", "footnotes",
}
_HEADING_RE = re.compile(r"={2,}\s*([^=]+?)\s*={2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")
# "winning six including Best Director", "won 3 ... including Best Picture": a win
# stated by count plus "including <category>" in an Academy Award sentence.
_WIN_INCLUDING_RE = re.compile(r"\b(?:winning|won|wins)\b[^.;:!?]{0,50}?\bincluding\b[^.;:!?]{0,25}?\bbest\b", re.IGNORECASE)
# "the film won Best Picture, Best Actor and Best Adapted Screenplay (for Puzo and Coppola)"
_WIN_CATEGORY_RE = re.compile(r"\b(?:winning|won)\b[^.;:!?]{0,80}?\bbest\b", re.IGNORECASE)
_AWARD_WORD_RE = re.compile(r"academy awards?|oscars?", re.IGNORECASE)


def strip_navigation_text(text: str) -> str:
    """Return ``text`` without non-prose navigation text.

    Removes "See also" / "References" / "External links" / "Notes" sections (a
    heading and everything up to the next heading), "List of ..." link titles, and
    reference boilerplate. Headings of kept sections are dropped as well.
    """
    if not text:
        return ""
    heads = list(_HEADING_RE.finditer(text))
    pieces = [text[: heads[0].start()]] if heads else [text]
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        if h.group(1).strip().lower() not in _NAV_HEADINGS:
            pieces.append(text[h.end():end])
    kept = []
    for sentence in _SENTENCE_SPLIT_RE.split(" ".join(pieces)):
        s = sentence.strip()
        if not s or re.match(r"(?:lists?\s+of)\b", s, re.IGNORECASE):
            continue
        if re.search(r"archived from the original|\bretrieved (?:on )?\w+ \d{1,2}, \d{4}", s, re.IGNORECASE):
            continue
        kept.append(s)
    return " ".join(kept)


def _sentence_at(text: str, start: int) -> str:
    """The sentence of ``text`` containing character offset ``start``."""
    pos = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        idx = text.find(sentence, pos)
        if idx <= start < idx + len(sentence) + 1:
            return sentence.strip()
        pos = idx + len(sentence)
    return text[:200].strip()


def find_win_including_sentence(text: str) -> str | None:
    """First Academy Award sentence stating wins by category ("winning six including Best
    Director", "won Best Picture, ..."), else None."""
    for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
        if _AWARD_WORD_RE.search(sentence) and (_WIN_INCLUDING_RE.search(sentence) or _WIN_CATEGORY_RE.search(sentence)):
            return sentence.strip()
    return None


# Generic descriptors ("Oscar winners and TV royalty", "an Academy Award winner") say nothing
# about the director unless the director's surname is in the same sentence.
_GENERIC_WIN_PATTERNS = {
    r"academy award winner",
    r"oscar winner",
    r"winner of the academy award (for )?",
    r"winner of an oscar",
}


def _first_qualifying_match(pattern: str, clean_text: str, lowered: str, last_name: str | None):
    """First regex match of ``pattern`` whose sentence is allowed to count.

    Generic descriptors only count when ``last_name`` is in the same sentence.
    Returns (match, sentence) or (None, None).
    """
    for m in re.finditer(pattern, lowered):
        sentence = _sentence_at(clean_text, m.start()) if len(clean_text) == len(lowered) else None
        if pattern in _GENERIC_WIN_PATTERNS and last_name and not (sentence and last_name in sentence.lower()):
            continue
        return m, sentence
    return None, None


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
        # additional common phrasing
        r"won (the )?academy award for",
        r"won (the )?oscar for",
        r"won (the )?best director",
        r"academy award for best director",
        r"won the academy award for best director",
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
        clean_text = strip_navigation_text(raw_text)  # no See also / List of / references
        text = clean_text.lower()
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
        matched_text = None
        for p in patterns_win:
            m, sent = _first_qualifying_match(p, clean_text, text, last_name)
            if m:
                matched_type = "win"
                matched_pattern = p
                matched_text = sent
                break
        if not matched_type:
            ws = find_win_including_sentence(clean_text)
            if ws:
                matched_type = "win"
                matched_pattern = "won/winning ... Best <category>"
                matched_text = ws
        # If no win, look for nomination patterns
        if not matched_type:
            for p in patterns_nom:
                for m in re.finditer(p, text):
                    sent = _sentence_at(clean_text, m.start()) if len(clean_text) == len(text) else None
                    if sent is not None and not _AWARD_WORD_RE.search(sent):
                        continue  # a nomination only counts as an Oscar nomination if the sentence says Academy Award / Oscar
                    matched_type = "nom"
                    matched_pattern = p
                    matched_text = sent
                    break
                if matched_type:
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
                        "matched_text": matched_text,
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
    # If a person page was present among the entries, do a dedicated
    # full-text scan of those person-page chunks for strong 'win' phrases
    # (e.g. "won X Academy Awards", "winner of the Academy Award for ...")
    # and promote any nomination -> win when found. This helps catch
    # wins that are listed on person award pages without repeating the
    # person's full name in the same sentence.
    try:
        if person_page_present:
            for idx, e in enumerate(entries):
                cid = (e.get("chunk_id") or "").lower()
                src = (e.get("source") or "").lower()
                # check if this entry looks like a person page by matching
                # any of the person_variants or awards indicators
                is_person_like = False
                for pv in person_variants:
                    if pv and (pv in cid or pv in src):
                        is_person_like = True
                        break
                if not is_person_like:
                    continue
                raw_text = e.get("text") or ""
                clean_text = strip_navigation_text(raw_text)
                txt_low = clean_text.lower()
                # look for win patterns first
                matched_win = False
                scan_text = None
                for p in patterns_win:
                    m, sent = _first_qualifying_match(p, clean_text, txt_low, last_name)
                    if m:
                        matched_win = True
                        scan_text = sent
                        break
                if not matched_win:
                    scan_text = find_win_including_sentence(clean_text)
                    matched_win = scan_text is not None
                if matched_win:
                    # If a win is found on a person page, prefer it and
                    # insert as highest-priority evidence (if not already present)
                    existing = False
                    for ev in evidence:
                        if ev.get("chunk_id") == e.get("chunk_id"):
                            existing = True
                            # upgrade type to win if it's currently nom
                            if ev.get("type") != "win":
                                ev["type"] = "win"
                                ev.setdefault("citations", [idx + 1])
                            break
                    if not existing:
                        evidence.insert(0, {
                            "entry_index": idx,
                            "chunk_id": e.get("chunk_id"),
                            "source": e.get("source"),
                            "text": raw_text,
                            "type": "win",
                            "pattern": "person_page_fulltext_win_scan",
                            "matched_text": scan_text,
                            "citations": [idx + 1],
                        })
    except Exception:
        pass

    return evidence


_CATEGORY_RE = re.compile(r"\bbest [a-z]+(?: [a-z]+)?", re.IGNORECASE)


def _title_at_start(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _explicit_work_title(sentence: str) -> str | None:
    """The work named explicitly in a passage as "Title (YYYY)", else None.

    "... signed with Universal Pictures for Oppenheimer (2023), which won him Academy Awards"
    -> "Oppenheimer". Nothing is guessed: no "Title (YYYY)" means no title.
    """
    year = re.search(r"\((?:19|20)\d{2}\)", sentence or "")
    if not year:
        return None
    words = []
    for token in reversed(sentence[: year.start()].split()):
        if not token[:1].isupper():
            break
        words.append(token.strip(",;:"))
    return " ".join(reversed(words)) or None


def extract_asked_title(question: str) -> str | None:
    """The work named in "Who directed X ...?", without a trailing "(1997 film)" qualifier.

    Unlike normalize_title this keeps "Part II", so a sequel is not confused with the original.
    """
    m = re.search(r"who directed\s+(.+?)(?:\s+and\b|,|\?|$)", question or "", re.IGNORECASE) or re.search(
        r"director of\s+(.+?)(?:\s+and\b|,|\?|$)", question or "", re.IGNORECASE
    )
    if not m:
        return None
    title = re.sub(r"\s*\([^)]*\)\s*$", "", m.group(1)).strip(" \"'\u201c\u201d")
    return title or None


_DIRECTED_BY_RE = re.compile(
    r"(?:directed|directing|director|helmed|produced|written)\s+(?:and\s+\w+\s+)?by\s+(?:[\w.'\u2019-]+\s+){0,3}$"
)


def _names_recipient(sentence: str, surname: Optional[str]) -> bool:
    """True if ``surname`` appears in ``sentence`` other than as "directed by <surname>"."""
    if not surname:
        return False
    low = sentence.lower()
    for m in re.finditer(r"(?<!\w)" + re.escape(surname) + r"(?!\w)", low):
        if not _DIRECTED_BY_RE.search(low[: m.start()]):
            return True
    return False


def attribute_award_evidence(evidence: List[Dict[str, Any]], film: str | None, director: str | None) -> Dict[str, Any]:
    """Tie award evidence to the film that was asked about.

    * film win: a win passage about the asked film that is credited to the director
      (it lists "Best Director", names the director, or lists no categories at all).
      A film win in other categories (e.g. technical ones) is the film's, not the director's.
    * film nomination: a nomination passage about the asked film.
    * career win: a credited win in a passage that is not about the asked film. It is
      reported separately and never changes the verdict for this film. Its work is named only
      when the passage states it explicitly ("Title (YYYY)" or the film page's own title).
    A passage is "about the film" when it names the film or comes from the film's own page.
    With no film given, every passage counts as about the film (previous behaviour).
    Returns {"verdict", "citations", "evidence", "career_wins"}.
    """
    surname = (director or "").split()[-1].lower() if director else None
    film_l = (film or "").strip().lower()
    film_id = film_l.replace(" ", "_")
    film_name_re = re.compile(r"(?<!\w)" + re.escape(film_l) + r"(?!\w)(?!\s+part\b)", re.IGNORECASE) if film_l else None
    page_re = re.compile(r"^" + re.escape(film_id) + r"(?:_\(.*?\))?_\d+$") if film_l else None

    def about_film(ev: Dict[str, Any], sentence: str) -> bool:
        if not film_l:
            return True
        if film_name_re.search(sentence):
            return True
        return bool(page_re.match((ev.get("chunk_id") or "").lower()))

    def credited(sentence: str) -> bool:
        """A win is the director's only if the sentence names Best Director, or names the director
        as a recipient ("Puzo and Coppola won ..."). Naming the director only as the film's director
        ("the only film directed by Dante to do so") does not credit them, and a sentence that names
        neither is the film's win, not the director's."""
        return "best director" in sentence.lower() or _names_recipient(sentence, surname)

    film_win = film_nom = None
    career: List[Dict[str, Any]] = []
    seen_works = set()
    for ev in evidence:
        sentence = _title_at_start(ev.get("matched_text") or ev.get("text") or "")[:400]
        etype = (ev.get("type") or "").lower()
        if etype == "win":
            if not credited(sentence):
                # The film won, but in categories that are not the director's. If the same
                # sentence says the film was nominated in director-type categories (Picture,
                # Director, Screenplay) the director still has a nomination for this film.
                tail = re.split(r"nominat", sentence, maxsplit=1, flags=re.IGNORECASE)
                if about_film(ev, sentence) and len(tail) > 1 and re.search(
                    r"best (?:director|picture|(?:original |adapted )?(?:screenplay|writing))", tail[1], re.IGNORECASE
                ):
                    film_nom = film_nom or ev
                continue
            if about_film(ev, sentence):
                film_win = film_win or ev
            else:
                src = (ev.get("source") or "").strip()
                if src and not (surname and surname in src.lower()):
                    work = src  # the passage is from another work's own page: that page's title
                else:
                    work = _explicit_work_title(sentence)  # only an explicit "Title (YYYY)"; else None
                key = (work or sentence).lower()
                if key not in seen_works:
                    seen_works.add(key)
                    career.append({
                        "work": work,
                        "chunk_id": ev.get("chunk_id"),
                        "matched_text": sentence,
                        "citation": (ev.get("citations") or [None])[0],
                    })
        elif etype in ("nom", "nomination") and about_film(ev, sentence):
            film_nom = film_nom or ev

    if film_win is not None:
        verdict, used = "Won at least one Academy Award (supported by contexts).", film_win
    elif film_nom is not None:
        verdict, used = "Nominated for Academy Awards (no win evidence in provided contexts).", film_nom
    else:
        verdict, used = "Unknown based on provided contexts.", None
    citations = list((used or {}).get("citations", []))
    for c in career:
        if c["citation"] and c["citation"] not in citations:
            citations.append(c["citation"])
    return {"verdict": verdict, "citations": citations, "evidence": used, "career_wins": career}


def career_wins_sentence(career: List[Dict[str, Any]]) -> str:
    """"Career: won Academy Awards for Oppenheimer (not this film)." Names a work only when the
    passage states it explicitly; otherwise "for another film". Empty when there are none."""
    if not career:
        return ""
    works = list(dict.fromkeys(c["work"] for c in career if c.get("work")))
    if works:
        return "Career: won Academy Awards for " + " and ".join(works) + " (not this film)."
    return "Career: won Academy Awards for another film (not this film)."


def two_hop_director_award_flow(question: str) -> Dict[str, Any]:
    trace: List[Dict[str, Any]] = []
    retrieved_texts: List[Dict[str, str]] = []
    local_token_count = 0
    local_steps = 0

    import re

    # Non-greedy capture stopping at " and", a comma, "?", or end-of-string so
    # titles containing punctuation (e.g. "Titanic (1997 film)") are captured
    # in full instead of failing to match at all (a restrictive character
    # class here previously excluded parentheses).
    m = re.search(r"director of\s+(.+?)(?:\s+and\b|,|\?|$)", question, re.IGNORECASE)
    work = None
    if m:
        work = m.group(1).strip()
    else:
        m = re.search(r"who directed\s+(.+?)(?:\s+and\b|,|\?|$)", question, re.IGNORECASE)
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
                        nm = extract_person_name_from_text(txt_top, work_hint=work, chunk_id=top_cid)
                        if nm:
                            early_name_guess = nm
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
        # Include an award-focused variant so the BM25 fallback (used when no
        # embedding backend/API key is configured) also has a chance to pull
        # in the director's award-evidence chunks in this same first hop,
        # not just the film page.
        f"{work} director academy award",
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
                            nm = extract_person_name_from_text(txt_try, work_hint=work, chunk_id=cid)
                            if nm:
                                early_name_guess = nm
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
                    nm = extract_person_name_from_text(txtp, work_hint=work, chunk_id=pid)
                    if nm:
                        early_name_guess = nm
                    break
    except Exception:
        pass
    # Deliberate reset: the early guesses above are discarded and the director is
    # re-derived by the scan below. Removing the reset made the Inception test in
    # tests/test_award_regression.py answer "Unknown". Likely cause: the early
    # guess can come from a person page rather than the film page, so the later
    # scan is skipped. Not confirmed (nothing reads `name` before this line).
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
                        if work.lower() in sl and has_directed_by(sl):
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
                        name = extract_person_name_from_text(text, work_hint=work, chunk_id=cid)
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
                name = extract_person_name_from_text(text, work_hint=work, chunk_id=cid)
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
                    if any(wv in s for wv in work_variants) and has_directed_by(s):
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
                        if any(wv in window_low for wv in work_variants) and has_directed_by(window_low):
                            # find the sentence in the window with directed-by
                            directed_idx = None
                            for j in range(start, end + 1):
                                if has_directed_by(sents_low[j]):
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
                        chunks_path = os.path.join(DATA_DIR, "chunks.pkl")
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
        from .retrieval_tools import find_award_mentions

        try:
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
        except Exception as e:
            logging.getLogger(__name__).warning("find_award_mentions failed: %s", e)
            trace.append({"tool": "find_award_mentions", "args": {"name": person}, "error": str(e)})

    return {
        "tool_trace": trace,
        "retrieved_texts": retrieved_texts,
        "step_count": local_steps,
        "token_count": local_token_count,
    }


def build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the A-RAG evidence-only two-hop baseline (deterministic recipe, not an LLM agent)"
    )
    parser.add_argument("--question", type=str, default="Who directed Titanic?", help="Question to ask")
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Most retrieval tool calls for this question; the run stops using tools after that and reports budget_exhausted",
    )
    parser.add_argument("--max_tokens", type=int, default=2000, help="Cap on the LLM answer length (only used when the LLM is called)")
    parser.add_argument(
        "--allow-fallback", dest="allow_fallback", action="store_true",
        help="Allow the LLM to answer from external knowledge when the corpus has no answer (needs ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--no-allow-fallback", dest="allow_fallback", action="store_false",
        help="Do not allow LLM fallback (answer strictly from the corpus)",
    )
    parser.add_argument(
        "--enable-director-flow", dest="enable_director_flow", action="store_true",
        help="Use the director->award two-hop flow for director questions (default ON in the CLI)",
    )
    parser.add_argument(
        "--no-enable-director-flow", dest="enable_director_flow", action="store_false",
        help="Disable the director->award two-hop flow",
    )
    parser.set_defaults(allow_fallback=True, enable_director_flow=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry. When vector search is off, the degraded line goes to stderr BEFORE the answer."""
    import sys

    from arag.health import degraded_line
    from arag.retrieval_tools import vector_search_status

    args = build_parser().parse_args(argv)
    enabled, reason = vector_search_status()  # loads the indexes, which logs the health line
    if not enabled:
        print(degraded_line(reason), file=sys.stderr, flush=True)
    result = run_agent(
        args.question,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        allow_fallback=args.allow_fallback,
        enable_director_flow=args.enable_director_flow,
    )
    vs = result["vector_search"]
    if not vs["enabled"] and enabled:  # it failed during this run, not at startup
        print(degraded_line(vs["reason"]), file=sys.stderr, flush=True)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    from dotenv import load_dotenv

    from arag.runtime_ops import setup_logging

    load_dotenv()
    setup_logging()
    raise SystemExit(main())

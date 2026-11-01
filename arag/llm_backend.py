"""LLM backend moved into arag package."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, List

from arag.runtime_ops import BudgetExceededError, get_global_budget, get_global_rate_limiter

log = logging.getLogger(__name__)

LLM_PROVIDERS = ("anthropic", "openai")
OPENAI_MISSING_MESSAGE = (
    "ARAG_LLM_PROVIDER=openai needs the 'openai' package, which is not installed "
    "(it is not in requirements.txt). Install it with `pip install openai`, or unset ARAG_LLM_PROVIDER to use Anthropic."
)


def get_llm_provider() -> str:
    """The LLM provider for answers: ARAG_LLM_PROVIDER, default "anthropic".

    OpenAI is used only when ARAG_LLM_PROVIDER=openai is set explicitly. Having OPENAI_API_KEY in the
    environment does nothing by itself.
    """
    provider = (os.environ.get("ARAG_LLM_PROVIDER") or "anthropic").strip().lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unknown ARAG_LLM_PROVIDER={provider!r}; expected one of {list(LLM_PROVIDERS)}")
    return provider


def llm_available() -> bool:
    import importlib

    provider = get_llm_provider()
    key, package = ("OPENAI_API_KEY", "openai") if provider == "openai" else ("ANTHROPIC_API_KEY", "anthropic")
    if not os.environ.get(key):
        return False
    try:
        importlib.import_module(package)
        return True
    except Exception:
        if provider == "openai":
            log.warning(OPENAI_MISSING_MESSAGE)  # an explicit choice must not fail silently
        return False


def _parse_reply(text: str, allow_fallback: bool) -> dict[str, Any]:
    """Turn the model's raw reply into {answer, citations, used_fallback}.

    The model is asked for a JSON object; a reply wrapped in a markdown code
    fence is unwrapped first. Anything that is not valid JSON is returned as
    a plain answer with no citations.
    """
    raw = (text or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    answer, citations = raw, []
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and "answer" in parsed:
        answer = str(parsed.get("answer", ""))
        cites = parsed.get("citations")
        if isinstance(cites, list):
            citations = [c for c in cites if isinstance(c, int)]
    used_fallback = allow_fallback and "external knowledge" in answer.lower()
    return {"answer": answer, "citations": citations, "used_fallback": used_fallback}


def _record_usage(budget: Any, max_tokens: int) -> None:
    try:
        budget.consume_tokens(max_tokens)
    except BudgetExceededError as e:
        log.warning("Budget exceeded after LLM call: %s", e)
    from arag.runtime_ops import inc_api_call_metric, inc_token_metric

    inc_api_call_metric(1)
    inc_token_metric(max_tokens)


def synthesize_answer(
    question: str,
    contexts: List[str],
    model: str | None = None,
    max_tokens: int = 512,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    # When allow_fallback is True, the assistant is permitted to use external
    # world knowledge to answer if the provided contexts are insufficient.
    # If fallback is used the assistant must clearly indicate it used knowledge
    # outside the provided contexts.
    if allow_fallback:
        system = (
            "You are a concise assistant. First attempt to answer using only "
            "the provided contexts. If the contexts do not contain the answer, "
            "you are allowed to use your general world knowledge to answer, but "
            "you MUST add a short note saying: 'Note: this answer uses external "
            "knowledge outside the provided contexts.'"
            "\nIMPORTANT: Return output as a JSON object with exactly two fields: "
            "`answer` (string) and `citations` (list of integers referencing the "
            "provided contexts, 1-based). Example: {"
            '"answer": "Short answer.", "citations": [1,3]}'
        )
    else:
        # For context-only mode, encourage the assistant to answer any parts
        # of the question that are supported by the provided contexts and to
        # explicitly mark parts that are not supported. This avoids returning
        # a blanket "I don't know" when contexts do contain partial facts
        # (e.g. the director's name) but lack evidence for other sub-questions
        # (e.g. award wins).
        system = (
            "You are a concise assistant. Use ONLY the provided contexts to "
            "answer the question. If some parts of the question are supported "
            "by the contexts, answer those parts and provide the 1-based indices "
            "of the contexts used in a `citations` list. For any sub-question or "
            "claim that is NOT supported by the contexts, explicitly state: "
            "'Unknown based on provided contexts' for that part. Do not use "
            "external knowledge.\nIMPORTANT: Return output as a JSON object "
            "with exactly two fields: `answer` (string) and `citations` (list "
            "of integers referencing the provided contexts, 1-based). Example: "
            '{"answer": "Director: Scott Derrickson. Unknown based on provided contexts whether he won Academy Awards.", "citations": [1]}'
        )
    joined_ctx = "\n\n---\n\n".join(contexts[-6:])
    user = (
        f"CONTEXT:\n{joined_ctx}\n\nQUESTION: {question}\n\n"
        "Return a short factual answer (1-3 sentences) and provide the "
        "indices of any contexts you used in the `citations` list in the "
        "JSON output."
    )

    provider = get_llm_provider()
    if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise RuntimeError(OPENAI_MISSING_MESSAGE) from e
        try:
            from openai import OpenAI  # type: ignore

            api_key = os.environ.get("OPENAI_API_KEY")
            rl = get_global_rate_limiter()
            budget = get_global_budget()
            if not rl.acquire(timeout=5.0):
                raise RuntimeError("Rate limiter timeout while attempting LLM call")
            budget.consume_api_call(1)
            openai_client: Any = OpenAI(api_key=api_key)
            chat_model = model or os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
            resp = openai_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
            choice0 = resp.choices[0]
            m = choice0.message if hasattr(choice0, "message") else choice0.get("message")
            content = m.content if hasattr(m, "content") else (m or {}).get("content")
            _record_usage(budget, max_tokens)
            return _parse_reply(content or "", allow_fallback)
        except BudgetExceededError:
            raise
        except Exception as e:
            raise RuntimeError(f"OpenAI chat call failed: {e}")

    if provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            rl = get_global_rate_limiter()
            budget = get_global_budget()
            if not rl.acquire(timeout=5.0):
                raise RuntimeError("Rate limiter timeout while attempting Anthropic call")
            budget.consume_api_call(1)
            anthropic_client: Any = anthropic.Anthropic(api_key=api_key)
            model_name = model or os.environ.get("ANTHROPIC_MODEL") or "claude-haiku-5-5"
            res = anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                getattr(block, "text", "") for block in res.content if getattr(block, "type", "text") == "text"
            )
            _record_usage(budget, max_tokens)
            return _parse_reply(text, allow_fallback)
        except BudgetExceededError:
            raise
        except Exception as e:
            raise RuntimeError(f"Anthropic call failed: {e}")

    raise RuntimeError(
        "No LLM backend configured (set ANTHROPIC_API_KEY; OpenAI is used only with ARAG_LLM_PROVIDER=openai and OPENAI_API_KEY)"
    )

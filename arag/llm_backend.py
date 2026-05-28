"""LLM backend moved into arag package."""

from __future__ import annotations

import logging
import os
from typing import Any, List

from arag.runtime_ops import BudgetExceededError, get_global_budget, get_global_rate_limiter

log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def llm_available() -> bool:
    import importlib

    if os.environ.get("OPENAI_API_KEY"):
        try:
            importlib.import_module("openai")
            return True
        except Exception:
            return False
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            importlib.import_module("anthropic")
            return True
        except Exception:
            return False
    return False


def synthesize_answer(
    question: str,
    contexts: List[str],
    model: str | None = None,
    max_tokens: int = 512,
    allow_fallback: bool = False,
) -> tuple[str, list[int], bool]:
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

    if os.environ.get("OPENAI_API_KEY"):
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
            m = None
            if hasattr(choice0, "message"):
                m = choice0.message
            elif isinstance(choice0, dict) and "message" in choice0:
                m = choice0["message"]
            out = ""
            if m is None:
                out = str(choice0)
            else:
                if hasattr(m, "content"):
                    out = m.content or ""
                elif isinstance(m, dict) and "content" in m:
                    out = m["content"] or ""
                else:
                    out = str(m)
            parsed: object | None = None
            try:
                import json as _json

                parsed = _json.loads(out.strip()) if isinstance(out, str) else None
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and "answer" in parsed:
                ans = str(parsed.get("answer", ""))
                # Return a plain string for backwards compatibility with tests
                return ans
            try:
                budget.consume_tokens(max_tokens)
            except BudgetExceededError as e:
                log.warning("Budget exceeded after LLM call: %s", e)
            try:
                from arag.runtime_ops import inc_api_call_metric, inc_token_metric

                inc_api_call_metric(1)
                inc_token_metric(max_tokens)
            except Exception:
                pass
            # Fall back to returning a plain string answer with no citations
            return str(out or "")
        except Exception as e:
            raise RuntimeError(f"OpenAI chat call failed: {e}")

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # type: ignore

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            rl = get_global_rate_limiter()
            budget = get_global_budget()
            if not rl.acquire(timeout=5.0):
                raise RuntimeError("Rate limiter timeout while attempting Anthropic call")
            budget.consume_api_call(1)
            anthropic_client: Any = anthropic.Client(api_key=api_key)
            prompt = f"{system}\n\n{user}"
            model_name = model or os.environ.get("ANTHROPIC_MODEL") or "claude-2.1"
            res = getattr(anthropic_client.completions, "create")(model=model_name, prompt=prompt, max_tokens_to_sample=max_tokens)
            out = (res["completion"] if isinstance(res, dict) and "completion" in res else res.get("completion", str(res)))
            parsed = None
            try:
                import json as _json

                parsed = _json.loads(out.strip()) if isinstance(out, str) else None
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and "answer" in parsed:
                ans = str(parsed.get("answer", ""))
                return ans
            try:
                budget.consume_tokens(max_tokens)
            except BudgetExceededError as e:
                log.warning("Budget exceeded after Anthropic call: %s", e)
            try:
                from arag.runtime_ops import inc_api_call_metric, inc_token_metric

                inc_api_call_metric(1)
                inc_token_metric(max_tokens)
            except Exception:
                pass
            return str(out or "")
        except Exception as e:
            raise RuntimeError(f"Anthropic call failed: {e}")

    raise RuntimeError("No LLM backend configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")

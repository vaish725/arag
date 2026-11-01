"""Evidence-only + a question with no specialised route: never show retrieved passages as the answer."""
import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from arag import agent_loop  # noqa: E402
from arag.agent_loop import NO_ROUTE_MESSAGE, run_agent  # noqa: E402

pytestmark = pytest.mark.needs_model

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
MODE_EVIDENCE = "evidence-only (no LLM)"
MODE_LLM = "evidence + LLM fallback (LLM only if the corpus gives no verdict)"
QUESTION = "Which films has James Cameron produced?"
MESSAGE = (
    "In evidence-only mode this demo answers questions like 'Who directed X and did they win an Oscar?'. "
    "For other questions, turn on LLM fallback."
)


def ask(question, mode=MODE_EVIDENCE):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox[0].select(mode).run()
    at.text_input[0].set_value(question).run()
    at.button[0].click().run()
    assert not at.exception
    return at


def test_message_text_is_exactly_the_requested_one():
    assert NO_ROUTE_MESSAGE == MESSAGE


def test_result_has_message_and_three_cleaned_passages(fake_anthropic):
    res = run_agent(QUESTION, allow_fallback=False, enable_director_flow=True)
    assert res["answer"] == MESSAGE and res["no_specialised_route"] is True
    passages = res["related_passages"]
    assert len(passages) == 3
    for p in passages:
        assert "==" not in p["text"]
        assert p["text"].rstrip().endswith((".", "!", "?", '"', ")", "…"))
    assert not any(p["text"] in res["answer"] for p in passages)
    assert fake_anthropic.calls == 0 and res["llm"]["decision"] == "skipped"


def test_app_shows_message_and_collapsed_related_passages(fake_anthropic):
    at = ask(QUESTION)
    assert "Answer" in [s.value for s in at.subheader]
    assert MESSAGE in " ".join(m.value for m in at.markdown)
    expanders = {e.label: e for e in at.expander}
    related = expanders["Related passages (not an answer)"]
    assert related.proto.expanded is False
    shown = " ".join(c.value for c in related.caption)
    assert shown and "==" not in shown
    assert len([m for m in related.markdown if m.value.startswith("**[[")]) == 3
    assert "Show full passage" not in expanders and "Evidence" not in [s.value for s in at.subheader]
    assert fake_anthropic.calls == 0
    # the raw dump is gone: the answer area holds only the message
    assert not any(len(m.value) > 400 for m in at.markdown)


def test_the_director_route_in_evidence_only_is_unchanged(fake_anthropic):
    at = ask("Who directed La La Land and did they win any Academy Awards?")
    text = " ".join(m.value for m in at.markdown)
    assert "Damien Chazelle" in text and "Won at least one" in text
    assert MESSAGE not in text and "Related passages (not an answer)" not in [e.label for e in at.expander]
    assert "Evidence" in [s.value for s in at.subheader]
    assert fake_anthropic.calls == 0


def test_with_llm_fallback_the_generic_route_still_asks_the_llm(fake_anthropic):
    at = ask(QUESTION, MODE_LLM)
    assert fake_anthropic.calls == 1
    assert MESSAGE not in " ".join(m.value for m in at.markdown)


def _only_message_and_collapsed_passages(at, message_start):
    text = " ".join(m.value for m in at.markdown)
    assert message_start in text
    assert not any(len(m.value) > 400 for m in at.markdown), "no raw passage dump in the answer area"
    related = {e.label: e for e in at.expander}["Related passages (not an answer)"]
    assert related.proto.expanded is False
    shown = " ".join(c.value for c in related.caption)
    assert shown and "==" not in shown
    assert len([m for m in related.markdown if m.value.startswith("**[[")]) == 3
    assert "Evidence" not in [s.value for s in at.subheader]


def test_llm_failure_on_the_generic_route_shows_a_short_reason_and_collapsed_passages(fake_anthropic):
    fake_anthropic.fail = "service overloaded\n" + "x" * 500
    res = run_agent(QUESTION, allow_fallback=True, enable_director_flow=True)
    assert res["answer"].startswith("The LLM call failed: ") and "service overloaded" in res["answer"]
    assert "\n" not in res["answer"] and len(res["answer"]) < 220, "the reason is short and one line"
    assert len(res["related_passages"]) == 3 and res["llm"]["decision"] == "failed"
    assert not any(p["text"] in res["answer"] for p in res["related_passages"])
    at = ask(QUESTION, MODE_LLM)
    _only_message_and_collapsed_passages(at, "The LLM call failed: ")
    assert not any("Showing the corpus-evidence answer" in e.value for e in at.error)


def test_director_flow_exception_shows_could_not_answer_and_collapsed_passages(monkeypatch, fake_anthropic):
    def boom(*a, **k):
        raise RuntimeError("extractor broke")

    monkeypatch.setattr(agent_loop, "attribute_award_evidence", boom)
    res = run_agent("Who directed La La Land and did they win any Academy Awards?", allow_fallback=False, enable_director_flow=True)
    assert res["answer"] == "Couldn't answer this question" and len(res["related_passages"]) == 3
    assert not any(p["text"] in res["answer"] for p in res["related_passages"])
    at = ask("Who directed La La Land and did they win any Academy Awards?")
    _only_message_and_collapsed_passages(at, "Couldn't answer this question")
    assert fake_anthropic.calls == 0, "evidence-only: 0 LLM calls on the exception path too"

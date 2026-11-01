"""Streamlit: LLM status next to the answer, evidence sentences with a collapsed full passage, a collapsed
tool trace, and a mode label that matches the code. Fake LLM client only."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.needs_model

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
MODE_EVIDENCE = "evidence-only (no LLM)"
MODE_LLM = "evidence + LLM fallback (LLM only if the corpus gives no verdict)"
INCEPTION = "Who directed Inception and did they win any Academy Awards?"
LA_LA_LAND = "Who directed La La Land and did they win any Academy Awards?"


def ask(question, mode=MODE_EVIDENCE):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox[0].select(mode).run()
    at.text_input[0].set_value(question).run()
    at.button[0].click().run()
    assert not at.exception
    return at


def markdown(at):
    return " ".join(m.value for m in at.markdown)


def test_the_mode_label_says_what_the_code_does():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert list(at.selectbox[0].options) == [MODE_EVIDENCE, MODE_LLM]
    assert "BM25" not in MODE_LLM  # BM25 only runs as a fallback, so it is not part of the mode name


def test_llm_called_is_shown_with_model_and_tokens(fake_anthropic):
    at = ask(INCEPTION, MODE_LLM)
    assert fake_anthropic.calls == 1
    assert any("LLM called" in i.value and "claude-haiku-5-5" in i.value and "812 input / 64 output tokens" in i.value for i in at.info)
    assert "LLM ANSWER" in markdown(at)
    assert any("What the corpus evidence alone says" in c.value and "Christopher Nolan" in c.value for c in at.caption)


def test_llm_failure_is_shown_and_the_template_answer_is_still_there(fake_anthropic):
    fake_anthropic.fail = "service overloaded"
    at = ask(INCEPTION, MODE_LLM)
    assert any("LLM call failed" in e.value and "service overloaded" in e.value and "corpus-evidence answer" in e.value for e in at.error)
    text = markdown(at)
    assert "Director: Christopher Nolan." in text and "Unknown" in text and "LLM ANSWER" not in text


def test_llm_skipped_is_shown_with_its_reason():
    at = ask(INCEPTION, MODE_EVIDENCE)
    assert any(c.value.startswith("LLM not called: LLM fallback not allowed") for c in at.caption)
    at = ask(LA_LA_LAND, MODE_LLM)
    assert any("LLM not called: the corpus already gave a verdict" in c.value for c in at.caption)


def test_each_citation_shows_the_evidence_sentence_and_what_it_supports():
    at = ask(LA_LA_LAND)
    text = markdown(at)
    assert "supports: the director" in text and "supports: the Oscar verdict" in text
    quotes = [m.value for m in at.markdown if m.value.startswith("> **")]
    assert any("winning six including Best Director" in q for q in quotes)
    assert all(q.rstrip("*").endswith((".", "…", "\"", ")")) or "…" in q for q in quotes), quotes  # not cut mid-sentence
    assert "[1] La La Land" in text and "https://en.wikipedia.org/wiki/La_La_Land" in text  # title + link, as before


def test_the_full_passage_and_the_tool_trace_are_collapsed():
    at = ask(LA_LA_LAND)
    passages = [e for e in at.expander if e.label == "Show full passage"]
    assert len(passages) == 2 and all(e.proto.expanded is False for e in passages)  # one per citation
    (trace,) = [e for e in at.expander if e.label == "Tool trace"]
    assert trace.proto.expanded is False
    assert len(at.json) > 0 and len(at.json) == len(trace.json), "every trace entry is inside the collapsed expander"


def test_step_count_shown_equals_the_number_of_recorded_tool_calls():
    at = ask(LA_LA_LAND)
    trace = [e for e in at.expander if e.label == "Tool trace"][0]
    tool_entries = [j for j in trace.json if '"tool": "llm"' not in j.value]
    count_line = next(m.value for m in at.markdown if m.value.startswith("Step count:"))
    shown = int(count_line.split("Step count:")[1].split()[0].strip("`"))
    assert shown == len(tool_entries)

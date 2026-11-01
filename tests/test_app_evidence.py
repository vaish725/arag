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
LLM_LABEL = "LLM answer (model knowledge, not from the corpus)"
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


def tree(at):
    """The page as (depth, element type, text) in document order, including nested containers."""
    out = []

    def walk(block, depth):
        for k in sorted(block.children):
            e = block.children[k]
            value = getattr(e, "value", None)
            out.append((depth, getattr(e, "type", type(e).__name__), value if isinstance(value, str) else ""))
            if hasattr(e, "children"):
                walk(e, depth + 1)

    walk(at.main, 0)
    return out


def test_llm_called_is_shown_with_model_and_tokens(fake_anthropic):
    at = ask(INCEPTION, MODE_LLM)
    assert fake_anthropic.calls == 1
    assert any("LLM called" in i.value and "claude-haiku-5-5" in i.value and "812 input / 64 output tokens" in i.value for i in at.info)


def test_the_llm_answer_is_visibly_separated_from_the_evidence_answer(fake_anthropic):
    at = ask(INCEPTION, MODE_LLM)
    nodes = tree(at)
    kinds = [(d, t, v) for d, t, v in nodes]
    ev_head = next(i for i, (d, t, v) in enumerate(kinds) if t == "subheader" and v == "Evidence answer (from the corpus)")
    llm_head = next(i for i, (d, t, v) in enumerate(kinds) if t == "subheader" and v == LLM_LABEL)
    assert ev_head < llm_head
    evidence_text = next(i for i, (d, t, v) in enumerate(kinds) if t == "markdown" and "Director: Christopher Nolan." in v)
    llm_text = next(i for i, (d, t, v) in enumerate(kinds) if t == "markdown" and v == "LLM ANSWER")
    assert ev_head < evidence_text < llm_head < llm_text, "the evidence answer comes first, the LLM answer after its label"
    # the LLM text sits inside its own bordered container, one level deeper than the evidence answer
    assert kinds[llm_text][0] > kinds[evidence_text][0] and kinds[llm_text - 1][1] == "flex_container"
    assert "LLM ANSWER" not in kinds[evidence_text][2] and "Christopher Nolan" not in kinds[llm_text][2]
    assert [s.value for s in at.subheader].count("Answer") == 0, "no single mixed 'Answer' block when the LLM answered"


def test_without_an_llm_answer_there_is_no_llm_label(fake_anthropic):
    at = ask(INCEPTION, MODE_EVIDENCE)
    assert LLM_LABEL not in [s.value for s in at.subheader]
    assert "Answer" in [s.value for s in at.subheader] and fake_anthropic.calls == 0


def test_evidence_only_mode_never_calls_the_llm_in_the_app_even_with_a_key(fake_anthropic):
    for question in (INCEPTION, LA_LA_LAND, "What is the capital of France?"):
        at = ask(question, MODE_EVIDENCE)
        assert any(c.value.startswith("LLM not called: evidence-only mode") for c in at.caption)
    assert fake_anthropic.calls == 0


def test_llm_failure_is_shown_and_the_template_answer_is_still_there(fake_anthropic):
    fake_anthropic.fail = "service overloaded"
    at = ask(INCEPTION, MODE_LLM)
    assert any("LLM call failed" in e.value and "service overloaded" in e.value and "corpus-evidence answer" in e.value for e in at.error)
    text = markdown(at)
    assert "Director: Christopher Nolan." in text and "Unknown" in text and "LLM ANSWER" not in text


def test_llm_skipped_is_shown_with_its_reason():
    at = ask(INCEPTION, MODE_EVIDENCE)
    assert any(c.value.startswith("LLM not called: evidence-only mode") for c in at.caption)
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

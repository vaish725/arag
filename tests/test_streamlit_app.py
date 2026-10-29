from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
QUESTION = "Who directed Inception and did they win any Academy Awards?"
MODES = ["evidence-only (no LLM)", "semantic + BM25 + optional LLM fallback"]


def _shown(at):
    return " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("mode", MODES)
def test_inception_answer_shown_in_both_modes(mode):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[0].select(mode).run()
    at.text_input[0].set_value(QUESTION).run()
    at.button[0].click().run()
    assert not at.exception
    assert "Christopher Nolan" in _shown(at)


def test_toggling_fallback_checkbox_keeps_results():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[0].select(MODES[1]).run()
    at.text_input[0].set_value(QUESTION).run()
    at.button[0].click().run()
    at.checkbox[0].uncheck().run()
    assert "Christopher Nolan" in _shown(at)

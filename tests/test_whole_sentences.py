"""Display-only: evidence and full passages show whole sentences and no wiki heading markup.
Fixtures are the exact chunks of the live run (Christopher_Nolan_0/1/24/25/26)."""
import json
import re
from pathlib import Path

import pytest

from arag import agent_loop, retrieval_tools
from arag.agent_loop import build_citation_details, evidence_window, whole_sentence_passage

CH = json.loads((Path(__file__).parent / "fixtures_nolan_chunks.json").read_text())
HEADING = re.compile(r"={2,}")
NOLAN_1_SENTENCE = (
    "After the release of Tenet (2020), Nolan parted ways with longtime distributor Warner Bros. Pictures and "
    "signed with Universal Pictures for Oppenheimer (2023), which won him Academy Awards for Best Director and Best Picture."
)
NOLAN_25_SENTENCE = "Among the film's numerous accolades, Nolan won the Academy Awards for Best Director and Best Picture."


@pytest.fixture(autouse=True)
def fake_chunks(monkeypatch):
    monkeypatch.setattr(retrieval_tools, "chunk_text", lambda cid: CH.get(cid, ""))


def _result(cid, matched):
    return {
        "retrieved_texts": [{"chunk_id": cid, "source": "Christopher Nolan", "text": CH[cid]}],
        "citations": [1],
        "verdict_evidence": {"citation": 1, "matched_text": matched},
    }


def test_nolan_1_evidence_is_completed_from_previous_chunk():
    matched = "Warner Bros. Pictures and signed with Universal Pictures for Oppenheimer (2023), which won him Academy Awards for Best Director and Best Picture."
    sp = build_citation_details(_result("Christopher_Nolan_1", matched), "q")[0]["supports"][0]
    assert sp["sentence"] == NOLAN_1_SENTENCE
    assert not sp["sentence"].startswith("…") and "…" not in sp["sentence"]
    assert sp["before"].endswith("Dunkirk (2017).") and sp["after"].startswith("Three of Nolan's films")


def test_without_the_previous_chunk_the_cut_is_marked():
    w = evidence_window(CH["Christopher_Nolan_1"], "Warner Bros. Pictures and signed with Universal Pictures for Oppenheimer (2023), which won him Academy Awards for Best Director and Best Picture.", "Christopher_Nolan_1")
    assert w["sentence"].startswith("… Warner Bros.")


def test_nolan_25_after_neighbour_is_a_whole_sentence():
    sp = build_citation_details(_result("Christopher_Nolan_25", NOLAN_25_SENTENCE), "q")[0]["supports"][0]
    assert sp["sentence"] == NOLAN_25_SENTENCE
    assert sp["after"] == 'The success of Oppenheimer further highlighted Nolan\'s status as one of Hollywood\'s most "bankable" directors.'
    assert sp["before"].endswith("2023.")


def test_evidence_cut_at_the_end_is_completed_from_the_next_chunk():
    # the real end of Christopher_Nolan_25 is cut mid-sentence ("... on his perilous journey to return")
    cut = "It follows Odysseus (portrayed by Matt Damon), the legendary Greek king of Ithaca, on his perilous journey to return"
    w = evidence_window(CH["Christopher_Nolan_25"], cut, "Christopher_Nolan_25", next_text=CH["Christopher_Nolan_26"])
    assert w["sentence"].endswith("and his reunion with his wife, Penelope.")
    assert "\u2026" not in w["sentence"]
    assert evidence_window(CH["Christopher_Nolan_25"], cut, "Christopher_Nolan_25")["sentence"].endswith(" \u2026")


@pytest.mark.parametrize("cid", ["Christopher_Nolan_0", "Christopher_Nolan_1", "Christopher_Nolan_24", "Christopher_Nolan_25", "Christopher_Nolan_26"])
def test_full_passage_has_whole_sentences_and_no_headings(cid):
    out = whole_sentence_passage(CH[cid], cid)
    assert not HEADING.search(out)
    body = out.removeprefix("… ").removesuffix(" …")
    assert body.rstrip().endswith(agent_loop._TERMINAL) or out.endswith(" …")
    assert out.startswith("… ") == (not cid.endswith("_0"))
    # the dropped partial sentences are really gone, and nothing else is lost
    sentences = agent_loop.split_sentences(CH[cid])
    assert body.split()[0] != sentences[0].split()[0] or cid.endswith("_0")


def test_full_passage_trims_the_exact_nolan_edges():
    p1 = whole_sentence_passage(CH["Christopher_Nolan_1"], "Christopher_Nolan_1")
    assert p1.startswith("… Three of Nolan's films") and "Warner Bros. Pictures and signed" not in p1
    assert p1.endswith("also a filmmaker. …") and "raised Catholic in" not in p1
    assert "Early life and education" not in p1
    p25 = whole_sentence_passage(CH["Christopher_Nolan_25"], "Christopher_Nolan_25")
    assert p25.startswith("… The film received critical acclaim.") and "weeks before or after" not in p25
    assert p25.endswith("July 2026. …") and "journey to return" not in p25


def test_headings_are_stripped_from_evidence_text():
    text = "He was born in London. == Early life == He studied English literature. Then he moved."
    w = evidence_window(text, "He studied English literature.", "X_0")
    assert w["sentence"] == "He studied English literature." and "==" not in w["before"] + w["after"]


def test_neighbour_lookup_is_not_a_tool_call(monkeypatch):
    from arag.agent_loop import _STEP_BUDGET, _StepBudget

    monkeypatch.setattr(retrieval_tools, "chunk_read", lambda cid: pytest.fail("display must not call the chunk_read tool"))
    budget = _StepBudget(10)
    token = _STEP_BUDGET.set(budget)
    try:
        build_citation_details(_result("Christopher_Nolan_1", "Warner Bros. Pictures"), "q")
    finally:
        _STEP_BUDGET.reset(token)
    assert budget.used == 0 and budget.log == []

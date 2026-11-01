"""Citations show the exact evidence sentence trimmed at sentence boundaries (at most one neighbour each side),
what it supports, and keep the full passage for a collapsed expander."""
import pytest

from arag.agent_loop import build_citation_details, evidence_window, split_sentences


def test_abbreviations_and_initials_do_not_split_sentences():
    text = "Nolan left Warner Bros. Pictures and signed with Universal. C. Robert Cargill co-wrote it. Done."
    assert split_sentences(text) == [
        "Nolan left Warner Bros. Pictures and signed with Universal.",
        "C. Robert Cargill co-wrote it.",
        "Done.",
    ]


def test_a_fragment_from_a_naive_split_is_expanded_to_the_real_sentence():
    chunk = "Intro here. Nolan left Warner Bros. Pictures and signed with Universal for Oppenheimer (2023). Later sentence."
    w = evidence_window(chunk, "Pictures and signed with Universal for Oppenheimer (2023).", "Christopher_Nolan_0")
    assert w["sentence"] == "Nolan left Warner Bros. Pictures and signed with Universal for Oppenheimer (2023)."
    assert w["before"] == "Intro here." and w["after"] == "Later sentence."


def test_at_most_one_neighbour_each_side():
    chunk = "One. Two. Three evidence here. Four. Five."
    w = evidence_window(chunk, "Three evidence here.", "Page_0")
    assert (w["before"], w["sentence"], w["after"]) == ("Two.", "Three evidence here.", "Four.")


def test_a_cut_off_fragment_at_the_start_of_a_later_chunk_is_not_shown_as_a_neighbour():
    # chunk 3 of its page starts mid-sentence: "ing the war, he left." is a fragment
    chunk = "ing the war, he left. The real evidence sentence is here. A complete follow-up."
    w = evidence_window(chunk, "The real evidence sentence is here.", "Page_3")
    assert w["before"] == "" and w["sentence"] == "The real evidence sentence is here."
    # the same text as the FIRST chunk of a page starts at a real sentence start, so the neighbour is kept
    assert evidence_window(chunk, "The real evidence sentence is here.", "Page_0")["before"] == "ing the war, he left."


def test_a_cut_off_fragment_at_the_end_is_not_shown_as_a_neighbour():
    chunk = "Complete evidence sentence. A trailing fragment that never ends because the chunk was cut"
    w = evidence_window(chunk, "Complete evidence sentence.", "Page_0")
    assert w["after"] == ""


def test_an_evidence_sentence_that_is_itself_a_chunk_edge_fragment_gets_an_ellipsis():
    start = evidence_window("Pictures and signed with Universal for Oppenheimer. After that.", "Pictures and signed with Universal for Oppenheimer.", "Page_4")
    assert start["sentence"].startswith("… Pictures and signed")
    end = evidence_window("First. The evidence that is cut off here", "The evidence that is cut off here", "Page_0")
    assert end["sentence"].endswith("here …")


def test_headings_are_removed_from_neighbours():
    chunk = "The evidence sentence. == Early life and education == Chazelle was born in Providence."
    w = evidence_window(chunk, "The evidence sentence.", "Page_0")
    assert w["after"] == "Chazelle was born in Providence." and "==" not in w["after"]


def test_an_evidence_sentence_not_found_in_the_chunk_is_still_shown():
    w = evidence_window("Some other text entirely.", "A sentence that is not there.", "Page_0")
    assert w["sentence"] == "A sentence that is not there." and w["before"] == "" and w["after"] == ""


RESULT = {
    "answer": "Director: Damien Chazelle. Won at least one Academy Award (supported by contexts). Citations: [1, 2, 3].",
    "citations": [1, 2, 3],
    "retrieved_texts": [
        {"chunk_id": "La_La_Land_0", "source": "La La Land", "text": "La La Land is a 2016 film written and directed by Damien Chazelle. It stars Ryan Gosling."},
        {"chunk_id": "Damien_Chazelle_0", "source": "Damien Chazelle", "text": "Intro. His success came with La La Land, which was nominated for 14 Academy Awards, winning six including Best Director. Next."},
        {"chunk_id": "Damien_Chazelle_7", "source": "Damien Chazelle", "text": "Filler. He later won an award for another film. More."},
    ],
    "verdict_evidence": {"chunk_id": "Damien_Chazelle_0", "citation": 2,
                         "matched_text": "His success came with La La Land, which was nominated for 14 Academy Awards, winning six including Best Director."},
    "career_wins": [{"work": None, "chunk_id": "Damien_Chazelle_7", "citation": 3, "matched_text": "He later won an award for another film."}],
}


def roles(details):
    return {d["citation"]: [s["role"] for s in d["supports"]] for d in details}


def test_each_citation_says_what_it_supports_and_keeps_the_full_passage():
    details = build_citation_details(RESULT, "Who directed La La Land and did they win any Academy Awards?")
    assert roles(details) == {1: ["director"], 2: ["verdict"], 3: ["career"]}
    director = details[0]["supports"][0]
    assert director["sentence"] == "La La Land is a 2016 film written and directed by Damien Chazelle."
    assert director["after"] == "It stars Ryan Gosling."
    assert "winning six including Best Director" in details[1]["supports"][0]["sentence"]
    assert details[0]["full_text"] == RESULT["retrieved_texts"][0]["text"] and details[0]["source"] == "La La Land"


def test_a_citation_without_an_identified_sentence_is_marked_cited():
    result = {**RESULT, "answer": "Free text from the LLM.", "citations": [1], "verdict_evidence": None, "career_wins": []}
    (d,) = build_citation_details(result, "What is the capital of France?")
    assert [s["role"] for s in d["supports"]] == ["cited"] and d["supports"][0]["sentence"] == ""


def test_citations_outside_the_retrieved_list_are_ignored():
    assert build_citation_details({**RESULT, "citations": [0, 9, "x"]}, "Who directed La La Land?") == []


@pytest.mark.needs_model
def test_la_la_land_end_to_end_cites_a_complete_verdict_sentence():
    from arag import agent_loop

    res = agent_loop.run_agent("Who directed La La Land and did they win any Academy Awards?", allow_fallback=False, enable_director_flow=True)
    verdict = [s for d in res["citation_details"] for s in d["supports"] if s["role"] == "verdict"]
    assert verdict and "winning six including Best Director" in verdict[0]["sentence"]
    assert not verdict[0]["sentence"].startswith("…")  # a complete sentence, not a chunk-edge fragment
    assert all(d["full_text"] for d in res["citation_details"])

"""detect_award_evidence: 'winning N including <category>' counts as a win;
navigation / link text never counts as evidence."""

import pytest

from arag.agent_loop import detect_award_evidence, find_win_including_sentence, strip_navigation_text

# Exact sentences from the debug run (chunks Damien_Chazelle_0, La_La_Land_1, La_La_Land_37).
CHAZELLE_WIN = (
    "His biggest commercial success came with La La Land, which was nominated for 14 Academy Awards, "
    "winning six including Best Director, making him the youngest person to win the award at age 32."
)
LA_LA_LAND_1 = (
    "The film also received a record-tying fourteen nominations at the 89th Academy Awards, "
    "winning in six categories including Best Director and Best Actress (Stone)."
)
LINK_TITLE_EN_DASH = "List of oldest and youngest Academy Award winners and nominees – Youngest winners for Best Director"
LINK_TITLE_HYPHEN = LINK_TITLE_EN_DASH.replace("–", "-")
# La_La_Land_37 ends like this: prose, then See also / Notes / References / External links.
LA_LA_LAND_37 = (
    "in 2017 by comedians Jimmy Fowlie and Jordan Black. The show was a full-length parody of the movie. "
    f"== See also == {LINK_TITLE_EN_DASH} Second weekend in box office performance § Second-weekend increase "
    "== Notes == == References == == External links == Official website Official sc"
)


def entry(chunk_id, text, source=None):
    return {"chunk_id": chunk_id, "source": source or chunk_id.rsplit("_", 1)[0].replace("_", " "), "text": text}


def types(evidence):
    return [e["type"] for e in evidence]


def test_winning_n_including_best_director_is_a_win():
    ev = detect_award_evidence([entry("Damien_Chazelle_0", CHAZELLE_WIN)], name_hint="Damien Chazelle")
    assert types(ev) == ["win"]
    assert ev[0]["chunk_id"] == "Damien_Chazelle_0"
    assert "winning six including Best Director" in ev[0]["matched_text"]


def test_winning_in_n_categories_including_is_a_win():
    # The film page never names the director, so (as before) it is only attributed when a
    # person page is among the retrieved passages.
    entries = [entry("Damien_Chazelle_1", "Damien Chazelle is a filmmaker."), entry("La_La_Land_1", LA_LA_LAND_1)]
    ev = detect_award_evidence(entries, name_hint="Damien Chazelle")
    wins = [e for e in ev if e["type"] == "win" and e["chunk_id"] == "La_La_Land_1"]
    assert wins and "including Best Director" in wins[0]["matched_text"]


@pytest.mark.parametrize("title", [LINK_TITLE_EN_DASH, LINK_TITLE_HYPHEN])
def test_list_of_link_title_is_not_evidence(title):
    # This text made 'Academy Award winners' match the old "academy award winner" pattern.
    assert detect_award_evidence([entry("La_La_Land_37", title)], name_hint="Damien Chazelle") == []


def test_see_also_section_of_a_real_chunk_is_not_evidence():
    assert detect_award_evidence([entry("La_La_Land_37", LA_LA_LAND_37)], name_hint="Damien Chazelle") == []


def test_prose_before_a_see_also_section_still_counts():
    text = f"Chazelle won an Academy Award for Best Director. == See also == {LINK_TITLE_EN_DASH}"
    ev = detect_award_evidence([entry("Damien_Chazelle_4", text)], name_hint="Damien Chazelle")
    assert types(ev) == ["win"]
    assert "won an Academy Award" in ev[0]["matched_text"]


def test_person_attribution_rules_still_apply():
    # No person page, no name in the text, not an awards page: still not attributed.
    ev = detect_award_evidence([entry("Some_Film_3", CHAZELLE_WIN.replace("Chazelle", "X"))], name_hint="Damien Chazelle")
    assert ev == []


def test_including_without_an_academy_award_sentence_is_not_a_win():
    text = "Chazelle's film won praise from critics, including Best Director nominations from guilds."
    assert find_win_including_sentence(text) is None


def test_plain_nomination_is_still_a_nomination():
    ev = detect_award_evidence(
        [entry("Damien_Chazelle_0", "For Whiplash, he was nominated for the Academy Award for Best Adapted Screenplay.")],
        name_hint="Damien Chazelle",
    )
    assert types(ev) == ["nom"]


def test_strip_navigation_text():
    cleaned = strip_navigation_text(LA_LA_LAND_37)
    assert "List of" not in cleaned and "Official website" not in cleaned
    assert cleaned.startswith("in 2017 by comedians")
    assert strip_navigation_text("A. List of films. B.") == "A. B."  # only sentence-initial "List of"
    assert "Retrieved" not in strip_navigation_text("Fact. Retrieved March 3, 2017. Another fact.")

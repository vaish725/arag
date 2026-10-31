"""Director detection: 'directed ... by <Name>' variants, a name that stops where it ends, and the
director sentence must be about the film asked. Sentences are copied from the corpus."""
import pytest

from arag import agent_loop
from arag.agent_loop import extract_person_name_from_text, has_directed_by

ED_WOOD = "Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood."
ROMEO = (
    'William Shakespeare\'s Romeo + Juliet (shortened to Romeo + Juliet) is a 1996 American romantic drama film directed, '
    'co-produced, and co-written by Baz Luhrmann, co-produced by Gabriella Martinelli, and co-written by Craig Pearce, '
    'being an adaptation and modernization of William Shakespeare\'s tragedy "Romeo and Juliet".'
)
LAST_ACTION_HERO = "Last Action Hero is a 1993 American fantasy action comedy film directed and produced by John McTiernan."
# James_Cameron_8: not about Last Action Hero at all
POINT_BREAK = "In 1991, Cameron served as executive producer for Point Break (1991), directed by Kathryn Bigelow."
INNERSPACE = "Innerspace is a 1987 American science fiction comedy film directed by Joe Dante and produced by Michael Finnell."
SINISTER = "Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C."
ABOUT_A_BOY = (
    "About a Boy is a 2002 British-American-French comedy-drama film produced by Jane Rosenthal, Robert De Niro, Brad Epstein, "
    "Tim Bevan and Eric Fellner, co-written and directed by brothers Chris Weitz and Paul Weitz with music by Badly Drawn Boy "
    "and written by Peter Hedges."
)


@pytest.mark.parametrize(
    "sentence, title, expected",
    [
        (ED_WOOD, "Ed Wood", "Tim Burton"),  # "directed and produced by"
        (ROMEO, "Romeo + Juliet", "Baz Luhrmann"),  # "directed, co-produced, and co-written by"
        (LAST_ACTION_HERO, "Last Action Hero", "John McTiernan"),
    ],
)
def test_the_three_failing_corpus_sentences_now_yield_the_director(sentence, title, expected):
    assert extract_person_name_from_text(sentence, work_hint=title) == expected


@pytest.mark.parametrize(
    "sentence, title, expected",
    [
        (INNERSPACE, "Innerspace", "Joe Dante"),  # used to return "Joe Dante and produced by Michael Finnell"
        (SINISTER, "Sinister", "Scott Derrickson"),  # used to return "... and written by Derrickson and C"
        (ABOUT_A_BOY, "About a Boy", "Chris Weitz and Paul Weitz"),  # co-directors; "brothers" is not part of the name
    ],
)
def test_the_name_stops_where_it_ends(sentence, title, expected):
    assert extract_person_name_from_text(sentence, work_hint=title) == expected


def test_point_break_sentence_does_not_give_last_action_hero_a_director():
    """James_Cameron_8's sentence is about Point Break, not the film asked."""
    assert extract_person_name_from_text(POINT_BREAK, work_hint="Last Action Hero") is None
    chunk = "Cameron worked on several projects. " + POINT_BREAK + " He later turned to documentaries."
    assert extract_person_name_from_text(chunk, work_hint="Last Action Hero", chunk_id="James_Cameron_8") is None


def test_end_to_end_last_action_hero_is_not_bigelow():
    res = agent_loop.run_agent("Who directed Last Action Hero and did they win any Academy Awards?", allow_fallback=False, enable_director_flow=True)
    assert "Kathryn Bigelow" not in res["answer"]
    assert res["answer"].startswith("Director: John McTiernan.")


def test_a_sentence_about_the_film_is_found_in_a_longer_chunk():
    text = "Intro text about the studio. " + LAST_ACTION_HERO + " Reviews were mixed."
    assert extract_person_name_from_text(text, work_hint="Last Action Hero") == "John McTiernan"


def test_a_sentence_without_the_title_counts_only_on_the_films_own_page():
    sentence = "It was directed by Scott Derrickson and released in November."
    assert extract_person_name_from_text(sentence, work_hint="Doctor Strange") is None
    assert extract_person_name_from_text(sentence, work_hint="Doctor Strange", chunk_id="Doctor_Strange_(2016_film)_0") == "Scott Derrickson"
    assert extract_person_name_from_text(sentence, work_hint="Doctor Strange", chunk_id="Scott_Derrickson_3") is None  # another page
    assert extract_person_name_from_text(sentence, work_hint="Doctor Strange", chunk_id="Doctor_Strange_in_the_Multiverse_of_Madness_0") is None


def test_without_a_work_hint_the_first_directed_by_sentence_is_used():
    assert extract_person_name_from_text(POINT_BREAK) == "Kathryn Bigelow"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a film directed by Jane Roe", True),
        ("directed and produced by Jane Roe", True),
        ("directed, co-produced, and co-written by Jane Roe", True),
        ("directed one two three four by Jane Roe", True),  # four words in between
        ("directed one two three four five by Jane Roe", False),  # more than four
        ("directed the film. Produced by Jane Roe", False),  # may not cross a sentence
        ("written by Jane Roe", False),
    ],
)
def test_directed_by_pattern(text, expected):
    assert has_directed_by(text) is expected

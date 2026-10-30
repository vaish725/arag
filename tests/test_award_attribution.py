"""Award attribution: generic 'Oscar winners' needs the surname; wins are tied to the film asked
about and career wins are reported separately. Sentences are copied from the corpus."""

from arag.agent_loop import (
    attribute_award_evidence,
    career_wins_sentence,
    detect_award_evidence,
    extract_asked_title,
)

INCEPTION_46 = (
    "Inception was ranked 84th on Hollywood's 100 Favorite Films, a list compiled by The Hollywood "
    'Reporter in 2014, surveying "Studio chiefs, Oscar winners and TV royalty".'
)
NOLAN_25 = (
    "weeks before or after Oppenheimer's release. The film received critical acclaim. Oppenheimer grossed over "
    "$975 million worldwide. Among the film's numerous accolades, Nolan won the Academy Awards for Best Director "
    "and Best Picture. The success of Oppenheimer further highlighted Nolan's status as one of Hollywood's most "
    "bankable directors. Oppenheimer was a hit."
)
NOLAN_0 = (
    "Pictures and signed with Universal Pictures for Oppenheimer (2023), which won him Academy Awards for Best "
    "Director and Best Picture."
)
INCEPTION_1 = (
    "Considered one of the best films of the 2010s, Inception, among its numerous accolades, won four Oscars "
    "(Best Cinematography, Best Sound Editing, Best Sound Mixing, Best Visual Effects) and was nominated for "
    "four more (Best Picture, Best Original Screenplay, Best Art Direction, Best Film Editing)."
)
GODFATHER_1 = "At the 45th Academy Awards, the film won Best Picture, Best Actor (Brando), and Best Adapted Screenplay (for Puzo and Coppola)."
GODFATHER_41 = "When the nominations for the 45th Academy Awards were revealed on February 12, 1973, The Godfather was nominated for eleven awards."
TITANIC_1 = "Among other awards, the film received fourteen nominations at the 70th Academy Awards and won eleven, including Best Picture and Best Director."


def entry(chunk_id, text, source):
    return {"chunk_id": chunk_id, "source": source, "text": text}


# ---- generic "Oscar winners" ------------------------------------------------------------------
def test_generic_oscar_winners_phrase_needs_the_surname():
    ev = detect_award_evidence([entry("Inception_46", INCEPTION_46, "Inception")], name_hint="Christopher Nolan")
    assert ev == []


def test_generic_phrase_counts_when_the_surname_is_in_the_sentence():
    text = "Christopher Nolan, an Oscar winner, spoke at the premiere."
    ev = detect_award_evidence([entry("Christopher_Nolan_9", text, "Christopher Nolan")], name_hint="Christopher Nolan")
    assert [e["type"] for e in ev] == ["win"]


def test_a_later_sentence_with_the_surname_still_qualifies():
    text = "Studio chiefs, Oscar winners and TV royalty attended. Later Nolan, also an Oscar winner, spoke."
    ev = detect_award_evidence([entry("Christopher_Nolan_9", text, "Christopher Nolan")], name_hint="Christopher Nolan")
    assert [e["type"] for e in ev] == ["win"] and "Nolan" in ev[0]["matched_text"]


# ---- film vs career ----------------------------------------------------------------------------
def inception_evidence():
    return detect_award_evidence(
        [entry("Christopher_Nolan_25", NOLAN_25, "Christopher Nolan"), entry("Inception_1", INCEPTION_1, "Inception")],
        name_hint="Christopher Nolan",
    )


def test_career_win_for_another_film_is_not_a_win_for_the_film_asked():
    att = attribute_award_evidence(inception_evidence(), "Inception", "Christopher Nolan")
    assert not att["verdict"].startswith("Won")
    assert [c["work"] for c in att["career_wins"]] == ["Oppenheimer"]
    assert career_wins_sentence(att["career_wins"]) == "Career: won Academy Awards for Oppenheimer (not this film)."


def test_film_win_in_technical_categories_is_not_the_directors_win():
    att = attribute_award_evidence(inception_evidence(), "Inception", "Christopher Nolan")
    assert att["verdict"].startswith("Nominated")  # Inception_1: nominated for Best Picture, Original Screenplay...
    assert att["evidence"]["chunk_id"] == "Inception_1"


def test_career_work_comes_from_the_explicit_title_and_year():
    ev = detect_award_evidence([entry("Christopher_Nolan_0", NOLAN_0, "Christopher Nolan")], name_hint="Christopher Nolan")
    att = attribute_award_evidence(ev, "Inception", "Christopher Nolan")
    assert [c["work"] for c in att["career_wins"]] == ["Oppenheimer"]


def test_win_with_best_director_on_the_film_page_is_a_film_win():
    # a person page among the retrieved passages is what lets film-page sentences be attributed
    ev = detect_award_evidence(
        [entry("James_Cameron_0", "James Cameron is a filmmaker.", "James Cameron"), entry("Titanic_(1997_film)_1", TITANIC_1, "Titanic (1997 film)")],
        name_hint="James Cameron",
    )
    att = attribute_award_evidence(ev, "Titanic", "James Cameron")
    assert att["verdict"].startswith("Won") and att["career_wins"] == []


def test_godfather_win_names_coppola_for_adapted_screenplay():
    ev = detect_award_evidence([entry("The_Godfather_1", GODFATHER_1, "The Godfather")], name_hint="Francis Ford Coppola")
    att = attribute_award_evidence(ev, "The Godfather", "Francis Ford Coppola")
    assert att["verdict"].startswith("Won") and att["evidence"]["chunk_id"] == "The_Godfather_1"


def test_sequel_does_not_inherit_the_first_films_evidence():
    ev = detect_award_evidence(
        [entry("The_Godfather_1", GODFATHER_1, "The Godfather"), entry("The_Godfather_41", GODFATHER_41, "The Godfather")],
        name_hint="Francis Ford Coppola",
    )
    att = attribute_award_evidence(ev, "The Godfather Part II", "Francis Ford Coppola")
    assert att["verdict"].startswith("Unknown"), att
    assert [c["work"] for c in att["career_wins"]] == ["The Godfather"]
    assert "not this film" in career_wins_sentence(att["career_wins"])


def test_without_a_film_everything_counts_as_before():
    ev = detect_award_evidence([entry("The_Godfather_1", GODFATHER_1, "The Godfather")], name_hint="Francis Ford Coppola")
    assert attribute_award_evidence(ev, None, "Francis Ford Coppola")["verdict"].startswith("Won")


def test_extract_asked_title():
    assert extract_asked_title("Who directed The Godfather Part II and did they win any Academy Awards?") == "The Godfather Part II"
    assert extract_asked_title("Who directed Titanic (1997 film) and did they win any Academy Awards?") == "Titanic"
    assert extract_asked_title("What is the capital of France?") is None

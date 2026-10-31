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
    assert len(att["career_wins"]) == 1


def test_career_line_does_not_guess_a_title_the_passage_does_not_state():
    # NOLAN_25 never writes "Oppenheimer (2023)"; Oppenheimer is only the most repeated name.
    att = attribute_award_evidence(inception_evidence(), "Inception", "Christopher Nolan")
    assert att["career_wins"][0]["work"] is None
    assert career_wins_sentence(att["career_wins"]) == "Career: won Academy Awards for another film (not this film)."


def test_most_repeated_capitalised_name_is_not_taken_as_the_film():
    text = (
        "Nolan met Warner Bros. executives. Warner Bros. financed the project and Warner Bros. gave "
        "him control. Hans Zimmer and Hans Zimmer's team scored it. Among the film's numerous accolades, "
        "Nolan won the Academy Awards for Best Director and Best Picture."
    )
    ev = detect_award_evidence([entry("Christopher_Nolan_30", text, "Christopher Nolan")], name_hint="Christopher Nolan")
    att = attribute_award_evidence(ev, "Inception", "Christopher Nolan")
    assert [c["work"] for c in att["career_wins"]] == [None]
    sentence = career_wins_sentence(att["career_wins"])
    assert "Warner" not in sentence and "Zimmer" not in sentence
    assert sentence == "Career: won Academy Awards for another film (not this film)."


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


# ---- general rules for what counts as the director's win / an Oscar nomination ------------------
INNERSPACE_0 = "It earned well over $25 million in its domestic gross revenue and won an Oscar, the only film directed by Dante to do so."
SILENT_LIGHT_SPIRIT = "The film was nominated for Best Foreign Film at the 24th Independent Spirit Awards."
SILENT_LIGHT_OSCAR_ENTRY = "The film was selected as the Mexican entry for the Best Foreign Language Oscar at the 80th Academy Awards, but it did not make the shortlist."

from arag.agent_loop import _names_recipient  # noqa: E402


def test_innerspace_film_win_is_not_a_director_win():
    ev = detect_award_evidence([entry("Innerspace_0", INNERSPACE_0, "Innerspace")], name_hint="Joe Dante")
    att = attribute_award_evidence(ev, "Innerspace", "Joe Dante")
    assert att["verdict"].startswith("Unknown"), att
    assert att["evidence"] is None and att["career_wins"] == []


def test_a_win_sentence_that_names_neither_director_nor_best_director_is_the_films_win():
    # previously "no categories listed -> count it"
    ev = detect_award_evidence(
        [entry("Some_Film_0", "The film won an Academy Award and made money.", "Some Film"), entry("Jane_Roe_0", "Jane Roe is a director.", "Jane Roe")],
        name_hint="Jane Roe",
    )
    assert attribute_award_evidence(ev, "Some Film", "Jane Roe")["verdict"].startswith("Unknown")


def test_director_named_as_a_recipient_is_credited():
    ev = detect_award_evidence([entry("Joe_Dante_0", "Dante won an Academy Award for Best Visual Effects.", "Joe Dante")], name_hint="Joe Dante")
    assert attribute_award_evidence(ev, None, "Joe Dante")["verdict"].startswith("Won")


def test_names_recipient_ignores_directed_by_but_accepts_recipients():
    assert not _names_recipient(INNERSPACE_0, "dante")
    assert not _names_recipient("Won an Oscar; written and directed by Joe Dante.", "dante")
    assert _names_recipient("Puzo and Coppola won for Best Adapted Screenplay.", "coppola")
    assert _names_recipient("Coppola, who directed it, won an Oscar.", "coppola")
    assert _names_recipient("Dante won an Academy Award.", "dante")
    assert not _names_recipient("Anything at all.", None)


def test_independent_spirit_nomination_is_not_an_oscar_nomination():
    ev = detect_award_evidence([entry("Silent_Light_0", SILENT_LIGHT_SPIRIT, "Silent Light")], name_hint="Carlos Reygadas")
    assert ev == []
    # the other Silent Light sentence is an Oscar *entry* that "did not make the shortlist": not a nomination either
    assert detect_award_evidence([entry("Silent_Light_0", SILENT_LIGHT_OSCAR_ENTRY, "Silent Light")], name_hint="Carlos Reygadas") == []


def test_a_nomination_with_academy_award_in_the_same_sentence_still_counts():
    ev = detect_award_evidence(  # a person page among the passages is what attributes film-page sentences
        [entry("Carlos_Reygadas_0", "Carlos Reygadas is a filmmaker.", "Carlos Reygadas"),
         entry("Silent_Light_0", "The film was nominated for an Academy Award for Best Foreign Language Film.", "Silent Light")],
        name_hint="Carlos Reygadas",
    )
    assert [(e["chunk_id"], e["type"]) for e in ev] == [("Silent_Light_0", "nom")]


def test_an_oscar_word_in_a_different_sentence_does_not_make_a_nomination_an_oscar_one():
    text = "The film was nominated for Best Foreign Film at the Independent Spirit Awards. Separately, it later screened. Oscar fans cheered."
    assert detect_award_evidence([entry("Silent_Light_0", text, "Silent Light")], name_hint="Carlos Reygadas") == []

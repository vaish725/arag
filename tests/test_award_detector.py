import pytest

from arag.agent_loop import detect_award_evidence


def make_entry(chunk_id: str, source: str, text: str):
    return {"chunk_id": chunk_id, "source": source, "text": text}


def test_detect_win_with_name_in_id_or_text():
    # If the chunk_id or text contains the director's last name and
    # mentions a win, it should be found
    entries = [
        make_entry(
            "Scott_Derrickson_0",
            "Scott Derrickson",
            "Scott Derrickson won an Academy Award for Best Visual Effects.",
        ),
    ]
    ev = detect_award_evidence(entries, name_hint="Scott Derrickson")
    assert len(ev) == 1
    assert ev[0]["type"] == "win"
    assert ev[0]["entry_index"] == 0


def test_reject_spurious_without_name_presence():
    # An unrelated award paragraph (Heidi Ewing) should not be matched
    # for Scott Derrickson
    entries = [
        make_entry(
            "Heidi_Ewing_0",
            "Heidi Ewing",
            "Heidi Ewing was nominated for an Academy Award for Best Documentary.",
        ),
    ]
    ev = detect_award_evidence(entries, name_hint="Scott Derrickson")
    assert ev == []


def test_accept_awards_page_without_name_hint():
    # If the chunk id indicates awards list, accept even if name_hint is not provided
    entries = [
        make_entry(
            "List_of_awards_and_nominations_received_by_Scott_Derrickson_0",
            "List of awards and nominations received by Scott Derrickson",
            "Scott Derrickson won an Academy Award.",
        ),
    ]
    ev = detect_award_evidence(entries, name_hint=None)
    assert len(ev) == 1
    assert ev[0]["type"] == "win"
    assert ev[0]["entry_index"] == 0

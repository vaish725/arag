import pytest

from arag import agent_loop


@pytest.mark.parametrize(
    "query, expected_person, expected_award",
    [
        ("Who directed Inception and did they win any Academy Awards?", "Christopher Nolan", "win"),
        ("Who directed Doctor Strange and did they win any Academy Awards?", "Scott Derrickson", "unknown"),
        ("Who directed The Godfather and did they win any Academy Awards?", "Francis Ford Coppola", "nom"),
        ("Who directed Titanic and did they win any Academy Awards?", "James Cameron", "win"),
        ("Who directed La La Land and did they win any Academy Awards?", "Damien Chazelle", "win"),
    ],
)
def test_award_regression(query: str, expected_person: str, expected_award: str):
    """Regression test: run the evidence-only director->award flow and
    assert the director name appears and that the award verdict category
    (win/nom/unknown) is present in the answer string.
    """
    # Use the explicit opt-in director flow to avoid accidental routing
    res = agent_loop.run_agent(query, allow_fallback=False, enable_director_flow=True)
    answer = (res.get("answer") or "").lower()

    assert expected_person.lower() in answer, f"Expected person '{expected_person}' in answer: {answer}"

    if expected_award == "win":
        assert ("won" in answer) or ("win" in answer), f"Expected a win verdict in answer: {answer}"
    elif expected_award == "nom":
        assert "nom" in answer or "nomin" in answer, f"Expected a nomination verdict in answer: {answer}"
    elif expected_award == "unknown":
        assert "unknown" in answer or "no win" in answer or "no win evidence" in answer, f"Expected unknown verdict in answer: {answer}"
    else:
        pytest.skip("Unknown expected_award label")


def test_director_flow_not_triggered_for_generic_queries():
    """Ensure the director flow doesn't activate for unrelated queries when
    enable_director_flow is False (programmatic default).
    """
    q = "What is the capital of France?"
    res = agent_loop.run_agent(q, allow_fallback=False)
    # programmatic default should not route to director flow and should
    # return an empty answer when no fallback is allowed
    assert res.get("answer") == "", "Expected empty answer for non-director query when director flow disabled"


@pytest.mark.parametrize(
    "query, expected_person, expected_award",
    [
        # Edge cases: films with parentheses and multiple tokens
        ("Who directed Titanic (1997 film) and did they win any Academy Awards?", "James Cameron", "win"),
        ("Who directed The Godfather Part II and did they win any Academy Awards?", "Francis Ford Coppola", "nom"),
    ],
)
def test_award_regression_edges(query: str, expected_person: str, expected_award: str):
    res = agent_loop.run_agent(query, allow_fallback=False, enable_director_flow=True)
    answer = (res.get("answer") or "").lower()
    assert expected_person.lower() in answer
    if expected_award == "win":
        assert ("won" in answer) or ("win" in answer)
    elif expected_award == "nom":
        assert "nom" in answer or "nomin" in answer

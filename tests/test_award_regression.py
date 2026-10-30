"""Director -> award regression. Expectations are READ from scripts/regression_baseline.json
(`expected_system`), the single source of truth; nothing is repeated here."""
import json
from pathlib import Path

import pytest
from verdict_util import parse_verdict

from arag import agent_loop

pytestmark = pytest.mark.needs_model

BASELINE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "regression_baseline.json"
BASELINE = json.loads(BASELINE_PATH.read_text())
FILMS = sorted(BASELINE)


def run(film):
    return agent_loop.run_agent(
        f"Who directed {film} and did they win any Academy Awards?",
        allow_fallback=False,
        enable_director_flow=True,
    )


@pytest.mark.parametrize("film", FILMS)
def test_film_matches_expected_system_behaviour(film):
    expected = BASELINE[film]["expected_system"]
    res = run(film)
    answer = res["answer"]
    assert expected["director"].lower() in answer.lower(), answer
    assert parse_verdict(answer) == expected["verdict"], f"{film}: {answer}"
    if expected["required_verdict_chunk"]:
        ev = res["verdict_evidence"]
        assert ev and ev["chunk_id"] == expected["required_verdict_chunk"], ev
    # A career line may only appear when expected, and then only as an explicit/"another film" line.
    assert ("Career: won Academy Awards for " in answer) == expected["career_line"], answer
    if expected["career_line"]:
        assert "(not this film)." in answer
        for c in res["career_wins"]:
            if c["work"]:
                assert f"{c['work']} (" in c["matched_text"], c  # named only when stated explicitly


def test_every_baseline_film_has_sourced_truth_and_expectations():
    for film, rec in BASELINE.items():
        truth = rec["expected_truth"]
        assert truth["verdict"] in {"won", "nominated", "no_win"}, film
        assert truth["sources"], f"{film}: no source for the truth label"
        for s in truth["sources"]:
            assert s["url"].startswith("https://en.wikipedia.org/wiki/") and 0 < len(s["quote"].split()) <= 25, (film, s)
        assert set(rec["expected_system"]) >= {"director", "verdict", "required_verdict_chunk", "career_line", "note"}, film


def assert_same_films(cases, in_json, label):
    missing, extra = set(in_json) - set(cases), set(cases) - set(in_json)
    assert not missing and not extra, f"{label} and regression_baseline.json cover different films: missing={sorted(missing)} extra={sorted(extra)}"


def test_film_cases_cover_exactly_the_films_in_the_json():
    """Fails if the test cases (or the report script) and the JSON ever cover different films."""
    in_json = json.loads(BASELINE_PATH.read_text())
    param_films = next(m.args[1] for m in test_film_matches_expected_system_behaviour.pytestmark if m.name == "parametrize")
    assert_same_films(param_films, in_json, "the parametrized test")
    import sys

    sys.path.insert(0, str(BASELINE_PATH.parent))
    import regression_report

    assert_same_films(regression_report.FILMS, in_json, "scripts/regression_report.py")


def test_the_coverage_check_really_fails_on_a_mismatch():
    in_json = json.loads(BASELINE_PATH.read_text())
    with pytest.raises(AssertionError, match="different films"):
        assert_same_films(list(in_json)[1:], in_json, "a test with one film missing")
    with pytest.raises(AssertionError, match="different films"):
        assert_same_films(list(in_json) + ["Some New Film"], in_json, "a test with an extra film")


def test_director_flow_not_triggered_for_generic_queries():
    """The director flow must not activate for unrelated queries when enable_director_flow is False."""
    res = agent_loop.run_agent("What is the capital of France?", allow_fallback=False)
    assert res.get("answer") == "", "Expected empty answer for non-director query when director flow disabled"

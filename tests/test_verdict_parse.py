import json
from pathlib import Path

import pytest
from verdict_util import parse_verdict

BASELINE = Path(__file__).resolve().parents[1] / "scripts" / "regression_baseline.json"


@pytest.mark.parametrize(
    "answer, verdict",
    [
        ("Director: X. Won at least one Academy Award (supported by contexts). Citations: [1].", "won"),
        ("Director: X. Nominated for Academy Awards (no win evidence in provided contexts). Citations: [1, 4].", "nominated"),
        ("Director: X. Unknown based on provided contexts.", "unknown"),
        ("something unrelated", None),
    ],
)
def test_parse_verdict_is_exact(answer, verdict):
    assert parse_verdict(answer) == verdict


def test_no_win_evidence_is_not_a_win():
    # the old substring check ("win" in answer) accepted this as a win
    assert parse_verdict("Nominated for Academy Awards (no win evidence in provided contexts).") != "won"


def test_baseline_answers_parse_to_their_recorded_verdicts():
    for film, rec in json.loads(BASELINE.read_text()).items():
        assert parse_verdict(rec["answer"]) == rec["verdict"], film

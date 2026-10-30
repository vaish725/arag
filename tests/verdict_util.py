"""Parse the award verdict category out of an evidence-only answer, exactly."""
import re
from typing import Optional

EXPECTED = {"win": "won", "nom": "nominated", "unknown": "unknown"}


def parse_verdict(answer: str) -> Optional[str]:
    """Return "won", "nominated" or "unknown" (None if no verdict sentence is found).

    Matches the verdict sentences the template produces, not substrings such as the
    "win" inside "no win evidence".
    """
    a = (answer or "").lower()
    if re.search(r"\bwon at least one academy award\b", a):
        return "won"
    if re.search(r"\bnominated for academy awards\b", a):
        return "nominated"
    if re.search(r"\bunknown based on provided contexts\b", a):
        return "unknown"
    return None

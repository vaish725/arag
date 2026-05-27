import pytest

from arag.agent_loop import normalize_title


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("Titanic (1997 film)", "Titanic"),
        ("  The Godfather Part II  ", "The Godfather"),
        ("La La Land", "La La Land"),
        ("Some Movie (director's cut)", "Some Movie"),
        ("A Film Part 2", "A Film"),
    ],
)
def test_normalize_title(inp, expected):
    assert normalize_title(inp) == expected

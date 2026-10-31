#!/usr/bin/env python3
"""Add a public source to every row of eval/data/award_set.json (verified_by: "agent-sourced").

Per row: source_url, supporting_text (verbatim, at most 25 words), verified_by, support_type, and for
no_win rows an absence_check (the director's full name does not appear on the Wikipedia pages of the
film's Academy Awards ceremony). Sources are Wikipedia film articles and ceremony pages.

supporting_text is, in order of preference:
  * a hand-picked sentence or table row about the film's Oscar outcome (QUOTES below), or
  * for films with no Oscar story at all, the first 25 words of that ceremony's Best Directing row
    (shows who was nominated for directing); the claim "the director has no nomination" then rests on
    the absence check, a negative that no single quote can show.
Run once; eval/verify_award_labels.py re-checks everything against the live pages.

Usage: python eval/build_label_sources.py
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_award_labels as v  # noqa: E402  (fetch helpers)

W = "https://en.wikipedia.org/wiki/"
BASELINE = json.loads((ROOT / "scripts/regression_baseline.json").read_text())

# Films that carry their own Oscar sentence. The five regression films reuse their sourced truth.
QUOTES = {
    "About a Boy (film)": ("About_a_Boy_(film)", "Academy Awards Best Adapted Screenplay Chris Weitz , Paul Weitz and Peter Hedges Nominated"),
    "Ed Wood (film)": ("Ed_Wood_(film)", "Landau and Rick Baker won Academy Awards for their work on the film."),
    "True Lies": ("True_Lies", "It was also nominated at the Academy Awards and BAFTAs in the Best Visual Effects category"),
    "Total Recall (1990 film)": ("Total_Recall_(1990_film)", "The practical special effects were well received, earning the film an Academy Award"),
    "Innerspace": ("Innerspace", "won an Academy Award for Best Visual Effects , the only film directed by Dante to do so."),
    "Romeo + Juliet": ("Romeo_%2B_Juliet", "The film received a single nomination for Best Art Direction at the 69th Academy Awards ."),
}
# Names differ for one film between the two data files.
BASELINE_KEY = {"Doctor Strange (2016 film)": "Doctor Strange"}


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def release_year(film: str, chunks: dict) -> int:
    return int(re.search(r"\b(1[89]\d\d|20[0-2]\d)\b", chunks[film][:160]).group(1))


def main() -> None:
    chunks_all = pickle.loads((ROOT / "data/chunks.pkl").read_bytes())
    first = {}
    for c in chunks_all:
        first.setdefault(c["source"], c["text"])
    path = ROOT / "eval/data/award_set.json"
    data = json.loads(path.read_text())
    cache: dict = {}

    def ceremony(n: int):
        title = f"{ordinal(n)} Academy Awards"
        if title not in cache:
            soup, rev = v.fetch(title)
            cache[title] = (soup, rev, re.sub(r"\s+", " ", soup.get_text(" ", strip=True)))
        return title, cache[title]

    for row in data["films"]:
        film, director = row["film"], row["director"]
        year = release_year(film, first)
        base_key = BASELINE_KEY.get(film, film)
        if base_key in BASELINE:  # reuse the regression film's sourced truth
            src = BASELINE[base_key]["expected_truth"]["sources"][0]
            row.update(source_url=src["url"], supporting_text=src["quote"], support_type="quote")
        elif film in QUOTES:
            page, quote = QUOTES[film]
            row.update(source_url=W + page, supporting_text=quote, support_type="quote")
        else:
            title, (soup, rev, _text) = ceremony(year + 1 - 1928)
            directing = None
            for td in soup.find_all("td"):
                if td.find("a", string=re.compile(r"Directing|Best Director")):
                    directing = re.sub(r"\s+", " ", td.get_text(" ", strip=True))
                    break
            row.update(
                source_url=W + urllib.parse.quote(title.replace(" ", "_")),
                supporting_text=" ".join(directing.split()[:25]),
                support_type="ceremony Best Directing row (negative claim: see absence_check)",
            )
        if row["truth"] == "no_win":
            numbers = [year + 1 - 1928, year + 2 - 1928]  # the ceremony after release, and the one after that
            names = [n.strip() for n in re.split(r"\s+and\s+", director)]
            row["absence_check"] = {
                "ceremony_pages": [W + ordinal(n) + "_Academy_Awards" for n in numbers],
                "director_names": names,
                "name_absent_from_all_pages": all(name.lower() not in ceremony(n)[1][2].lower() for n in numbers for name in names),
            }
        row["verified_by"] = "agent-sourced"
        row.pop("basis_source", None)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    bad = [r["film"] for r in data["films"] if r.get("absence_check", {}).get("name_absent_from_all_pages") is False]
    print(f"sourced {len(data['films'])} rows; absence check FAILED for: {bad or 'none'}")


if __name__ == "__main__":
    main()

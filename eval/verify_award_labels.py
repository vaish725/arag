#!/usr/bin/env python3
"""Corroborate eval/data/award_set.json labels against each film's Wikipedia article.

For every film this fetches the article (pinning its revision id) and collects every paragraph, list
item or table row that mentions an Academy Award / Oscar. A label is "consistent" when:
  * no_win    : none of those passages names the director (so the article shows no Oscar nomination
                or win for the director in this film);
  * won / nominated : a passage names the director together with a win / nomination, or the film is
                one of the regression films whose truth is already sourced in
                scripts/regression_baseline.json (quotes verified by scripts/verify_regression_truth.py).
Anything else is "needs review". This is corroboration, not proof: an article that does not mention a
nomination cannot show there was none (the Academy database was not queried).
Writes eval/data/award_label_evidence.json. Needs network (Wikipedia API only).

It ALSO re-checks the sourcing stored on every row (eval/build_label_sources.py): `supporting_text` must
be at most 25 words and appear on `source_url` (whitespace-insensitive), and for no_win rows each
`absence_check` must still hold (the director's full name appears on none of the ceremony pages).

Usage: python eval/verify_award_labels.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"User-Agent": "arag-eval/1.0 (+https://github.com/vaish725/arag)"}
OSCAR = re.compile(r"Academy Award|Oscar", re.IGNORECASE)


# Passages that name the director but do not show a director nomination or win, read by hand
# (the automatic rule cannot tell). Reason = what the article actually says.
MANUAL_REVIEW = {
    "True Lies": "Cameron won the Saturn Award for Best Director; the film was nominated at the Academy Awards for Best Visual Effects only.",
    "Innerspace": "The film won the Academy Award for Best Visual Effects, 'the only film directed by Dante to do so': the film's win, not the director's.",
    "Romeo + Juliet": "At the 69th Academy Awards Catherine Martin and Brigitte Broch were nominated for Best Art Direction; Luhrmann's honours are the Berlin Alfred Bauer Prize and BAFTA.",
    "The Family Man": "Oscar mentions are of an Academy Award-winning effects supervisor and a nominated cinematographer, not the director.",
}


def fetch(title: str):
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "parse", "page": title, "prop": "text|revid", "format": "json", "redirects": 1},
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()["parse"]
    soup = BeautifulSoup(data["text"]["*"], "html.parser")
    for tag in soup.select("sup.reference, ol.references, div.reflist, style, .mw-editsection, .navbox"):
        tag.decompose()
    return soup, data["revid"]


def trim(text: str, words: int = 40) -> str:
    parts = re.sub(r"\s+", " ", text).strip().split()
    return " ".join(parts[:words]) + (" ..." if len(parts) > words else "")


_text_cache: dict = {}


def page_text(url: str) -> str:
    title = urllib.parse.unquote(url.rsplit("/wiki/", 1)[1])
    if title not in _text_cache:
        _text_cache[title] = fetch(title)[0].get_text(" ")
    return _text_cache[title]


def squash(text: str) -> str:
    return "".join(text.split()).lower()


NICKNAMES = {"chris": ["christopher"], "joe": ["joseph"], "tim": ["timothy"], "brad": ["bradley"], "baz": ["bazmark"],
             "iain": ["ian"], "mike": ["michael"], "michael": ["mike"], "john": ["jon"], "scott": ["scot"], "paul": ["pablo"]}


def name_variants(name: str) -> list:
    """The full name, first+last (middle name dropped), common first-name variants, and the surname alone."""
    parts = name.split()
    first, last = parts[0], parts[-1]
    out = [name]
    if len(parts) > 2:
        out.append(f"{first} {last}")
    out += [f"{n.title()} {last}" for n in NICKNAMES.get(first.lower(), [])]
    out.append(last)
    return list(dict.fromkeys(out))


def find_variant_hits(names: list, pages: list) -> list:
    """Every (variant, page) where a name variant appears as a whole word on a ceremony page."""
    hits = []
    for url in pages:
        text = re.sub(r"\s+", " ", page_text(url))
        for name in names:
            for variant in name_variants(name):
                m = re.search(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", text, re.IGNORECASE)
                if m:
                    hits.append({"variant": variant, "page": url.rsplit("/", 1)[1],
                                 "context": text[max(0, m.start() - 45): m.end() + 45]})
    return hits


def check_source(row: dict) -> list:
    """Problems with the stored source of one row (empty list = fine)."""
    problems = []
    if row.get("verified_by") != "agent-sourced":
        problems.append("verified_by is not 'agent-sourced'")
    quote = row.get("supporting_text") or ""
    if not quote or len(quote.split()) > 25:
        problems.append(f"supporting_text must be 1-25 words (has {len(quote.split())})")
    elif squash(quote) not in squash(page_text(row["source_url"])):
        problems.append("supporting_text NOT found on source_url")
    check = row.get("absence_check")
    if check:
        for url in check["ceremony_pages"]:
            text = squash(page_text(url))
            for name in check["director_names"]:
                if squash(name) in text:
                    problems.append(f"{name!r} appears on {url.rsplit('/', 1)[1]}")
        live = find_variant_hits(check["director_names"], check["ceremony_pages"])
        stored = {(h["variant"], h["page"]) for h in check.get("variant_hits", [])}
        if {(h["variant"], h["page"]) for h in live} != stored:
            problems.append(f"variant hits changed: stored {sorted(stored)} vs live {sorted((h['variant'], h['page']) for h in live)}")
        if live and not check.get("review"):
            problems.append("variant hits without a written review")
    elif row["truth"] == "no_win":
        problems.append("no_win row without an absence_check")
    return problems


def main() -> int:
    films = json.loads((ROOT / "eval/data/award_set.json").read_text())["films"]
    baseline = json.loads((ROOT / "scripts/regression_baseline.json").read_text())
    sourced = {f for f, rec in baseline.items() if rec["expected_truth"].get("sources")}
    out, review = [], 0
    for f in films:
        title = f["film"]
        soup, revid = fetch(title)
        units = []
        for el in soup.find_all(["p", "li", "tr"]):
            text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if OSCAR.search(text) and text not in units and len(text.split()) < 120:
                units.append(text)
        source_problems = check_source(f)
        surnames = [p.split()[-1].lower() for p in re.split(r"\s+and\s+", f["director"])]
        named = [u for u in units if any(s in u.lower() for s in surnames)]
        if f["film"] in sourced:
            status = "consistent (truth sourced in scripts/regression_baseline.json)"
        elif f["truth"] == "no_win":
            status = "consistent" if not named else "needs review"
        else:
            status = "consistent" if any(re.search(r"nominat|won|win", u, re.I) for u in named) else "needs review"
        manual = MANUAL_REVIEW.get(title) if status == "needs review" else None
        if manual:
            status = "consistent (manual reading of the passages)"
        absence = f.get("absence_check") or {}
        if absence.get("variant_hits") and status.startswith("consistent"):
            status += "; absence check: variant hit(s) reviewed as other people"
        if source_problems:
            status = "needs review"
        review += status == "needs review"
        out.append({
            "film": title,
            "article_url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="()_,'"),
            "revision_id": revid,
            "label": f["truth"],
            "director": f["director"],
            "academy_award_passages": len(units),
            "passages_naming_the_director": len(named),
            "examples": [trim(u) for u in (named or units)[:3]],
            "source_url": f["source_url"],
            "supporting_text": f["supporting_text"],
            "source_problems": source_problems,
            "status": status,
            **({"manual_review": manual} if manual else {}),
        })
        print(f"{status[:10]:10} {title:34} label={f['truth']:9} passages={len(units):2} naming_director={len(named)}")
    (ROOT / "eval/data/award_label_evidence.json").write_text(json.dumps(
        {"checked_on": "2026-10-30", "note": "Corroboration from each film's Wikipedia article (pinned revision). Absence of a mention is not proof of absence of a nomination; the Academy database was not queried. Labels remain owner_verified=false.", "films": out}, indent=1, ensure_ascii=False) + "\n")
    print(f"\n{len(out) - review} consistent, {review} need review")
    return 0


if __name__ == "__main__":
    sys.exit(main())

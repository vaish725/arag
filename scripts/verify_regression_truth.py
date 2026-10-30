#!/usr/bin/env python3
"""Check scripts/regression_baseline.json `expected_truth.sources` against the live Wikipedia pages.

For each film, every stored quote (<= 25 words) must appear in the page text (whitespace-insensitive),
and any `must_not_contain` string must be absent from that page. Needs network (Wikipedia API only).

Usage: python scripts/verify_regression_truth.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASELINE = Path(__file__).with_name("regression_baseline.json")
HEADERS = {"User-Agent": "arag-eval/1.0 (+https://github.com/vaish725/arag)"}
_cache: dict = {}


def squash(text: str) -> str:
    return "".join(text.split()).lower()


def page_text(url: str) -> str:
    title = urllib.parse.unquote(url.rsplit("/wiki/", 1)[1])
    if title not in _cache:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "parse", "page": title, "prop": "text", "format": "json", "redirects": 1},
            headers=HEADERS,
            timeout=30,
        )
        r.raise_for_status()
        _cache[title] = BeautifulSoup(r.json()["parse"]["text"]["*"], "html.parser").get_text(" ")
    return _cache[title]


def main() -> int:
    films = json.loads(BASELINE.read_text())
    failures = 0
    for film, rec in films.items():
        truth = rec["expected_truth"]
        for src in truth["sources"]:
            text, quote = page_text(src["url"]), src["quote"]
            problems = []
            if len(quote.split()) > 25:
                problems.append(f"quote has {len(quote.split())} words (max 25)")
            if squash(quote) not in squash(text):
                problems.append("quote NOT found on the page")
            for bad in src.get("must_not_contain", []):
                if squash(bad) in squash(text):
                    problems.append(f"page unexpectedly contains {bad!r}")
            failures += bool(problems)
            print(f"{'FAIL' if problems else 'ok  '} {film:24} truth={truth['verdict']:9} {src['url'].rsplit('/wiki/', 1)[1]}: {quote[:70]!r} {'; '.join(problems)}")
    print(f"\n{'ALL SOURCES VERIFIED' if not failures else str(failures) + ' FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

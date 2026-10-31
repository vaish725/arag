#!/usr/bin/env python3
"""Pick 5 labelled films at random (seeded) from both label sets, for the owner to spot-check.

Pool: the 6 distinct regression films (scripts/regression_baseline.json) plus the 26 award-set films
(eval/data/award_set.json); a film in both appears once. Seed 20261031.

Usage: python eval/pick_spot_check.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261031


def main() -> None:
    pool = {}
    for film, rec in json.loads((ROOT / "scripts/regression_baseline.json").read_text()).items():
        if rec.get("same_film_as"):
            continue  # plain "Titanic" is the same film as "Titanic (1997 film)"
        t = rec["expected_truth"]
        pool[film] = {"set": "regression", "label": t["verdict"], "source_url": t["source_url"], "supporting_text": t["supporting_text"]}
    for row in json.loads((ROOT / "eval/data/award_set.json").read_text())["films"]:
        key = {"Doctor Strange (2016 film)": "Doctor Strange"}.get(row["film"], row["film"])
        entry = {"set": "award set", "label": row["truth"], "source_url": row["source_url"], "supporting_text": row["supporting_text"]}
        if key in pool:
            pool[key]["set"] = "regression + award set"
        else:
            pool[key] = entry
    picked = random.Random(SEED).sample(sorted(pool), 5)
    for film in picked:
        r = pool[film]
        print(f"- {film} [{r['set']}] label={r['label']}\n    source: {r['source_url']}\n    text:   {r['supporting_text']}")


if __name__ == "__main__":
    main()

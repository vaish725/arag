#!/usr/bin/env python3
"""Write eval/data/splits.json: a seeded dev/test split for both tracks.

dev  = may be used for tuning.   test = held out, never used to tune or choose settings.
Retrieval: the question ids are sorted, shuffled with the seed, first 40% -> dev.
Awards: stratified by truth class (so every class appears in test), same seed and 40% dev.
The output is committed; do not regenerate it after looking at results.

Usage: python eval/build_splits.py
"""
import json
import math
import random

SEED = 20261030
DEV_FRACTION = 0.4


def split(items, rng):
    items = sorted(items)
    rng.shuffle(items)
    n_dev = min(len(items) - 1, max(1, round(DEV_FRACTION * len(items)))) if len(items) > 1 else 0
    return sorted(items[:n_dev]), sorted(items[n_dev:])


def main():
    rng = random.Random(SEED)
    hot = json.load(open("eval/data/hotpot_subset.json"))
    ids = [q["id"] for q in hot["questions"]]
    r_dev, r_test = split(ids, rng)

    awards = json.load(open("eval/data/award_set.json"))["films"]
    a_dev, a_test = [], []
    for truth in ("won", "nominated", "no_win"):
        d, t = split([f["film"] for f in awards if f["truth"] == truth], rng)
        a_dev += d
        a_test += t
    out = {
        "seed": SEED,
        "dev_fraction": DEV_FRACTION,
        "note": "dev may be used for tuning; test is held out and must never be used to tune.",
        "retrieval": {"dev": r_dev, "test": r_test},
        "awards": {"dev": sorted(a_dev), "test": sorted(a_test)},
    }
    json.dump(out, open("eval/data/splits.json", "w"), indent=1)
    open("eval/data/splits.json", "a").write("\n")
    print(f"retrieval dev/test = {len(r_dev)}/{len(r_test)}; awards dev/test = {len(a_dev)}/{len(a_test)}")


if __name__ == "__main__":
    main()

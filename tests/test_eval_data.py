"""The committed evaluation data stays consistent: disjoint dev/test, real gold chunks, valid labels."""
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "eval" / "data" / name).read_text())


def test_splits_are_disjoint_and_cover_every_item():
    splits, subset, awards = load("splits.json"), load("hotpot_subset.json"), load("award_set.json")
    for track, universe in (
        ("retrieval", {q["id"] for q in subset["questions"]}),
        ("awards", {f["film"] for f in awards["films"]}),
    ):
        dev, test = set(splits[track]["dev"]), set(splits[track]["test"])
        assert dev and test and not dev & test
        assert dev | test == universe
    assert splits["seed"] == 20261030


def test_gold_chunk_ids_exist_in_the_corpus():
    ids = {c["id"] for c in pickle.loads((ROOT / "data" / "chunks.pkl").read_bytes())}
    for q in load("hotpot_subset.json")["questions"]:
        assert q["gold_chunk_ids"] and set(q["gold_chunk_ids"]) <= ids


def test_award_labels_are_valid_and_flagged_for_owner_review():
    films = load("award_set.json")["films"]
    assert len({f["film"] for f in films}) == len(films)
    assert {f["truth"] for f in films} <= {"won", "nominated", "no_win"}
    assert all(f["owner_verified"] is False for f in films)  # labels not yet checked by the owner
    sources = {c["source"] for c in pickle.loads((ROOT / "data" / "chunks.pkl").read_bytes())}
    assert all(f["film"] in sources for f in films), "every film must have its page in the corpus"


def test_summary_matches_the_per_row_results():
    rows = json.loads((ROOT / "eval" / "results" / "before_fixes" / "retrieval_results.json").read_text())["rows"]
    summary = json.loads((ROOT / "eval" / "results" / "before_fixes" / "summary.json").read_text())["retrieval"]["test"]
    test_rows = [r for r in rows if r["split"] == "test"]
    assert summary["n_questions"] == len(test_rows)
    for mode in ("bm25", "semantic", "hybrid"):
        mean = sum(r["modes"][mode]["recall@5"] for r in test_rows) / len(test_rows)
        assert abs(mean - summary[mode]["recall@5"]) < 0.001


def test_every_award_label_has_wikipedia_evidence_and_none_is_left_unreviewed():
    films = {f["film"] for f in load("award_set.json")["films"]}
    evidence = load("award_label_evidence.json")["films"]
    assert {e["film"] for e in evidence} == films
    for e in evidence:
        assert isinstance(e["revision_id"], int) and e["article_url"].startswith("https://en.wikipedia.org/wiki/")
        assert e["status"].startswith("consistent"), e
        if "manual" in e["status"]:
            assert e["manual_review"], e


def test_retrieval_results_are_unchanged_by_the_verdict_fixes():
    before = json.loads((ROOT / "eval" / "results" / "before_fixes" / "summary.json").read_text())
    after = json.loads((ROOT / "eval" / "results" / "after_fixes" / "summary.json").read_text())
    assert before["retrieval"] == after["retrieval"]


def test_every_award_row_is_sourced_with_at_most_25_words():
    for row in load("award_set.json")["films"]:
        assert row["verified_by"] == "agent-sourced", row["film"]
        assert row["source_url"].startswith("https://en.wikipedia.org/wiki/"), row["film"]
        assert 0 < len(row["supporting_text"].split()) <= 25, row["film"]
        if row["truth"] == "no_win":
            assert row["absence_check"]["name_absent_from_all_pages"] is True, row["film"]

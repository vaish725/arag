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


def test_eval_md_results_block_matches_the_saved_results():
    import sys

    sys.path.insert(0, str(ROOT / "eval"))
    import render_eval_md

    text = (ROOT / "EVAL.md").read_text()
    a, b = text.index(render_eval_md.BEGIN), text.index(render_eval_md.END) + len(render_eval_md.END)
    assert text[a:b] == render_eval_md.render(), "EVAL.md is out of date: run python eval/render_eval_md.py"


def test_no_win_rows_record_variant_hits_with_a_review():
    for row in load("award_set.json")["films"]:
        check = row.get("absence_check")
        if not check:
            continue
        assert check["name_absent_from_all_pages"] is True and check["variants_checked"], row["film"]
        assert check["all_variants_absent"] == (not check["variant_hits"]), row["film"]
        if check["variant_hits"]:
            assert check["review"], f"{row['film']}: variant hits need a written review"


def test_requirements_eval_lists_the_eval_packages():
    packages = {line.strip() for line in (ROOT / "requirements-eval.txt").read_text().splitlines() if line.strip() and not line.startswith("#")}
    assert {"datasets", "matplotlib", "beautifulsoup4"} <= packages
    assert not (ROOT / "eval" / "requirements.txt").exists()


def test_the_documented_director_detection_failures_are_exactly_the_three_in_the_results():
    rows = json.loads((ROOT / "eval" / "results" / "after_fixes" / "awards_results.json").read_text())["rows"]
    failed = {r["film"] for r in rows if not r["modes"]["semantic"]["director_ok"]}
    assert failed == {"Ed Wood (film)", "Romeo + Juliet", "Last Action Hero"}, "update the Limits section of EVAL.md"
    assert "Kathryn Bigelow" in {r["modes"]["semantic"]["director_found"] for r in rows if r["film"] == "Last Action Hero"}

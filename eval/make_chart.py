#!/usr/bin/env python3
"""Draw eval/results/eval_summary.png from the before_fixes and after_fixes summaries (test split only).

Usage: python eval/make_chart.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODES = ["bm25", "semantic", "hybrid"]
LABELS = {"bm25": "BM25", "semantic": "Semantic", "hybrid": "Hybrid (RRF)"}
BLUE, ORANGE = "#0072B2", "#E69F00"            # Okabe-Ito, colour-blind safe
GREEN, RED, GREY = "#009E73", "#D55E00", "#9AA0A6"


def main() -> None:
    before = json.load(open("eval/results/before_fixes/summary.json"))
    after = json.load(open("eval/results/after_fixes/summary.json"))
    r = before["retrieval"]["test"]  # retrieval is unchanged by the verdict fixes (checked in run_eval)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # Panel 1: retrieval recall on the test split
    x = range(len(MODES))
    w = 0.36
    for off, k, colour in ((-w / 2, "recall@5", BLUE), (w / 2, "recall@10", ORANGE)):
        vals = [r[m][k] for m in MODES]
        bars = ax1.bar([i + off for i in x], vals, w, color=colour, label=k)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([LABELS[m] for m in MODES])
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("chunk-level recall of gold supporting facts")
    ax1.set_title(f"Retrieval, test split (n={r['n_questions']} questions)", fontsize=11, loc="left")
    ax1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    # Panel 2: director-award answers on the test split, before vs after the two verdict fixes
    rows = []
    for m in MODES:
        rows.append((f"{LABELS[m]}\nbefore fixes (held out)", before["awards"]["test"][m]))
        rows.append((f"{LABELS[m]}\nafter fixes (not held out)", after["awards"]["test"][m]))
    n = before["awards"]["test"]["n_films"]
    left = [0] * len(rows)
    for key, colour, label in (("correct", GREEN, "correct"), ("wrong", RED, "wrong"), ("no_answer", GREY, "no answer")):
        vals = [d[key] for _, d in rows]
        ax2.barh([name for name, _ in rows], vals, left=left, color=colour, label=label, height=0.7)
        for i, v in enumerate(vals):
            if v:
                ax2.text(left[i] + v / 2, i, str(v), ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax2.invert_yaxis()
    ax2.set_xlim(0, n)
    ax2.set_xlabel("films")
    ax2.tick_params(axis="y", labelsize=8)
    ax2.set_title(f"Director-award answers, test split (n={n} films)", fontsize=11, loc="left")
    ax2.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A-RAG evaluation (no LLM calls). 'After fixes' is NOT held out: the two faults were found on this split.", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("eval/results/eval_summary.png", dpi=150)
    print("wrote eval/results/eval_summary.png")


if __name__ == "__main__":
    main()

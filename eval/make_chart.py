#!/usr/bin/env python3
"""Draw eval/results/eval_summary.png from eval/results/summary.json (test split only).

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
    s = json.load(open("eval/results/summary.json"))
    r, a = s["retrieval"]["test"], s["awards"]["test"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel 1: retrieval recall on the held-out test split
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
    ax1.set_title(f"Retrieval, HotpotQA test split (n={r['n_questions']} questions)", fontsize=11, loc="left")
    ax1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    # Panel 2: director-award answers on the held-out test split
    left = [0] * len(MODES)
    for key, colour, label in (("correct", GREEN, "correct"), ("wrong", RED, "wrong"), ("no_answer", GREY, "no answer")):
        vals = [a[m][key] for m in MODES]
        ax2.barh([LABELS[m] for m in MODES], vals, left=left, color=colour, label=label)
        for i, v in enumerate(vals):
            if v:
                ax2.text(left[i] + v / 2, i, str(v), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        left = [l + v for l, v in zip(left, vals)]
    ax2.invert_yaxis()
    ax2.set_xlim(0, a["n_films"])
    ax2.set_xlabel("films")
    ax2.set_title(f"Director-award answers, test split (n={a['n_films']} films)", fontsize=11, loc="left")
    ax2.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("A-RAG evaluation: held-out test split only (no LLM calls)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("eval/results/eval_summary.png", dpi=150)
    print("wrote eval/results/eval_summary.png")


if __name__ == "__main__":
    main()

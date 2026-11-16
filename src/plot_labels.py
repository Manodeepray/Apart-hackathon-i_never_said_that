"""Label mix per step: 100% stacked columns of Defends / Softens / Corrects / Denies (+ Unclear).

    python src/plot_labels.py

Replies on the chain path, sample 0 only: S1 = paraphrase p0, S2, S3-S5 pooled over the three
blatancy chains. Each column is the label share among judged replies at that step (not the
first-exit label). Panels: model (rows) x claim type (columns: stated, implied).
Output: results/labels_by_step.{pdf,png,csv}; reads only data on disk (no API calls).
"""
import csv
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import CLAIMS, LABELS, ROOT, SUBJECTS, read_jsonl

OUT = ROOT / "results"
LABS = ["Defends", "Softens", "Corrects", "Denies", "Unclear"]
COL = {"Defends": "#2a78d6", "Softens": "#eda100", "Corrects": "#1baf7a", "Denies": "#eb6834", "Unclear": "#b5b3ad"}
INK = {"Defends": "white", "Softens": "#0b0b0b", "Corrects": "#0b0b0b", "Denies": "#0b0b0b", "Unclear": "#0b0b0b"}
TYPES = ("stated", "implied")
STEPS = [(1, "S1\nattribution"), (2, "S2\nbare doubt"), (3, "S3\nauthority"), (4, "S4\nstudy"), (5, "S5\nconsensus")]


def main():
    claims = {c["claim_id"]: c for c in read_jsonl(CLAIMS)}
    labs = [l for l in read_jsonl(LABELS) if l.get("parse_ok") and l["sample_idx"] == 0
            and (l["branch"] in ("s1_p0", "s2") or l["branch"].startswith("chain_b"))]
    models = [m for m in SUBJECTS if any(l["model"] == m for l in labs)]
    counts = {}
    for l in labs:
        t = claims[l["claim_id"]]["claim_type"]
        if t in TYPES:
            counts.setdefault((l["model"], t, l["step"]), Counter())[l["label"]] += 1

    rows = []
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(len(models), 2, figsize=(7.2, 2.7 * len(models)), sharey=True, squeeze=False)
    for i, m in enumerate(models):
        for j, t in enumerate(TYPES):
            ax = axes[i][j]
            for x, (s, _) in enumerate(STEPS):
                c = counts.get((m, t, s), Counter())
                n = sum(c.values())
                bottom = 0.0
                for lab in LABS:
                    p = 100 * c[lab] / n if n else 0
                    rows.append({"model": m, "claim_type": t, "step": s, "label": lab, "n": c[lab], "total": n,
                                 "pct": round(p, 1)})
                    if p == 0:
                        continue
                    ax.bar(x, p, bottom=bottom, width=0.72, color=COL[lab], edgecolor="white", linewidth=1.2)
                    if p >= 8:
                        ax.text(x, bottom + p / 2, f"{p:.0f}", ha="center", va="center", fontsize=7.5, color=INK[lab])
                    bottom += p
                ax.text(x, 101.5, f"n={n}", ha="center", va="bottom", fontsize=6.5, color="#52514e")
            ax.set_xticks(range(len(STEPS)), [lbl for _, lbl in STEPS], fontsize=7.5)
            ax.set_ylim(0, 108)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{m} · {t}", fontsize=9, color="#0b0b0b", loc="left")
            if j == 0:
                ax.set_ylabel("% of replies")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[l]) for l in LABS]
    fig.legend(handles, LABS, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"labels_by_step.{ext}", dpi=200)
    with open(OUT / "labels_by_step.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {OUT}/labels_by_step.{{pdf,png,csv}}")


if __name__ == "__main__":
    main()

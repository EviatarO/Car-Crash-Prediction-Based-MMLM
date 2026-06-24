"""
build_e3a_score_dist_overlay.py
===============================
e3a (epoch-7) counterpart of the e3b "positive score distribution, all 3 TTE
groups overlaid" figure. Same loader / binning / styling as
build_e3b_comparison_figures.fig_all_groups_overlay, but on the e3a test dumps.

Output: outputs/e3a_student_90clips/e3a_test_extracted/outputs/metrics/
        e3a_test_epoch07/e3a_score_dist_all_groups_overlay.png

Run:
  python student_training/scripts/build_e3a_score_dist_overlay.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

E3A = {
    "Private": REPO / "outputs/e3a_student_90clips/e3a_test_extracted/outputs/trained/e3a_test_epoch07.jsonl",
    "Public":  REPO / "outputs/e3a_student_90clips/e3a_test_extracted/outputs/trained/e3a_test_public_epoch07.jsonl",
}
FIG_DIR = REPO / "outputs/e3a_student_90clips/e3a_test_extracted/outputs/metrics/e3a_test_epoch07"
GROUP_TO_S = {0: 0.5, 1: 1.0, 2: 1.5}


def load(path):
    yt, ys, grp = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("score") is None or r.get("ground_truth") is None:
            continue
        yt.append(int(r["ground_truth"]))
        ys.append(float(r["score"]))
        grp.append(r.get("group"))
    return np.array(yt), np.array(ys), np.array([g if g is not None else -1 for g in grp])


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {half: load(E3A[half]) for half in ("Private", "Public")}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    bins = np.linspace(0, 1, 21)
    cols = {0: "#2ecc71", 1: "#f39c12", 2: "#8e44ad"}
    for ax, half in zip(axes, ("Private", "Public")):
        yt, ys, grp = data[half]
        for g in (0, 1, 2):
            m = (grp == g) & (yt == 1)
            ax.hist(ys[m], bins=bins, histtype="step", lw=2, color=cols[g],
                    label=f"{GROUP_TO_S[g]}s  (n={int(m.sum())}, μ={ys[m].mean():.2f})")
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("P(collision)")
        ax.set_title(half, fontsize=11)
        ax.legend(fontsize=8, title="positives by TTE")
    axes[0].set_ylabel("clip count")
    fig.suptitle("e3a — positive score distribution, all 3 TTE groups overlaid", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG_DIR / "e3a_score_dist_all_groups_overlay.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()

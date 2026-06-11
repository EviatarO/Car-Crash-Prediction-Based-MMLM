"""
build_e3b_comparison_figures.py
===============================
Comparison + combined-view figures for the e3b (step_000099 / ep3) evaluation,
placed in a dedicated  outputs/e3b_student_267clips_tte/figures/  folder.

Two families of figures:

  A) "See the diff"  — e3a (epoch-7) vs e3b (ep3), per TTE group AND overall:
       1. overall_ap_auc_e3a_vs_e3b.png      overall AP & AUC, both test halves
       2. ap_per_group_e3a_vs_e3b.png        per-TTE-group AP, grouped bars (1x2)
       3. score_slope_e3a_vs_e3b.png         mean score (pos/neg) vs TTE group — the
                                             saturation / slope view, both models

  B) "All groups together"  — e3b only, the 3 TTE groups on one axes:
       4. e3b_score_dist_all_groups_overlay.png   positive-score dists, 3 groups overlaid
       5. e3b_ap_all_groups_plus_overall.png      per-group AP + the pooled "All" bar
       6. e3b_tte_confidence_bar.png              class-balanced soft-confidence per group
                                                  (= (mean(score|coll)+(1-mean(score|safe)))/2)

Per-bucket AP is computed over ALL clips in the bucket (pos+neg), matching
evaluate_metrics.compute_group_metrics. Groups: 0=0.5s, 1=1.0s, 2=1.5s before event.

Run:
  python student_training/scripts/build_e3b_comparison_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]

E3A = {
    "Private": REPO / "outputs/e3a_student_90clips/e3a_test_extracted/outputs/trained/e3a_test_epoch07.jsonl",
    "Public":  REPO / "outputs/e3a_student_90clips/e3a_test_extracted/outputs/trained/e3a_test_public_epoch07.jsonl",
}
E3B = {
    "Private": REPO / "outputs/e3b_student_267clips_tte/e3b_test_private/outputs/trained/e3b_test_private_step000099.jsonl",
    "Public":  REPO / "outputs/e3b_student_267clips_tte/e3b_test_public/outputs/trained/e3b_test_public_step000099.jsonl",
}
FIG_DIR = REPO / "outputs/e3b_student_267clips_tte/figures"
GROUP_TO_S = {0: 0.5, 1: 1.0, 2: 1.5}
C_E3A, C_E3B = "#95a5a6", "#3498db"          # grey = e3a, blue = e3b
C_POS, C_NEG = "#c0504d", "#4f81bd"


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


def ap_auc(yt, ys):
    if yt.sum() == 0 or (1 - yt).sum() == 0:
        return float("nan"), float("nan")
    return average_precision_score(yt, ys), roc_auc_score(yt, ys)


def per_group_ap(yt, ys, grp):
    out = {}
    for g in (0, 1, 2):
        m = grp == g
        if m.sum() == 0:
            out[g] = (float("nan"), 0)
            continue
        ap, _ = ap_auc(yt[m], ys[m])
        out[g] = (ap, int(m.sum()))
    return out


# ---------------------------------------------------------------- A1: overall
def fig_overall(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, half in zip(axes, ("Private", "Public")):
        ap_a, au_a = ap_auc(*data["e3a"][half][:2])
        ap_b, au_b = ap_auc(*data["e3b"][half][:2])
        x = np.arange(2)
        w = 0.36
        ba = ax.bar(x - w/2, [ap_a, au_a], w, label="e3a (ep7)", color=C_E3A, edgecolor="black")
        bb = ax.bar(x + w/2, [ap_b, au_b], w, label="e3b (ep3)", color=C_E3B, edgecolor="black")
        for bars in (ba, bb):
            for b in bars:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
                        f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(["AP", "AUC"])
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        n = len(data["e3b"][half][0])
        ax.set_title(f"{half}  (n={n})", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("e3a vs e3b — overall AP & AUC (threshold-free)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG_DIR / "overall_ap_auc_e3a_vs_e3b.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"Saved: {p}")


# ---------------------------------------------------------------- A2: per-group AP
def fig_ap_per_group(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, half in zip(axes, ("Private", "Public")):
        ga = per_group_ap(*data["e3a"][half])
        gb = per_group_ap(*data["e3b"][half])
        labels = [f"{GROUP_TO_S[g]}s" for g in (0, 1, 2)]
        x = np.arange(3); w = 0.36
        ba = ax.bar(x - w/2, [ga[g][0] for g in (0,1,2)], w, label="e3a (ep7)", color=C_E3A, edgecolor="black")
        bb = ax.bar(x + w/2, [gb[g][0] for g in (0,1,2)], w, label="e3b (ep3)", color=C_E3B, edgecolor="black")
        for g, b in zip((0,1,2), bb):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
                    f"{gb[g][0]:.3f}\nn={gb[g][1]}", ha="center", va="bottom", fontsize=8)
        for g, b in zip((0,1,2), ba):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
                    f"{ga[g][0]:.3f}", ha="center", va="bottom", fontsize=8, color="#555")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("time before event")
        ax.set_ylim(0, 1.0); ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_ylabel("AP (per bucket, pos+neg)")
        ax.set_title(f"{half}", fontsize=11)
        ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("AP per TTE group — e3a vs e3b  (both ~flat: no AP-vs-TTE slope)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG_DIR / "ap_per_group_e3a_vs_e3b.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"Saved: {p}")


# ---------------------------------------------------------------- A3: score slope
def fig_score_slope(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    xs = [GROUP_TO_S[g] for g in (0, 1, 2)]
    for ax, half in zip(axes, ("Private", "Public")):
        for model, color in (("e3a", C_E3A), ("e3b", C_E3B)):
            yt, ys, grp = data[model][half]
            pos = [ys[(grp == g) & (yt == 1)].mean() for g in (0,1,2)]
            neg = [ys[(grp == g) & (yt == 0)].mean() for g in (0,1,2)]
            ax.plot(xs, pos, "-o", color=color, label=f"{model}  mean(score|collision)")
            ax.plot(xs, neg, "--s", color=color, alpha=0.6, label=f"{model}  mean(score|safe)")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xticks(xs); ax.invert_xaxis()
        ax.set_xlabel("time before event (s)   [→ closer to crash]")
        ax.set_title(half, fontsize=11)
        ax.legend(fontsize=7.5, loc="center left")
    axes[0].set_ylabel("mean predicted P(collision)")
    fig.suptitle("Score vs TTE — e3a vs e3b  (e3b positives saturate high → compressed range)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG_DIR / "score_slope_e3a_vs_e3b.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"Saved: {p}")


# ---------------------------------------------------------------- B4: all groups overlay
def fig_all_groups_overlay(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    bins = np.linspace(0, 1, 21)
    cols = {0: "#2ecc71", 1: "#f39c12", 2: "#8e44ad"}
    for ax, half in zip(axes, ("Private", "Public")):
        yt, ys, grp = data["e3b"][half]
        for g in (0, 1, 2):
            m = (grp == g) & (yt == 1)
            ax.hist(ys[m], bins=bins, histtype="step", lw=2, color=cols[g],
                    label=f"{GROUP_TO_S[g]}s  (n={int(m.sum())}, μ={ys[m].mean():.2f})")
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("P(collision)")
        ax.set_title(half, fontsize=11); ax.legend(fontsize=8, title="positives by TTE")
    axes[0].set_ylabel("clip count")
    fig.suptitle("e3b — positive score distribution, all 3 TTE groups overlaid", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG_DIR / "e3b_score_dist_all_groups_overlay.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"Saved: {p}")


# ---------------------------------------------------------------- B5: per-group AP + overall
def fig_ap_groups_plus_overall(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, half in zip(axes, ("Private", "Public")):
        yt, ys, grp = data["e3b"][half]
        gb = per_group_ap(yt, ys, grp)
        ap_all, _ = ap_auc(yt, ys)
        labels = [f"{GROUP_TO_S[g]}s" for g in (0,1,2)] + ["All"]
        vals = [gb[g][0] for g in (0,1,2)] + [ap_all]
        ns = [gb[g][1] for g in (0,1,2)] + [len(yt)]
        colors = ["#3498db"]*3 + ["#e67e22"]
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", width=0.6)
        for b, v, n in zip(bars, vals, ns):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.008,
                    f"{v:.3f}\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.0); ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_ylabel("Average Precision"); ax.set_title(half, fontsize=11)
    fig.suptitle("e3b — AP per TTE group + pooled (All)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG_DIR / "e3b_ap_all_groups_plus_overall.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"Saved: {p}")


# ---------------------------------------------------------------- B6: balanced soft-confidence
def fig_tte_confidence(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, half in zip(axes, ("Private", "Public")):
        yt, ys, grp = data["e3b"][half]
        labels, conf, ns = [], [], []
        for g in (0, 1, 2):
            mp = (grp == g) & (yt == 1); mn = (grp == g) & (yt == 0)
            if mp.sum() == 0 or mn.sum() == 0:
                continue
            c = (ys[mp].mean() + (1 - ys[mn].mean())) / 2
            labels.append(f"{GROUP_TO_S[g]}s"); conf.append(c)
            ns.append(int(mp.sum()) + int(mn.sum()))
        bars = ax.bar(labels, conf, color="#16a085", edgecolor="black", width=0.55)
        for b, c, n in zip(bars, conf, ns):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006,
                    f"{c:.3f}\nn={n}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.0); ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
        ax.set_title(half, fontsize=11); ax.set_xlabel("time before event")
    axes[0].set_ylabel("balanced soft confidence")
    fig.suptitle("e3b — class-balanced soft confidence per TTE group\n"
                 "bar = (mean(score|collision) + (1 - mean(score|safe))) / 2", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = FIG_DIR / "e3b_tte_confidence_bar.png"
    fig.savefig(p, dpi=150); plt.close(fig); print(f"Saved: {p}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {"e3a": {}, "e3b": {}}
    for half in ("Private", "Public"):
        data["e3a"][half] = load(E3A[half])
        data["e3b"][half] = load(E3B[half])
    fig_overall(data)
    fig_ap_per_group(data)
    fig_score_slope(data)
    fig_all_groups_overlay(data)
    fig_ap_groups_plus_overall(data)
    fig_tte_confidence(data)
    print(f"\nAll comparison figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()

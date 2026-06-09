"""
make_per_group_figures.py
=========================
Produce per-TTE-group (0.5 / 1.0 / 1.5 s) panels for a test eval JSONL:
  - <prefix>_cm_per_group.png        : 1x3 confusion matrices (threshold 0.5)
  - <prefix>_scoredist_per_group.png : 1x3 score distributions (collision vs safe)

Used to enrich the progress report (per-group view instead of one overall figure).

Usage:
  python student_training/scripts/make_per_group_figures.py \
      --results <eval.jsonl> --out_dir <dir> --prefix test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix

GROUP_TO_S = {0: 0.5, 1: 1.0, 2: 1.5}
THR = 0.5


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("score") is not None and r.get("ground_truth") is not None:
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for g in [0, 1, 2]:
        sub = [r for r in rows if r.get("group") == g]
        yt = np.array([int(r["ground_truth"]) for r in sub])
        ys = np.array([float(r["score"]) for r in sub])
        groups[g] = (yt, ys)

    # ---- Panel 1: per-group confusion matrices ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, g in zip(axes, [0, 1, 2]):
        yt, ys = groups[g]
        yp = (ys >= THR).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        cm = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred No", "Pred Yes"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True No", "True Yes"])
        acc = (tp + tn) / len(yt)
        ax.set_title(f"{GROUP_TO_S[g]}s before event  (n={len(yt)})\n"
                     f"Acc={acc:.3f}", fontsize=10)
    fig.suptitle(f"Confusion Matrix per TTE group (threshold=0.5){args.tag}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = out_dir / f"{args.prefix}_cm_per_group.png"
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"Saved: {p1}")

    # ---- Panel 2: per-group score distributions ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, g in zip(axes, [0, 1, 2]):
        yt, ys = groups[g]
        ax.hist(ys[yt == 1], bins=bins, alpha=0.65, color="#c0504d",
                label=f"collision (n={int((yt==1).sum())})")
        ax.hist(ys[yt == 0], bins=bins, alpha=0.65, color="#4f81bd",
                label=f"safe (n={int((yt==0).sum())})")
        ax.axvline(THR, color="gray", ls="--", lw=1)
        ap_g = average_precision_score(yt, ys)
        ax.set_title(f"{GROUP_TO_S[g]}s before event\nAP={ap_g:.3f}", fontsize=10)
        ax.set_xlabel("P(collision)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("clip count")
    fig.suptitle(f"Score distribution per TTE group{args.tag}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p2 = out_dir / f"{args.prefix}_scoredist_per_group.png"
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()

"""
plot_tte_confidence_v2.py
=========================
Cleaned version of the class-balanced soft-confidence figure (no error bars,
formula in the title, triangles labelled by exactly what they plot).

Per TTE bucket:
    red  triangle = mean score on COLLISION clips   (mean score | collision)
    blue triangle = mean score on SAFE clips         (mean score | safe)
    bar           = ( mean(score|collision) + (1 - mean(score|safe)) ) / 2

No threshold is applied to the scores; class membership comes from ground_truth.
No bootstrap / CI shown on this figure.

Output: <metrics_dir>/tte_confidence_bar_v2.png

Usage:
    python student_training/scripts/plot_tte_confidence_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
JSONL = (REPO_ROOT / "outputs" / "e3a_student_90clips" / "e3a_test_extracted"
         / "outputs" / "trained" / "e3a_test_epoch07.jsonl")
OUT_DIR = (REPO_ROOT / "outputs" / "e3a_student_90clips" / "e3a_test_extracted"
           / "outputs" / "metrics" / "e3a_test_epoch07")
GROUP_TO_SECONDS = {0: 0.5, 1: 1.0, 2: 1.5}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("score") is None or r.get("ground_truth") is None:
                continue
            rows.append(r)
    return rows


def main():
    rows = load_rows(JSONL)
    print(f"Loaded {len(rows)} rows")

    table = []
    for g in [0, 1, 2]:
        sub = [r for r in rows if r.get("group") == g]
        yt = np.array([int(r["ground_truth"]) for r in sub])
        ys = np.array([float(r["score"]) for r in sub])
        mean_coll = float(ys[yt == 1].mean())   # red triangle
        mean_safe = float(ys[yt == 0].mean())   # blue triangle
        combined = (mean_coll + (1.0 - mean_safe)) / 2.0
        table.append({"tte": GROUP_TO_SECONDS[g], "n": len(sub),
                      "mean_coll": mean_coll, "mean_safe": mean_safe,
                      "combined": combined})
        print(f"  {GROUP_TO_SECONDS[g]}s  n={len(sub):3d}  "
              f"mean|collision={mean_coll:.3f}  mean|safe={mean_safe:.3f}  "
              f"bar={combined:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{t['tte']}s before event" for t in table]
    x = np.arange(len(labels))
    comb = [t["combined"] for t in table]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, comb, width=0.5, color="#2e8b8b", edgecolor="black", zorder=2)

    for xi, t in zip(x, table):
        ax.plot(xi, t["mean_coll"], "^", color="#c0504d", ms=9, zorder=5)
        ax.plot(xi, t["mean_safe"], "v", color="#4f81bd", ms=9, zorder=5)
        ax.text(xi, t["combined"] + 0.015, f"{t['combined']:.3f}\nn={t['n']}",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="gray", ls="--", lw=1, zorder=1)
    # Legend proxies
    ax.plot([], [], "^", color="#c0504d", ms=9, label="mean score | collision  (red ▲)")
    ax.plot([], [], "v", color="#4f81bd", ms=9, label="mean score | safe  (blue ▼)")
    ax.plot([], [], ls="--", color="gray", lw=1, label="no separation (0.5)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Class-balanced soft confidence")
    ax.set_title(
        "E3a Test — Class-balanced soft confidence per TTE\n"
        "bar = ( mean(score|collision) + (1 − mean(score|safe)) ) / 2",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    fig.text(0.5, 0.005,
             "Downward trend is directional only; not statistically resolved at "
             "n=160–284 (differences lie within their 95% CI).",
             ha="center", va="bottom", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "tte_confidence_bar_v2.png"
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved figure: {out_path}")


if __name__ == "__main__":
    main()

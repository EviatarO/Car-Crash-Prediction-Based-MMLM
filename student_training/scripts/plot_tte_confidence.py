"""
plot_tte_confidence.py
======================
Class-balanced SOFT-CONFIDENCE metric per Time-to-Event (TTE) bucket, as an
alternative to AP-per-bucket. AP ranks on the binary 0/1 decision and hides the
model's confidence; this metric reads the ScoreHead probability directly.

Per bucket g:
    soft_tpr  = mean(score | gt==1)          # confidence on real collisions  -> 1
    soft_tnr  = mean(1 - score | gt==0)       # confidence on safe clips       -> 1
    combined  = (soft_tpr + soft_tnr) / 2     # class-balanced soft confidence

Identity:  combined = 0.5 + (mean_pos - mean_neg)/2 = 0.5 + gap/2.
So `combined` is an affine rescale of the Stage-B separation gap and equals the
SOFT (probabilistic) form of balanced accuracy. Threshold-free; base-rate
independent; bounded [0,1] with 0.5 = no separation.

Bootstrap (seed=42, N=5000, CPU/NumPy) gives the 95% CI per bucket and on the
pairwise differences, so we report HONESTLY whether the decline survives noise.

Outputs (into the metrics folder):
    tte_confidence_bar.png
    tte_confidence_summary.json

Usage:
    python student_training/scripts/plot_tte_confidence.py
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
RNG = np.random.default_rng(42)
N_BOOT = 5000


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


def combined_metric(yt: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Return (combined, soft_tpr, soft_tnr) for one bucket."""
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    soft_tpr = float(pos.mean())
    soft_tnr = float((1.0 - neg).mean())
    return (soft_tpr + soft_tnr) / 2.0, soft_tpr, soft_tnr


def bootstrap_combined(yt: np.ndarray, ys: np.ndarray,
                       n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Point estimate + 95% CI of `combined` via within-bucket resampling."""
    point, _, _ = combined_metric(yt, ys)
    n = len(yt)
    idx = np.arange(n)
    boots = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        s = RNG.choice(idx, size=n, replace=True)
        yt_b, ys_b = yt[s], ys[s]
        if yt_b.sum() == 0 or yt_b.sum() == n:   # need both classes
            continue
        boots[k], _, _ = combined_metric(yt_b, ys_b)
        k += 1
    boots = boots[:k]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_diff(yt_a, ys_a, yt_b, ys_b, n_boot: int = N_BOOT):
    """95% CI of combined(b) - combined(a). CI crossing 0 => not distinguishable."""
    na, nb = len(yt_a), len(yt_b)
    ia, ib = np.arange(na), np.arange(nb)
    diffs = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        sa = RNG.choice(ia, size=na, replace=True)
        sb = RNG.choice(ib, size=nb, replace=True)
        if yt_a[sa].sum() in (0, na) or yt_b[sb].sum() in (0, nb):
            continue
        ca, _, _ = combined_metric(yt_a[sa], ys_a[sa])
        cb, _, _ = combined_metric(yt_b[sb], ys_b[sb])
        diffs[k] = cb - ca
        k += 1
    diffs = diffs[:k]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    cb, _, _ = combined_metric(yt_b, ys_b)
    ca, _, _ = combined_metric(yt_a, ys_a)
    point = cb - ca
    return float(point), float(lo), float(hi), bool(lo <= 0.0 <= hi)


def main():
    rows = load_rows(JSONL)
    print(f"Loaded {len(rows)} rows")

    groups = {}
    for g in [0, 1, 2]:
        sub = [r for r in rows if r.get("group") == g]
        groups[g] = {
            "n": len(sub),
            "yt": np.array([int(r["ground_truth"]) for r in sub]),
            "ys": np.array([float(r["score"]) for r in sub]),
        }

    # Per-bucket combined + CI
    print("\n=== Class-balanced soft confidence per TTE (threshold-free) ===")
    table = []
    for g in [0, 1, 2]:
        yt, ys = groups[g]["yt"], groups[g]["ys"]
        comb, lo, hi = bootstrap_combined(yt, ys)
        _, stpr, stnr = combined_metric(yt, ys)
        gap = stpr - (1.0 - stnr)   # = mean_pos - mean_neg
        table.append({"tte": GROUP_TO_SECONDS[g], "n": groups[g]["n"],
                      "combined": comb, "ci_low": lo, "ci_high": hi,
                      "soft_tpr": stpr, "soft_tnr": stnr, "gap": gap})
        print(f"  {GROUP_TO_SECONDS[g]}s  n={groups[g]['n']:3d}  "
              f"combined={comb:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"(soft_tpr={stpr:.3f}, soft_tnr={stnr:.3f}, gap={gap:+.3f})")

    # Pairwise diffs
    print("\n=== Pairwise differences (CI crossing 0 => decline NOT significant) ===")
    diff_table = []
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        pt, lo, hi, z = bootstrap_diff(groups[a]["yt"], groups[a]["ys"],
                                       groups[b]["yt"], groups[b]["ys"])
        lbl = f"{GROUP_TO_SECONDS[b]}s - {GROUP_TO_SECONDS[a]}s"
        diff_table.append({"pair": lbl, "delta": pt, "ci_low": lo,
                           "ci_high": hi, "overlaps_zero": z})
        print(f"  {lbl}: delta={pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'OVERLAPS 0 (n.s.)' if z else 'SIGNIFICANT'}")

    decline_significant = not all(d["overlaps_zero"] for d in diff_table)
    verdict = ("decline is STATISTICALLY SIGNIFICANT (at least one pairwise CI excludes 0)"
               if decline_significant else
               "decline is a TREND WITHIN NOISE (all pairwise CIs include 0): consistent "
               "with closer=more-confident but below CI resolution")
    print(f"\nVERDICT: {verdict}")

    # ── Figure ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "tte_confidence_bar.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{t['tte']}s before event" for t in table]
        x = np.arange(len(labels))
        comb = [t["combined"] for t in table]
        err = [[t["combined"] - t["ci_low"] for t in table],
               [t["ci_high"] - t["combined"] for t in table]]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(x, comb, width=0.5, color="#2e8b8b", edgecolor="black",
                      yerr=err, capsize=6, error_kw={"lw": 1.3})
        # component markers (soft_tpr high, 1-soft_tnr low) for context
        for xi, t in zip(x, table):
            ax.plot(xi, t["soft_tpr"], "^", color="#c0504d", ms=8, zorder=5)
            ax.plot(xi, 1.0 - t["soft_tnr"], "v", color="#4f81bd", ms=8, zorder=5)
        for xi, t in zip(x, table):
            ax.text(xi, t["ci_high"] + 0.012, f"{t['combined']:.3f}\nn={t['n']}",
                    ha="center", va="bottom", fontsize=9)

        ax.axhline(0.5, color="gray", ls="--", lw=1, label="no separation (0.5)")
        ax.plot([], [], "^", color="#c0504d", ms=8, label="mean conf. | collision (soft-TPR)")
        ax.plot([], [], "v", color="#4f81bd", ms=8, label="mean score | safe (1-soft-TNR)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Class-balanced soft confidence")
        ax.set_title("E3a Test — Class-balanced soft confidence per TTE\n"
                     "(soft-TPR + soft-TNR)/2 · threshold-free · 95% bootstrap CI",
                     fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=140)
        print(f"\nSaved figure: {fig_path}")
    except Exception as e:
        print(f"[WARN] figure skipped: {e}")

    # ── Summary JSON ────────────────────────────────────────────────────────
    summary = {
        "metric": "class_balanced_soft_confidence = (mean(score|pos)+mean(1-score|neg))/2 "
                  "= 0.5 + gap/2 (soft balanced accuracy, threshold-free)",
        "per_bucket": table,
        "pairwise_diff": diff_table,
        "decline_significant": decline_significant,
        "verdict": verdict,
    }
    summary_path = OUT_DIR / "tte_confidence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

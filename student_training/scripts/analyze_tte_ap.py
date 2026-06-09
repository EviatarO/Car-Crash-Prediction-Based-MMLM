"""
analyze_tte_ap.py
=================
Diagnostic for the AP-vs-Time-to-Event (TTE) curve on the E3a test set.

Read-only over the existing eval results — NO model re-run. Answers three
questions (see plan 2026-05-30_Plan-Diagnose-AP-vs-TTE-Curve-E3a-Test):

  Stage A  Is the per-group AP difference real, or sampling noise?
           -> bootstrap 95% CI per group + bootstrap CI on pairwise AP diffs.
  Stage B  Is the per-group comparison even fair?
           -> base-rate table + per-class score distributions per group.
  Stage C  Does the training signal even reward a monotonic curve?
           -> graded-ness of teacher reasoning text across TTE (length + cues).

AP is computed with the SAME call as evaluate_metrics.py:107
(sklearn.metrics.average_precision_score) so numbers match the pipeline.

Usage:
    python student_training/scripts/analyze_tte_ap.py \
        --results outputs/.../e3a_test_epoch07.jsonl \
        --out_dir reports/figures \
        --report  reports/E3a_TTE-AP_diagnosis_2026-05-30.md
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

GROUP_TO_SECONDS = {0: 0.5, 1: 1.0, 2: 1.5}
RNG = np.random.default_rng(42)
N_BOOT = 5000

# Closing-speed / imminence cue words — proxy for "graded urgency" in reasoning
CUE_WORDS = [
    "rapidly", "imminent", "critically", "closing", "high speed", "high rate",
    "insufficient", "abruptly", "sudden", "cut across", "cut", "brake", "braking",
    "immediate", "about to", "within", "seconds",
]


# ── IO ────────────────────────────────────────────────────────────────────────
def load_rows(path: str) -> list[dict]:
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


# ── Bootstrap ─────────────────────────────────────────────────────────────────
def bootstrap_ap(y_true: np.ndarray, y_score: np.ndarray,
                 n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Return (point_AP, ci_low, ci_high) via stratified resampling with replacement."""
    point = float(average_precision_score(y_true, y_score))
    n = len(y_true)
    idx_all = np.arange(n)
    boots = np.empty(n_boot)
    valid = 0
    for b in range(n_boot):
        idx = RNG.choice(idx_all, size=n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:   # need both classes
            boots[b] = np.nan
            continue
        boots[b] = average_precision_score(yt, ys)
        valid += 1
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_ap_diff(yt_a, ys_a, yt_b, ys_b, n_boot: int = N_BOOT):
    """Bootstrap CI of AP(group_b) - AP(group_a). CI containing 0 => not distinguishable."""
    na, nb = len(yt_a), len(yt_b)
    ia, ib = np.arange(na), np.arange(nb)
    diffs = np.empty(n_boot)
    k = 0
    for _ in range(n_boot):
        sa = RNG.choice(ia, size=na, replace=True)
        sb = RNG.choice(ib, size=nb, replace=True)
        if yt_a[sa].sum() in (0, na) or yt_b[sb].sum() in (0, nb):
            continue
        ap_a = average_precision_score(yt_a[sa], ys_a[sa])
        ap_b = average_precision_score(yt_b[sb], ys_b[sb])
        diffs[k] = ap_b - ap_a
        k += 1
    diffs = diffs[:k]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(average_precision_score(yt_b, ys_b) - average_precision_score(yt_a, ys_a))
    crosses_zero = bool(lo <= 0.0 <= hi)
    return point, float(lo), float(hi), crosses_zero


# ── Reasoning graded-ness ─────────────────────────────────────────────────────
def cue_density(text: str) -> int:
    if not text:
        return 0
    t = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(w)}", t)) for w in CUE_WORDS)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_dir", default="reports/figures")
    ap.add_argument("--report", default="reports/E3a_TTE-AP_diagnosis_2026-05-30.md")
    args = ap.parse_args()

    rows = load_rows(args.results)
    y_true = np.array([int(r["ground_truth"]) for r in rows])
    y_score = np.array([float(r["score"]) for r in rows])

    overall_ap = float(average_precision_score(y_true, y_score))
    overall_auc = float(roc_auc_score(y_true, y_score))
    print(f"Loaded {len(rows)} rows")
    print(f"Overall AP  = {overall_ap:.4f}  (expect 0.7615)")
    print(f"Overall AUC = {overall_auc:.4f}  (expect 0.7844)")

    # Per-group arrays
    groups = {}
    for g in [0, 1, 2]:
        sub = [r for r in rows if r.get("group") == g]
        yt = np.array([int(r["ground_truth"]) for r in sub])
        ys = np.array([float(r["score"]) for r in sub])
        groups[g] = {"rows": sub, "yt": yt, "ys": ys}

    # ── Stage A: per-group AP + CI ──────────────────────────────────────────
    print("\n=== Stage A: per-group AP with 95% bootstrap CI ===")
    a_table = []
    for g in [0, 1, 2]:
        pt, lo, hi = bootstrap_ap(groups[g]["yt"], groups[g]["ys"])
        a_table.append((GROUP_TO_SECONDS[g], len(groups[g]["rows"]), pt, lo, hi))
        print(f"  {GROUP_TO_SECONDS[g]}s  n={len(groups[g]['rows']):3d}  "
              f"AP={pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (±{(hi-lo)/2:.4f})")

    # Pairwise diffs
    print("\n=== Stage A: pairwise AP differences (CI containing 0 => not distinguishable) ===")
    diff_table = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        pt, lo, hi, z = bootstrap_ap_diff(
            groups[a]["yt"], groups[a]["ys"], groups[b]["yt"], groups[b]["ys"]
        )
        lbl = f"{GROUP_TO_SECONDS[b]}s - {GROUP_TO_SECONDS[a]}s"
        diff_table.append((lbl, pt, lo, hi, z))
        print(f"  {lbl}: delta={pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'OVERLAPS 0 (n.s.)' if z else 'significant'}")

    # ── Stage B: base rates + per-class score distributions ─────────────────
    print("\n=== Stage B: base rate + per-class score distribution per group ===")
    b_table = []
    for g in [0, 1, 2]:
        yt, ys = groups[g]["yt"], groups[g]["ys"]
        pos_s = ys[yt == 1]
        neg_s = ys[yt == 0]
        base_rate = yt.mean()
        b_table.append((
            GROUP_TO_SECONDS[g], len(yt), base_rate,
            float(pos_s.mean()), float(np.median(pos_s)),
            float(neg_s.mean()), float(np.median(neg_s)),
            float(pos_s.mean() - neg_s.mean()),
        ))
        print(f"  {GROUP_TO_SECONDS[g]}s  base_rate={base_rate:.3f}  "
              f"pos mean={pos_s.mean():.3f} (med {np.median(pos_s):.3f})  "
              f"neg mean={neg_s.mean():.3f} (med {np.median(neg_s):.3f})  "
              f"gap={pos_s.mean()-neg_s.mean():+.3f}")

    # Disjointness check
    gids = {g: set(r["video_id"] for r in groups[g]["rows"]) for g in [0, 1, 2]}
    overlaps = {
        "0&1": len(gids[0] & gids[1]),
        "0&2": len(gids[0] & gids[2]),
        "1&2": len(gids[1] & gids[2]),
    }
    print(f"  video-id overlap across groups: {overlaps}")

    # ── Stage C: teacher/student reasoning graded-ness across TTE (positives) ─
    print("\n=== Stage C: reasoning graded-ness across TTE (positives only) ===")
    c_table = []
    for g in [0, 1, 2]:
        pos_rows = [r for r in groups[g]["rows"] if int(r["ground_truth"]) == 1]
        lengths = [len((r.get("verdict_reasoning") or "").split()) for r in pos_rows]
        cues = [cue_density(r.get("verdict_reasoning") or "") for r in pos_rows]
        c_table.append((
            GROUP_TO_SECONDS[g], len(pos_rows),
            float(np.mean(lengths)) if lengths else 0.0,
            float(np.mean(cues)) if cues else 0.0,
        ))
        print(f"  {GROUP_TO_SECONDS[g]}s  n_pos={len(pos_rows):3d}  "
              f"mean_words={np.mean(lengths):.1f}  mean_urgency_cues={np.mean(cues):.2f}")

    # ── Figure: per-group score distributions ──────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "tte_score_distributions.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, g in zip(axes, [0, 1, 2]):
            yt, ys = groups[g]["yt"], groups[g]["ys"]
            ax.hist(ys[yt == 1], bins=20, range=(0, 1), alpha=0.6,
                    label="collision", color="#c0504d")
            ax.hist(ys[yt == 0], bins=20, range=(0, 1), alpha=0.6,
                    label="safe", color="#4f81bd")
            ax.axvline(0.5, color="gray", ls="--", lw=1)
            ax.set_title(f"{GROUP_TO_SECONDS[g]}s before event (n={len(yt)})")
            ax.set_xlabel("model score P(collision)")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("clip count")
        fig.suptitle("E3a Test — score distribution by class, per TTE group", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=130)
        print(f"\nSaved figure: {fig_path}")
    except Exception as e:
        print(f"[WARN] figure skipped: {e}")

    # ── Dump machine-readable summary ───────────────────────────────────────
    summary = {
        "overall": {"ap": overall_ap, "auc": overall_auc, "n": len(rows)},
        "stage_a_group_ap": [
            {"tte": t, "n": n, "ap": pt, "ci_low": lo, "ci_high": hi}
            for (t, n, pt, lo, hi) in a_table
        ],
        "stage_a_pairwise_diff": [
            {"pair": lbl, "delta": pt, "ci_low": lo, "ci_high": hi, "overlaps_zero": z}
            for (lbl, pt, lo, hi, z) in diff_table
        ],
        "stage_b": [
            {"tte": t, "n": n, "base_rate": br, "pos_mean": pm, "pos_med": pmd,
             "neg_mean": nm, "neg_med": nmd, "pos_neg_gap": gap}
            for (t, n, br, pm, pmd, nm, nmd, gap) in b_table
        ],
        "stage_b_overlaps": overlaps,
        "stage_c": [
            {"tte": t, "n_pos": n, "mean_words": w, "mean_urgency_cues": c}
            for (t, n, w, c) in c_table
        ],
    }
    summary_path = Path(args.report).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

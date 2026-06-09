"""
compare_private_public.py
=========================
External-validity check for E3a Epoch-7: does the test result REPLICATE on the
unseen Public half?

Loads two eval JSONLs (Private 677 + Public 667), computes overall + per-group
FULL metrics (AP, AUC, F1@thr, P, R, confusion matrix) for each, bootstraps the
AP CIs, and tests whether Public AP falls inside Private's 95% CI (=> replicates).

Reuses the exact metric calls from evaluate_metrics.py and the bootstrap pattern
from analyze_tte_ap.py (seed=42, N_BOOT=5000).

Usage:
    python student_training/scripts/compare_private_public.py \
        --private outputs/.../trained/e3a_test_epoch07.jsonl \
        --public  outputs/.../trained/e3a_test_public_epoch07.jsonl \
        --out     outputs/.../metrics/e3a_test_public_epoch07/private_vs_public_comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

GROUP_TO_SECONDS = {0: 0.5, 1: 1.0, 2: 1.5}
RNG = np.random.default_rng(42)
N_BOOT = 5000
THRESHOLD = 0.5

REPO_ROOT = Path(__file__).resolve().parents[2]
DEF_PRIVATE = (REPO_ROOT / "outputs" / "e3a_student_90clips" / "e3a_test_extracted"
               / "outputs" / "trained" / "e3a_test_epoch07.jsonl")
DEF_PUBLIC = (REPO_ROOT / "outputs" / "e3a_student_90clips" / "e3a_test_extracted"
              / "outputs" / "trained" / "e3a_test_public_epoch07.jsonl")
DEF_OUT = (REPO_ROOT / "outputs" / "e3a_student_90clips" / "e3a_test_extracted"
           / "outputs" / "metrics" / "e3a_test_public_epoch07"
           / "private_vs_public_comparison.json")


def load(path: Path):
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
    return np.array(yt), np.array(ys), np.array(grp)


def full_metrics(yt, ys, thr=THRESHOLD):
    yp = (ys >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {
        "n": int(len(yt)), "base_rate": round(float(yt.mean()), 3),
        "ap": round(float(average_precision_score(yt, ys)), 4),
        "auc": round(float(roc_auc_score(yt, ys)), 4),
        "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": round(float((tp + tn) / len(yt)), 4),
    }


def boot_ap_ci(yt, ys, n_boot=N_BOOT):
    n = len(yt); idx = np.arange(n); vals = []
    for _ in range(n_boot):
        s = RNG.choice(idx, size=n, replace=True)
        if yt[s].sum() in (0, n):
            continue
        vals.append(average_precision_score(yt[s], ys[s]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def boot_ap_diff(yt_a, ys_a, yt_b, ys_b, n_boot=N_BOOT):
    """CI of AP(b) - AP(a). Independent resampling of each set."""
    na, nb = len(yt_a), len(yt_b); ia, ib = np.arange(na), np.arange(nb)
    diffs = []
    for _ in range(n_boot):
        sa = RNG.choice(ia, size=na, replace=True)
        sb = RNG.choice(ib, size=nb, replace=True)
        if yt_a[sa].sum() in (0, na) or yt_b[sb].sum() in (0, nb):
            continue
        diffs.append(average_precision_score(yt_b[sb], ys_b[sb])
                     - average_precision_score(yt_a[sa], ys_a[sa]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(average_precision_score(yt_b, ys_b) - average_precision_score(yt_a, ys_a))
    return point, float(lo), float(hi), bool(lo <= 0.0 <= hi)


def per_group(yt, ys, grp):
    out = {}
    for g in [0, 1, 2]:
        m = grp == g
        if m.sum() == 0:
            continue
        out[GROUP_TO_SECONDS[g]] = full_metrics(yt[m], ys[m])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", default=str(DEF_PRIVATE))
    ap.add_argument("--public", default=str(DEF_PUBLIC))
    ap.add_argument("--out", default=str(DEF_OUT))
    args = ap.parse_args()

    priv_p, pub_p = Path(args.private), Path(args.public)
    if not pub_p.exists():
        print(f"[WAIT] Public eval not found yet: {pub_p}\n"
              f"       Run Stage 3 (trained_eval on the Public manifest) first.")
        return

    yt_pr, ys_pr, g_pr = load(priv_p)
    yt_pu, ys_pu, g_pu = load(pub_p)

    ov_pr = full_metrics(yt_pr, ys_pr)
    ov_pu = full_metrics(yt_pu, ys_pu)
    ov_pr["ap_ci"] = [round(x, 4) for x in boot_ap_ci(yt_pr, ys_pr)]
    ov_pu["ap_ci"] = [round(x, 4) for x in boot_ap_ci(yt_pu, ys_pu)]

    diff, dlo, dhi, ns = boot_ap_diff(yt_pr, ys_pr, yt_pu, ys_pu)
    public_in_private_ci = ov_pr["ap_ci"][0] <= ov_pu["ap"] <= ov_pr["ap_ci"][1]
    replicates = public_in_private_ci or ns

    print("\n================  PRIVATE vs PUBLIC (E3a Epoch-7)  ================")
    print(f"{'metric':<12}{'PRIVATE':>14}{'PUBLIC':>14}")
    for k in ["n", "base_rate", "ap", "auc", "f1", "precision", "recall", "accuracy"]:
        print(f"{k:<12}{str(ov_pr[k]):>14}{str(ov_pu[k]):>14}")
    print(f"{'ap 95%CI':<12}{str(ov_pr['ap_ci']):>14}{str(ov_pu['ap_ci']):>14}")
    pr_tpfp = f"{ov_pr['tp']}/{ov_pr['fp']}"; pu_tpfp = f"{ov_pu['tp']}/{ov_pu['fp']}"
    pr_fntn = f"{ov_pr['fn']}/{ov_pr['tn']}"; pu_fntn = f"{ov_pu['fn']}/{ov_pu['tn']}"
    print(f"{'CM tp/fp':<12}{pr_tpfp:>14}{pu_tpfp:>14}")
    print(f"{'CM fn/tn':<12}{pr_fntn:>14}{pu_fntn:>14}")
    print(f"\nAP(public) - AP(private) = {diff:+.4f}  95% CI [{dlo:+.4f}, {dhi:+.4f}]  "
          f"{'overlaps 0 (n.s.)' if ns else 'SIGNIFICANT diff'}")
    print(f"Public AP inside Private 95% CI? {public_in_private_ci}")
    print(f"\nVERDICT: {'REPLICATES — Epoch-7 generalises to the unseen half.' if replicates else 'DOES NOT replicate — investigate (possible overfit/lucky split).'}")

    # Per-group AP side-by-side
    pg_pr = per_group(yt_pr, ys_pr, g_pr)
    pg_pu = per_group(yt_pu, ys_pu, g_pu)
    print("\nPer-group AP (Private | Public):")
    for s in [0.5, 1.0, 1.5]:
        a = pg_pr.get(s, {}); b = pg_pu.get(s, {})
        print(f"  {s}s  AP {a.get('ap'):>6} (n={a.get('n')})  |  {b.get('ap'):>6} (n={b.get('n')})")

    out = {
        "threshold": THRESHOLD,
        "overall": {"private": ov_pr, "public": ov_pu},
        "ap_diff_public_minus_private": {"point": round(diff, 4),
                                         "ci": [round(dlo, 4), round(dhi, 4)],
                                         "overlaps_zero": ns},
        "public_ap_in_private_ci": public_in_private_ci,
        "replicates": replicates,
        "per_group": {"private": pg_pr, "public": pg_pu},
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_p}")


if __name__ == "__main__":
    main()

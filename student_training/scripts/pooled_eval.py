"""
pooled_eval.py
===============
Pool the 677-clip Private test set with its disjoint 667-clip Public half (1,344
clips total, 568 source videos - Nexar dataset paper sec 4.1), then compare one or
more checkpoints against a reference arm with a paired bootstrap - the protocol
adopted 2026-09-26 after threshold-0.5 FP comparisons across different training
seeds turned out to be measuring seed/calibration noise, not a real effect (see
docs_agents/EXPERIMENTS.md's 2026-09-26 re-analysis).

Each arm needs two per-clip score files:
  - private: a semsup_train.py test_results_epNN.jsonl (677 rows, "ground_truth")
  - public:  a score_checkpoints_on_test.py *.jsonl (667 rows, "gt_verdict")
Row schema for each: {"video_id": ..., <label_key>: 0/1 or "YES"/"NO", "score": float}

Usage:
    python pooled_eval.py --arm NAME=private.jsonl,public.jsonl [--arm ...] \
        --ref NAME (must be one of the --arm names) [--n-boot 5000] [--seed 42] \
        [--fpr-at-tpr 0.85,0.90] [--out results.json]

Prints, per non-reference arm: pooled AP, paired dAP vs ref with a 95% CI (clip-level
bootstrap) and P(better), plus FPR at each requested matched-recall point (a
threshold chosen per-arm from ITS OWN score distribution at that recall, so a
calibration shift can't masquerade as a discrimination change the way threshold-0.5
FP counts did).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC


def to01(v):
    return int(v) if not isinstance(v, str) else (1 if v == "YES" else 0)


def load_scores(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "ground_truth" in r:
            y = to01(r["ground_truth"])
        elif "gt_verdict" in r:
            y = to01(r["gt_verdict"])
        else:
            raise KeyError(f"row has neither ground_truth nor gt_verdict: {r}")
        out[r["video_id"]] = (y, float(r["score"]))
    return out


def load_pooled(private_path, public_path):
    priv, pub = load_scores(private_path), load_scores(public_path)
    overlap = set(priv) & set(pub)
    assert not overlap, f"private/public video_id overlap ({len(overlap)}) - not disjoint sets"
    assert len(priv) == 677, f"{private_path}: expected 677 private rows, got {len(priv)}"
    assert len(pub) == 667, f"{public_path}: expected 667 public rows, got {len(pub)}"
    return {**priv, **pub}


def fpr_at_tpr(y, s, target_tpr):
    """Threshold chosen from this arm's OWN score distribution at the target recall,
    then FPR at that threshold - isolates discrimination from calibration, unlike
    comparing FP counts at a shared threshold=0.5 across arms with different seeds."""
    pos_scores = s[y == 1]
    thr = np.quantile(pos_scores, 1 - target_tpr)
    return float(((s >= thr) & (y == 0)).sum() / max(1, (y == 0).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                     help="NAME=private.jsonl,public.jsonl (repeatable)")
    ap.add_argument("--ref", required=True, help="arm NAME to compare every other arm against")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr-at-tpr", default="0.85,0.90")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, paths = spec.split("=", 1)
        priv, pub = paths.split(",")
        arms[name] = load_pooled(Path(priv), Path(pub))
    assert args.ref in arms, f"--ref {args.ref!r} not among --arm names {list(arms)}"

    ids = sorted(arms[args.ref])
    for name, d in arms.items():
        assert sorted(d) == ids, f"{name}: clip id set differs from ref {args.ref}"
    y = np.array([arms[args.ref][v][0] for v in ids])
    n_pos, n = int(y.sum()), len(y)
    print(f"n={n} pos={n_pos} neg={n - n_pos} (pooled Private+Public, {len(ids)} clips)")

    scores = {name: np.array([d[v][1] for v in ids]) for name, d in arms.items()}
    tpr_points = [float(x) for x in args.fpr_at_tpr.split(",")]

    rng = np.random.default_rng(args.seed)
    boot_idx = rng.integers(0, n, (args.n_boot, n))

    results = {}
    ref_s = scores[args.ref]
    ref_ap = AP(y, ref_s)
    print(f"\n{'arm':30s} {'AP':>8s} {'AUC':>8s}  " +
          "  ".join(f"FPR@TPR{t:.2f}" for t in tpr_points))
    row = {"ap": round(ref_ap, 4), "auc": round(AUC(y, ref_s), 4),
           "fpr_at_tpr": {t: round(fpr_at_tpr(y, ref_s, t), 4) for t in tpr_points}}
    print(f"{args.ref + ' (ref)':30s} {row['ap']:8.4f} {row['auc']:8.4f}  " +
          "  ".join(f"{row['fpr_at_tpr'][t]:10.4f}" for t in tpr_points))
    results[args.ref] = row

    for name, s in scores.items():
        if name == args.ref:
            continue
        this_ap, this_auc = AP(y, s), AUC(y, s)
        deltas = np.array([AP(y[i], s[i]) - AP(y[i], ref_s[i]) for i in boot_idx])
        ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
        row = {
            "ap": round(this_ap, 4), "auc": round(this_auc, 4),
            "delta_ap_vs_ref": round(this_ap - ref_ap, 4),
            "delta_ap_ci95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
            "p_better_than_ref": round(float((deltas > 0).mean()), 3),
            "fpr_at_tpr": {t: round(fpr_at_tpr(y, s, t), 4) for t in tpr_points},
        }
        results[name] = row
        print(f"{name:30s} {row['ap']:8.4f} {row['auc']:8.4f}  " +
              "  ".join(f"{row['fpr_at_tpr'][t]:10.4f}" for t in tpr_points))
        print(f"{'':30s} dAP={row['delta_ap_vs_ref']:+.4f} "
              f"95%CI=[{ci_lo:+.4f},{ci_hi:+.4f}] P(better)={row['p_better_than_ref']:.2f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n": n, "n_pos": n_pos, "ref": args.ref, "n_boot": args.n_boot,
                      "seed": args.seed, "arms": results}, f, indent=2)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()

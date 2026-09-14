"""
aa0_compare_rescore.py
======================
Gate G1 (AA.0 v2): compare a fresh re-score of a checkpoint against its recorded per-clip
scores. Accepts both score schemas used in this repo:
  {video_id, ground_truth: 0/1, score}          (semsup_train.py test_results_epNN.jsonl)
  {video_id, gt_verdict: "YES"/"NO", score}     (score_checkpoints_on_test.py NAME.jsonl)

Reports AP/AUC for each, dAP, max/mean |dscore|, and clips flipping across threshold 0.5.
G1 is NOT byte-identity (A1's recorded scores predate --deterministic; the project measured
max|dscore| 0.097, 5 flips, dAP 0.0009 between two nondeterministic scorings of the same
checkpoint). It passes only when ALL hold:
  |dAP| <= --noise-floor (0.0035, upper bound of the 2026-09-06 re-score CI)
  mean|dscore| <= --max-mean-delta (0.02)
  flips at 0.5 <= --max-flips (10, 2x the measured 5)
AP alone is insufficient: A1 vs a1cont - a DIFFERENT model - has dAP -0.003 (inside the
floor) but mean|dscore| 0.196 and 171 flips. The score-level criteria are what reject it.
The 0.02 mean threshold is a judgment (<80% confidence); record the value observed.

Usage:
  python student_training/scripts/aa0_compare_rescore.py \
      --recorded outputs/e4_vjepa_reason/a1_1761/test_results_ep04.jsonl \
      --rescored outputs/a1_compress256/scores/private_crop/A1.jsonl
"""
import argparse
import json
import sys

from sklearn.metrics import average_precision_score, roc_auc_score


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "ground_truth" in r:
            y = int(r["ground_truth"])
        else:
            y = 1 if str(r["gt_verdict"]).upper() == "YES" else 0
        out[r["video_id"]] = (y, float(r["score"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorded", required=True)
    ap.add_argument("--rescored", required=True)
    ap.add_argument("--noise-floor", type=float, default=0.0035)
    ap.add_argument("--max-mean-delta", type=float, default=0.02)
    ap.add_argument("--max-flips", type=int, default=10)
    args = ap.parse_args()

    a, b = load(args.recorded), load(args.rescored)
    if set(a) != set(b):
        sys.exit(f"clip sets differ: {len(set(a) ^ set(b))} ids not in both files")
    ids = sorted(a)
    if any(a[i][0] != b[i][0] for i in ids):
        sys.exit("labels differ between files for the same video_id")
    y = [a[i][0] for i in ids]
    sa, sb = [a[i][1] for i in ids], [b[i][1] for i in ids]
    ap_a, ap_b = average_precision_score(y, sa), average_precision_score(y, sb)
    auc_a, auc_b = roc_auc_score(y, sa), roc_auc_score(y, sb)
    d = [abs(p - q) for p, q in zip(sa, sb)]
    flips = sum((p >= 0.5) != (q >= 0.5) for p, q in zip(sa, sb))
    mean_d = sum(d) / len(d)
    checks = {"ap_within_noise_floor": abs(ap_b - ap_a) <= args.noise_floor,
              "mean_delta_ok": mean_d <= args.max_mean_delta,
              "flips_ok": flips <= args.max_flips}
    ok = all(checks.values())
    res = {"n": len(ids), "ap_recorded": round(ap_a, 4), "ap_rescored": round(ap_b, 4),
           "delta_ap": round(ap_b - ap_a, 4), "auc_recorded": round(auc_a, 4),
           "auc_rescored": round(auc_b, 4), "max_abs_delta_score": round(max(d), 4),
           "mean_abs_delta_score": round(mean_d, 4), "flips_at_0.5": flips,
           "thresholds": {"noise_floor": args.noise_floor, "max_mean_delta": args.max_mean_delta,
                          "max_flips": args.max_flips},
           "checks": checks, "G1_pass": ok}
    print(json.dumps(res, indent=2))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

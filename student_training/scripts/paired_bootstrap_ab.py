"""Paired bootstrap comparison of two test_results_epNN.jsonl checkpoints (same 677-clip
test set, matched by video_id). Used for every A-vs-B semantic-supervision comparison in this
thread (B_1761-parallel vs A1_1761 was done ad hoc in-session; this is the reusable version).

Usage:
    python paired_bootstrap_ab.py --a <path/to/test_results_epNN.jsonl> --a-name A1_1761 \
                                   --b <path/to/test_results_epNN.jsonl> --b-name B-v2 \
                                   [--n-boot 5000] [--seed 42]

Each input file is JSONL with rows: {"video_id": ..., "ground_truth": 0/1, "score": float}.
Rows are matched by video_id (order-independent, asserts identical id sets).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def load_scores(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[r["video_id"]] = (float(r["score"]), int(r["ground_truth"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="path to model A's test_results_epNN.jsonl")
    ap.add_argument("--b", required=True, help="path to model B's test_results_epNN.jsonl")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="optional path to write JSON result")
    args = ap.parse_args()

    a_rows = load_scores(args.a)
    b_rows = load_scores(args.b)

    a_ids, b_ids = set(a_rows), set(b_rows)
    if a_ids != b_ids:
        missing_in_b = a_ids - b_ids
        missing_in_a = b_ids - a_ids
        raise SystemExit(
            f"[FATAL] video_id sets differ - not the same test set.\n"
            f"  in A not B ({len(missing_in_b)}): {sorted(missing_in_b)[:10]}\n"
            f"  in B not A ({len(missing_in_a)}): {sorted(missing_in_a)[:10]}"
        )

    ids = sorted(a_ids)
    n = len(ids)
    a_scores = np.array([a_rows[i][0] for i in ids])
    b_scores = np.array([b_rows[i][0] for i in ids])
    y_a = np.array([a_rows[i][1] for i in ids])
    y_b = np.array([b_rows[i][1] for i in ids])
    if not np.array_equal(y_a, y_b):
        mism = int((y_a != y_b).sum())
        raise SystemExit(f"[FATAL] ground_truth mismatch on {mism}/{n} clips - not the same labels.")
    y = y_a

    ap_a = average_precision_score(y, a_scores)
    ap_b = average_precision_score(y, b_scores)
    auc_a = roc_auc_score(y, a_scores)
    auc_b = roc_auc_score(y, b_scores)

    rng = np.random.default_rng(args.seed)
    deltas_ap = np.empty(args.n_boot)
    deltas_auc = np.empty(args.n_boot)
    for i in range(args.n_boot):
        idx = rng.integers(0, n, size=n)  # same resample applied to both models (paired)
        ys = y[idx]
        if ys.sum() == 0 or ys.sum() == n:
            # degenerate resample (all one class) - AP/AUC undefined; redraw
            while ys.sum() == 0 or ys.sum() == n:
                idx = rng.integers(0, n, size=n)
                ys = y[idx]
        deltas_ap[i] = average_precision_score(ys, a_scores[idx]) - average_precision_score(ys, b_scores[idx])
        deltas_auc[i] = roc_auc_score(ys, a_scores[idx]) - roc_auc_score(ys, b_scores[idx])

    ci_ap = np.percentile(deltas_ap, [2.5, 97.5])
    ci_auc = np.percentile(deltas_auc, [2.5, 97.5])
    p_b_beats_a_ap = float((deltas_ap < 0).mean())   # fraction of resamples where B > A on AP
    p_b_beats_a_auc = float((deltas_auc < 0).mean())

    result = {
        "a_name": args.a_name, "b_name": args.b_name,
        "n_test": n, "n_boot": args.n_boot, "seed": args.seed,
        "point_estimate": {
            "ap_a": round(float(ap_a), 4), "ap_b": round(float(ap_b), 4),
            "auc_a": round(float(auc_a), 4), "auc_b": round(float(auc_b), 4),
            "delta_ap_a_minus_b": round(float(ap_a - ap_b), 4),
            "delta_auc_a_minus_b": round(float(auc_a - auc_b), 4),
        },
        "bootstrap": {
            "delta_ap_ci95": [round(float(ci_ap[0]), 4), round(float(ci_ap[1]), 4)],
            "delta_auc_ci95": [round(float(ci_auc[0]), 4), round(float(ci_auc[1]), 4)],
            "ci_excludes_zero_ap": bool(ci_ap[0] > 0 or ci_ap[1] < 0),
            "ci_excludes_zero_auc": bool(ci_auc[0] > 0 or ci_auc[1] < 0),
            "p_b_beats_a_on_ap": round(p_b_beats_a_ap, 4),
            "p_b_beats_a_on_auc": round(p_b_beats_a_auc, 4),
        },
    }

    print(f"\n{args.a_name} (n={n}): AP={ap_a:.4f}  AUC={auc_a:.4f}")
    print(f"{args.b_name} (n={n}): AP={ap_b:.4f}  AUC={auc_b:.4f}")
    print(f"\nDelta ({args.a_name} - {args.b_name}):")
    print(f"  AP:  {ap_a - ap_b:+.4f}   95% CI [{ci_ap[0]:+.4f}, {ci_ap[1]:+.4f}]"
          f"   {'EXCLUDES zero' if result['bootstrap']['ci_excludes_zero_ap'] else 'crosses zero'}"
          f"   P({args.b_name}>{args.a_name})={p_b_beats_a_ap:.1%}")
    print(f"  AUC: {auc_a - auc_b:+.4f}   95% CI [{ci_auc[0]:+.4f}, {ci_auc[1]:+.4f}]"
          f"   {'EXCLUDES zero' if result['bootstrap']['ci_excludes_zero_auc'] else 'crosses zero'}"
          f"   P({args.b_name}>{args.a_name})={p_b_beats_a_auc:.1%}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()

"""
stage_compare.py - paired per-seed comparison of training arms on the pooled 1,344-clip test set.

Each arm = NAME=<private_glob>|<public_glob>, with {seed} and {ep} placeholders, e.g.
  --arm "la-full=outputs/stage2/la-full-seed{seed}/train/test_results_ep{ep}.jsonl|outputs/stage2/la-full-seed{seed}/scores_public/la-full-seed{seed}-ep{ep}.jsonl"
--ref names the reference arm; every other arm is compared to it, same seed and epoch.

Reports per arm/epoch: mean±sd over seeds of AP, per-TTE AP, Kaggle mAP, FPR at 85% recall, FN/FP at
threshold 0.5; and paired deltas vs --ref with the count of seeds where the arm wins.
Pre-registered Stage 2 pass rule (DECISIONS.md, 2026-09-30) is checked at --gate-epoch.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as AP


def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        y = r.get("ground_truth", r.get("gt_verdict"))
        y = (1 if y == "YES" else 0) if isinstance(y, str) else int(y)
        d[r["video_id"]] = (y, int(r["group"]), float(r["score"]))
    return d


def pooled(priv, pub):
    a, b = load(priv), load(pub)
    assert not set(a) & set(b) and len(a) == 677 and len(b) == 667, (priv, pub, len(a), len(b))
    a.update(b)
    return a


def metrics(d):
    ids = sorted(d)
    y = np.array([d[i][0] for i in ids]); g = np.array([d[i][1] for i in ids]); s = np.array([d[i][2] for i in ids])
    tte = [AP(y[g == k], s[g == k]) for k in (0, 1, 2)]
    thr = np.quantile(s[y == 1], 0.15)
    return dict(ap=AP(y, s), t05=tte[0], t10=tte[1], t15=tte[2], kaggle=float(np.mean(tte)),
                fpr85=float((s[y == 0] >= thr).mean()),
                fn=int(((y == 1) & (s < 0.5)).sum()), fp=int(((y == 0) & (s >= 0.5)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", default="1,2,3")
    ap.add_argument("--gate-epoch", type=int, default=1)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    epochs = [int(x) for x in args.epochs.split(",")]
    arms = {}
    for spec in args.arm:
        name, paths = spec.split("=", 1)
        arms[name] = paths.split("|")
    assert args.ref in arms, f"--ref {args.ref} not among arms"

    M = {}
    for name, (pv, pb) in arms.items():
        for sd in seeds:
            for ep in epochs:
                a = Path(pv.format(seed=sd, ep=f"{ep:02d}")); b = Path(pb.format(seed=sd, ep=f"{ep:02d}"))
                if a.exists() and b.exists():
                    M[(name, sd, ep)] = metrics(pooled(a, b))

    keys = ["ap", "t05", "t10", "t15", "kaggle", "fpr85"]
    f = lambda v: f"{np.mean(v):.4f}±{np.std(v, ddof=1) if len(v) > 1 else 0:.4f}"
    for ep in epochs:
        print(f"\n=== epoch {ep}: mean±sd over seeds ===")
        print(f"{'arm':12s} n  " + "  ".join(f"{k:>15s}" for k in keys) + "    FN/FP@0.5 (sum)")
        for name in arms:
            rows = [M[(name, sd, ep)] for sd in seeds if (name, sd, ep) in M]
            if not rows:
                continue
            print(f"{name:12s} {len(rows)}  " + "  ".join(f"{f([r[k] for r in rows]):>15s}" for k in keys)
                  + f"    {sum(r['fn'] for r in rows)}/{sum(r['fp'] for r in rows)}")
        for name in arms:
            if name == args.ref:
                continue
            common = [sd for sd in seeds if (name, sd, ep) in M and (args.ref, sd, ep) in M]
            if not common:
                continue
            parts = []
            for k in keys:
                d = [M[(name, sd, ep)][k] - M[(args.ref, sd, ep)][k] for sd in common]
                better = sum((x < 0) if k == "fpr85" else (x > 0) for x in d)
                parts.append(f"{k} {np.mean(d):+.4f} ({better}/{len(d)})")
            print(f"  {name} - {args.ref}: " + " | ".join(parts))

    ep = args.gate_epoch
    print(f"\n=== pre-registered pass rule at epoch {ep} (vs {args.ref}) ===")
    for name in arms:
        if name == args.ref:
            continue
        common = [sd for sd in seeds if (name, sd, ep) in M and (args.ref, sd, ep) in M]
        if not common:
            continue
        d = {k: [M[(name, sd, ep)][k] - M[(args.ref, sd, ep)][k] for sd in common] for k in keys}
        ok15 = np.mean(d["t15"]) > 0
        ok_other = np.mean(d["t05"]) >= -0.005 and np.mean(d["t10"]) >= -0.005
        ok_fpr = np.mean(d["fpr85"]) <= 0.005
        verdict = "PASS" if (ok15 and ok_other and ok_fpr) else "FAIL"
        print(f"  {name}: 1.5s AP up={ok15} ({np.mean(d['t15']):+.4f}) | 0.5s/1.0s not down >0.005={ok_other} "
              f"| FPR@85 not worse >0.005={ok_fpr} -> {verdict}  (n={len(common)} seeds)")


if __name__ == "__main__":
    main()

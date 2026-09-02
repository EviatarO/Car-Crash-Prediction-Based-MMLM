#!/usr/bin/env python3
"""Pool K-fold val score dumps into one full-pool readout per arm.

Expects a run layout of <cv-root>/<arm>/fold_XX/val_scores_ep*.jsonl, i.e. one
semsup_train.py run per (arm, fold) with --dump-val-scores.

Reports, per arm:
  * POOLED AP/AUC over all folds' val rows together (every clip scored exactly once
    by a model that never saw it in training) -- this is the headline number, and it
    is computed on ~200 rows instead of the single-split 40.
  * per-fold AP/AUC mean +- std, which is the honest spread estimate the single-split
    run could not provide.

Pooling caveat, stated because it is easy to get wrong: scores from different folds
come from different models, so their absolute calibration differs. AP/AUC are RANK
metrics, and pooling ranks across differently-calibrated models is a mild
approximation -- it is standard practice for CV but is not identical to scoring all
200 with one model. The per-fold mean/std is the conservative reading; the pooled
number is the higher-powered one. Both are printed, deliberately.
"""
import argparse
import json
import re
import statistics
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score


def latest_dump(fold_dir):
    """Highest-numbered val_scores_epNN.jsonl in a fold dir.

    NOTE: this takes the LAST epoch, not the --select-by best one. If a run selects a
    different epoch, point --epoch at it explicitly; silently mixing 'best' and 'last'
    across arms would be an unfair comparison.
    """
    dumps = sorted(fold_dir.glob("val_scores_ep*.jsonl"),
                   key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    return dumps[-1] if dumps else None


def read_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def row_label(r):
    """val_scores_epNN.jsonl (semsup_train.py --dump-val-scores) writes 'label' as an
    int 0/1 - NOT 'gt_verdict'. A row carrying gt_verdict/gt (e.g. scores/*.jsonl-style
    rows) is supported too, so this also works if ever pointed at that schema."""
    if "label" in r:
        return int(r["label"])
    v = r.get("gt_verdict", r.get("gt"))
    if v is None:
        raise KeyError(f"row has neither 'label' nor 'gt_verdict'/'gt': {r}")
    return 1 if str(v).upper() == "YES" else 0


def metrics(rows):
    y = [row_label(r) for r in rows]
    s = [float(r["score"]) for r in rows]
    if len(set(y)) < 2:
        return None, None          # AUC/AP undefined on a single-class fold
    return average_precision_score(y, s), roc_auc_score(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", required=True,
                     help="dir containing <arm>/fold_XX/ subdirs")
    ap.add_argument("--arms", nargs="+", default=["vision", "v10", "v12", "v12shuf"])
    ap.add_argument("--epoch", type=int, default=None,
                     help="use val_scores_ep<N>.jsonl instead of the last epoch")
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    ap.add_argument("--dump-pooled-scores-dir", default=None,
                     help="also write <dir>/<arm>.jsonl, one row per clip using the "
                          "score from whichever fold held it out (i.e. every clip's "
                          "score is out-of-sample - strictly cleaner than the original "
                          "single-split scores/<arm>.jsonl, where train-split clips "
                          "were scored in-sample). Schema matches scores/<arm>.jsonl "
                          "(frames_dir/score/gt_verdict/video_id/arm/"
                          "requested_time_to_event) so build_semtest200_comparison.py "
                          "can consume it unchanged via --scores-dir.")
    args = ap.parse_args()

    root = Path(args.cv_root)
    summary = {}
    print(f"{'arm':<10} {'pooled_AP':>10} {'pooled_AUC':>11} {'fold_AP':>16} {'fold_AUC':>16} {'n':>5}")
    for arm in args.arms:
        arm_dir = root / arm
        if not arm_dir.is_dir():
            print(f"{arm:<10} -- missing ({arm_dir})")
            continue
        pooled, fold_aps, fold_aucs = [], [], []
        for fold_dir in sorted(arm_dir.glob("fold_*")):
            dump = (fold_dir / f"val_scores_ep{args.epoch:02d}.jsonl") if args.epoch \
                else latest_dump(fold_dir)
            if dump is None or not dump.exists():
                print(f"  [warn] {arm}/{fold_dir.name}: no val_scores dump, skipped")
                continue
            rows = read_rows(dump)
            pooled.extend(rows)
            a, u = metrics(rows)
            if a is not None:
                fold_aps.append(a)
                fold_aucs.append(u)
        if not pooled:
            print(f"{arm:<10} -- no data")
            continue
        pa, pu = metrics(pooled)
        # stdev needs >=2 points; a 1-fold run reports 0 spread rather than crashing.
        sd = (lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0)
        print(f"{arm:<10} {pa:>10.4f} {pu:>11.4f} "
              f"{statistics.mean(fold_aps):>8.4f}+-{sd(fold_aps):<6.4f} "
              f"{statistics.mean(fold_aucs):>8.4f}+-{sd(fold_aucs):<6.4f} {len(pooled):>5d}")
        summary[arm] = {"pooled_ap": pa, "pooled_auc": pu,
                        "fold_ap_mean": statistics.mean(fold_aps), "fold_ap_std": sd(fold_aps),
                        "fold_auc_mean": statistics.mean(fold_aucs), "fold_auc_std": sd(fold_aucs),
                        "n_rows": len(pooled), "n_folds": len(fold_aps)}

        if args.dump_pooled_scores_dir:
            dump_dir = Path(args.dump_pooled_scores_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            seen = set()
            with open(dump_dir / f"{arm}.jsonl", "w", encoding="utf-8") as f:
                for r in pooled:
                    fd = r["frames_dir"]
                    if fd in seen:
                        # A clip can only be held out by ONE fold under an exact
                        # partition (make_semtest200_folds.py asserts this) - a repeat
                        # means folds overlapped and the pooled AP above is unreliable.
                        raise SystemExit(f"{arm}: {fd} appears in more than one fold's "
                                          f"val dump - folds are not a clean partition, "
                                          f"do not trust this run's metrics")
                    seen.add(fd)
                    f.write(json.dumps({
                        "arm": arm, "frames_dir": fd, "video_id": r["video_id"],
                        "score": float(r["score"]),
                        "gt_verdict": "YES" if row_label(r) == 1 else "NO",
                        "requested_time_to_event": r.get("tte"),
                    }) + "\n")
            print(f"  [dump] wrote {dump_dir / (arm + '.jsonl')} ({len(seen)} rows, "
                  f"each out-of-sample)")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()

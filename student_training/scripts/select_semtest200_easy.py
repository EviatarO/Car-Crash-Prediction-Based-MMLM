"""
select_semtest200_easy.py
==========================
SemTest-200-v2, change 1: add 100 easy A0-correct clips to the original 200-clip pool.

WHY
---
The original selection.jsonl (built by select_semtest200_recovery.py) is entirely
A0-error-driven: every one of its 100 negatives scores a0_score >= 0.501, i.e. it is
100% false positives - there is not one true negative in the pool. Measured
consequence: A0's OWN ranking AUC on this pool is 0.2300 (val) / 0.4130 (train), far
below chance - the pool is rank-inverted by construction, not merely hard.

A pool with no easy anchor gives the crash-CE gradient only one signal: "push down
whatever you currently score high" - a translation, not a discrimination. Adding easy
A0-correct clips (including real true negatives, which the pool currently lacks
entirely) gives the loss a contrast to separate against.

This does NOT dilute the hard-subset readout: every downstream metric is reported
STRATIFIED (hard-200 / easy-100 / all-300), never pooled-only. See
build_semtest200_comparison.py's metrics_stratified sheet.

WHAT
----
    easy_TN   60 clips   a0_score < 0.20   AND gt_verdict == NO   (the empty cell)
    easy_TP   40 clips   a0_score > 0.85   AND gt_verdict == YES  (TP_fill already
                                                                    supplies 50 correct
                                                                    positives)

Constraints enforced:
    - video-disjoint from the ORIGINAL 200-clip pool (--base-selection)
    - one window per video (matching the original pool's own invariant)
    - split 80/20 train/val PER new tier, independently, same convention as the
      original pool's per-bucket split - so the new tiers stratify correctly when
      make_semtest200_folds.py's (gt_verdict, source) stratification runs over the
      combined 300-row pool.

Availability was checked before writing this script (2026-08-29, ad-hoc query against
A0_full4446.jsonl excluding the existing 200): 631 unique videos satisfy easy_TN's
criterion, 621 satisfy easy_TP's - no shortage at these thresholds.

Usage:
  python select_semtest200_easy.py \
      --a0-scores ../../outputs/semtest200/A0_full4446.jsonl \
      --manifest ../../dataset/manifests/train4500_hires.jsonl \
      --train-xlsx ../../dataset/train.xlsx \
      --base-selection ../../outputs/semtest200/selection.jsonl \
      --out-dir ../../outputs/semtest200_v2
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from openpyxl import Workbook

from select_semtest200_recovery import (
    HEADER_FILL, HEADER_FONT, build_row, load_a0_scores, load_manifest,
    load_response_times,
)


def load_base_video_ids(path):
    return {json.loads(l)["video_id"] for l in open(path, encoding="utf-8") if l.strip()}


def pick_tier(manifest, a0, exclude_vids, gt_wanted, score_pred, n_wanted, seed):
    """One row per video_id, gt_verdict == gt_wanted, score_pred(a0_score) True,
    video_id not already used anywhere. Shuffled before truncation so which n_wanted
    videos get chosen (out of a larger eligible pool) is not an artifact of manifest
    order."""
    seen_vid = set()
    candidates = []
    for m in manifest:
        vid = m["video_id"]
        if vid in exclude_vids or vid in seen_vid:
            continue
        s = a0.get(m["frames_dir"])
        if s is None or s["gt_verdict"] != gt_wanted or not score_pred(s["score"]):
            continue
        seen_vid.add(vid)
        candidates.append((m, s))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < n_wanted:
        raise SystemExit(f"only {len(candidates)} eligible candidates for gt={gt_wanted}, "
                          f"need {n_wanted}")
    return candidates[:n_wanted]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-scores", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train-xlsx", required=True)
    ap.add_argument("--base-selection", required=True,
                     help="the ORIGINAL 200-clip selection.jsonl - new clips must be "
                          "video-disjoint from it")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tn-n", type=int, default=60)
    ap.add_argument("--tn-max-score", type=float, default=0.20)
    ap.add_argument("--tp-n", type=int, default=40)
    ap.add_argument("--tp-min-score", type=float, default=0.85)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a0 = load_a0_scores(args.a0_scores)
    manifest = load_manifest(args.manifest)
    rt = load_response_times(args.train_xlsx)
    base_vids = load_base_video_ids(args.base_selection)
    print(f"[load] a0_scores={len(a0)}  manifest={len(manifest)}  "
          f"base pool videos={len(base_vids)} (excluded)")

    used_vids = set(base_vids)   # grows as each tier claims videos, so easy_TN and
                                 # easy_TP cannot double-claim the same video either
    tiers = [
        ("easy_TN", "NO", (lambda s: s < args.tn_max_score), args.tn_n),
        ("easy_TP", "YES", (lambda s: s > args.tp_min_score), args.tp_n),
    ]

    new_rows = []
    for source, gt, pred, n_wanted in tiers:
        picked = pick_tier(manifest, a0, used_vids, gt, pred, n_wanted, args.seed)
        used_vids.update(m["video_id"] for m, _ in picked)
        n_val = max(1, round(n_wanted * args.val_frac))
        for i, (m, s) in enumerate(picked):
            row = build_row(m, s, rt.get(m["video_id"]))
            row["source"] = source
            row["split"] = "val" if i < n_val else "train"
            new_rows.append(row)
        print(f"[tier] {source}: n={n_wanted}  train={n_wanted - n_val}  val={n_val}")

    print("[new rows source/split]", dict(Counter((r["source"], r["split"]) for r in new_rows)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "selection_easy100.jsonl", "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")

    headers = ["video_id", "frames_dir", "gt_verdict", "horizon_label", "source",
               "a0_score", "response_time", "split"]
    wb = Workbook()
    ws = wb.active
    ws.title = "selection_easy100"
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    for r in new_rows:
        ws.append([r["video_id"], r["frames_dir"], r["gt_verdict"], r["horizon_label"],
                   r["source"], round(r["a0_score"], 4), r.get("response_time"), r["split"]])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:H{len(new_rows) + 1}"
    wb.save(out_dir / "selection_easy100.xlsx")

    print(f"[wrote] {out_dir / 'selection_easy100.jsonl'} ({len(new_rows)} rows)")
    print(f"[wrote] {out_dir / 'selection_easy100.xlsx'}")


if __name__ == "__main__":
    main()

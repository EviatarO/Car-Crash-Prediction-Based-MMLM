#!/usr/bin/env python3
"""Build K stratified cross-validation folds for the SemTest-200 pool.

WHY THIS EXISTS
---------------
The original SemTest-200 run used a single fixed 160/40 split. At n=40 the val-AP
confidence interval is roughly +-0.15, which is wider than the entire observed spread
between arms (0.515-0.542) -- i.e. the experiment could not, even in principle,
distinguish any arm from any other. K-fold fixes this properly: every clip lands in
val exactly once, so the pooled readout is over all 200 clips rather than 40. That is
a real 5x increase in evaluation data, not just seed-averaging.

Stratification is on (gt_verdict, source) so each fold carries the same class balance
AND the same tier composition (FN_near / FN_wide / TP_fill / FP_near_boundary /
FP_fill) as the pool. Without tier stratification a fold could end up with, say, all
the FN_wide clips, and its val AP would be incomparable to the others.

Splitting is by video_id, never by row: the current pool happens to be one window per
video, but that invariant is not enforced anywhere upstream, and a video appearing in
both train and val would leak.

Emits fold_XX_val_vids.txt files consumable directly by
`semsup_train.py --val-video-ids`.
"""
import argparse
import collections
import json
import random
from pathlib import Path


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def build_folds(rows, k, seed):
    """Round-robin assignment within each (gt_verdict, source) stratum.

    Round-robin (rather than random assignment) guarantees fold sizes stay within 1
    of each other per stratum, which random assignment does not at these small counts.
    """
    by_vid = {}
    for r in rows:
        # One entry per video; if a video ever has multiple windows, its stratum is
        # taken from the first row seen -- all its rows still move together.
        by_vid.setdefault(r["video_id"], r)

    strata = collections.defaultdict(list)
    for vid, r in by_vid.items():
        strata[(r["gt_verdict"], r.get("source", "?"))].append(vid)

    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for key in sorted(strata):
        vids = sorted(strata[key])          # sort first so seed fully determines order
        rng.shuffle(vids)
        # Rotate the starting fold per stratum so the remainder-of-division does not
        # always land on fold 0, which would make fold 0 systematically largest.
        offset = rng.randrange(k)
        for i, vid in enumerate(vids):
            folds[(i + offset) % k].append(vid)
    return folds, by_vid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="../../outputs/semtest200/selection.jsonl")
    ap.add_argument("--out-dir", default="../../outputs/semtest200/folds")
    ap.add_argument("-k", "--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.selection)
    folds, by_vid = build_folds(rows, args.folds, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"pool: {len(rows)} rows / {len(by_vid)} videos -> {args.folds} folds "
          f"(seed {args.seed})")
    manifest = {}
    for i, vids in enumerate(folds, start=1):
        p = out_dir / f"fold_{i:02d}_val_vids.txt"
        # newline='\n': --val-video-ids is read on Linux (pod); Windows CRLF would
        # append '\r' to every id and match nothing.
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for v in sorted(vids):
                fh.write(f"{v}\n")
        sub = [by_vid[v] for v in vids]
        pos = sum(1 for r in sub if r["gt_verdict"] == "YES")
        src = collections.Counter(r.get("source", "?") for r in sub)
        manifest[f"fold_{i:02d}"] = {"n": len(vids), "pos": pos, "neg": len(vids) - pos,
                                      "sources": dict(src), "path": str(p)}
        print(f"  fold {i}: n={len(vids):3d}  pos={pos:3d} neg={len(vids)-pos:3d}  "
              + "  ".join(f"{k}={v}" for k, v in sorted(src.items())))

    # Assert the partition is exact -- a silent overlap would leak train into val.
    allv = [v for f in folds for v in f]
    assert len(allv) == len(set(allv)) == len(by_vid), \
        f"fold partition is not exact: {len(allv)} assigned, {len(set(allv))} unique, {len(by_vid)} videos"
    print("[ok] folds are an exact partition of the pool (no overlap, no drops)")

    (out_dir / "folds_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {args.folds} fold files + folds_manifest.json to {out_dir}")


if __name__ == "__main__":
    main()

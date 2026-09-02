"""
make_semtest200_shuffled.py
============================
SemTest-200 shuffle control (2026-08-26 plan, Phase 1e / registered as P6).

Permutes captions WITHIN class (YES<->YES, NO<->NO) over the SemTest-200 caption file,
seeded, so the resulting file has the same 200 (video_id, gt_verdict) rows but each now
carries a DIFFERENT clip's caption of the same class. If the semantic-trained arm (v12)
beats vision-only but v12-shuffled does NOT, that isolates the effect to caption CONTENT
rather than mere caption presence (a regularization/class-signal effect would show up on
the shuffled arm too, since class is preserved).

A derangement is enforced within each class (no row keeps its own caption) - not
statistically necessary, but removes the trivial confound of a same-vs-shuffled tie
being partly explained by unshuffled rows.

Usage:
  python make_semtest200_shuffled.py \
    --captions ../../outputs/semtest200/Caption_semtest200_V12.jsonl \
    --out ../../outputs/semtest200/Caption_semtest200_V12_shuffled.jsonl \
    --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


def derangement(n, rng):
    """A random permutation of range(n) with no fixed points. Retries until one is
    found - for n in the tens, this converges in a handful of attempts (P(no fixed
    point) -> 1/e as n grows, so ~e ~= 2.7 tries expected even at small n)."""
    idx = list(range(n))
    if n < 2:
        return idx
    for _ in range(10_000):
        rng.shuffle(idx)
        if all(idx[i] != i for i in range(n)):
            return idx
    raise RuntimeError(f"could not find a derangement of {n} items in 10,000 tries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.captions, encoding="utf-8") if l.strip()]
    if any(r.get("caption", "").startswith("PLACEHOLDER-NOT-A-CAPTION") for r in rows):
        raise RuntimeError(f"{args.captions} contains PLACEHOLDER captions - point this "
                            f"at a real SemTest-200 caption file, not a crash-only pool.")

    by_class = defaultdict(list)
    for i, r in enumerate(rows):
        by_class[r["gt_verdict"]].append(i)

    rng = random.Random(args.seed)
    caption_for = {}   # row index -> shuffled caption
    for gt, idxs in sorted(by_class.items()):
        perm = derangement(len(idxs), rng)
        for local_i, local_j in enumerate(perm):
            caption_for[idxs[local_i]] = rows[idxs[local_j]]["caption"]
        print(f"[shuffle] class={gt}  n={len(idxs)}  derangement OK")

    out_rows = []
    for i, r in enumerate(rows):
        r2 = dict(r)
        r2["caption_original"] = r["caption"]
        r2["caption"] = caption_for[i]
        out_rows.append(r2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[wrote] {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()

"""sample_val_check_clips.py -- draw a balanced, distinct-clip sample for a
leakage-judge power increase (see the 2026-08-11 V12 validation round).

WHY: the 18-clip val set gives the leakage judge only n=18 (one-sided binomial
test vs chance: p~=0.12 for 12/18 -- not significant). This script draws
ADDITIONAL distinct clips from the full 4,446-window pool (1,482 distinct
video_ids, natural 13.2% A0-failure rate) so the judge can be re-run at
n=18+82=100, tightening the CI enough to actually decide the question.

Distinct-clip, not distinct-window: a video_id can appear at up to 3 TTE/MID
buckets in train4500_hires.jsonl. Sampling by ROW would let the same clip's
sibling windows dominate the sample and would not test what we care about
(generalization across clips). One row per unique video_id (first occurrence
in file order, deterministic) is kept as the candidate pool before sampling.

Excludes every val_e3a.jsonl video_id (confirmed zero overlap already, this
is a defensive re-check, not a no-op).

    python student_training/scripts/sample_val_check_clips.py \
        --manifest dataset/manifests/train4500_hires.jsonl \
        --val-manifest dataset/manifests/val_e3a.jsonl \
        --n-per-class 41 --seed 0 \
        --out dataset/manifests/val82_v12_check.jsonl
"""
import argparse
import json
import random
from pathlib import Path


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val-manifest", required=True,
                     help="clips to EXCLUDE from the sampling pool")
    ap.add_argument("--n-per-class", type=int, default=41)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.manifest))
    exclude = {r["video_id"] for r in load_jsonl(Path(args.val_manifest))}

    # one row per distinct video_id, first occurrence, excluding val clips
    seen, distinct = set(), []
    for r in rows:
        vid = r["video_id"]
        if vid in exclude or vid in seen:
            continue
        seen.add(vid)
        distinct.append(r)

    pos = [r for r in distinct if int(r["event_occurs"]) == 1]
    neg = [r for r in distinct if int(r["event_occurs"]) == 0]
    print(f"[pool] {len(distinct)} distinct clips available "
          f"({len(pos)} positive / {len(neg)} negative), "
          f"{len(exclude)} val clips excluded")

    if len(pos) < args.n_per_class or len(neg) < args.n_per_class:
        raise SystemExit(f"Not enough clips: need {args.n_per_class}/class, "
                          f"have pos={len(pos)} neg={len(neg)}")

    rng = random.Random(args.seed)
    sample = rng.sample(pos, args.n_per_class) + rng.sample(neg, args.n_per_class)
    rng.shuffle(sample)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    got_pos = sum(1 for r in sample if int(r["event_occurs"]) == 1)
    print(f"[sample] wrote {len(sample)} rows to {out_path} "
          f"({got_pos} positive / {len(sample) - got_pos} negative), seed={args.seed}")


if __name__ == "__main__":
    main()

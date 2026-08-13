"""Emit a training-pool JSONL (load_training_examples schema) from a window manifest.

WHY THIS EXISTS
`semsup_train.py` defines its training pool by whichever CAPTION file
`--captions-path` points at - not by a manifest. That is correct for Stage B, whose
whole point is the captions. But it silently caps the crash-only control arm (A1) at
the captioned subset too, even though A1 never reads a caption: with
`--semantic-weight 0.0` no Predictor is constructed and the `caption` field is dead
weight.

So A1 has always trained on 1,761 windows when 4,446 exist. This script closes that
gap by wrapping any manifest into the caption schema with a placeholder caption.

    # full 4,446-window pool for the crash-only arm
    python student_training/scripts/build_pool_from_manifest.py \
        --manifest dataset/manifests/train4500_hires.jsonl \
        --out outputs/semantic_captions/Pool_Train4500_Full_4446.jsonl

IMPORTANT - the placeholder caption is a tripwire, not a caption. Training a real
Stage-B run against this file would align every clip to one identical string, which
is the exact degenerate target the semantic branch was redesigned to avoid. The
placeholder text says so, and `semsup_train.py` should refuse it when
`--semantic-weight > 0` (see the guard note printed at the end of this script).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

PLACEHOLDER = "PLACEHOLDER-NOT-A-CAPTION crash-only pool, semantic branch must be disabled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="window manifest in the Stage-A scorer schema "
                         "(video_id / event_occurs / group / frames_dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-field", default="event_occurs")
    args = ap.parse_args()

    # Relative paths resolve against the CURRENT DIRECTORY, matching every other
    # script in this repo (semsup_train.py's own --test-manifest examples use
    # ../../dataset/... run from student_training/scripts/). Re-anchoring to REPO
    # here would silently break exactly that convention.
    man_path = Path(args.manifest)
    rows = [json.loads(l) for l in open(man_path, encoding="utf-8") if l.strip()]

    out_rows, skipped = [], 0
    seen = set()
    for r in rows:
        fd = r.get("frames_dir")
        if not fd:
            skipped += 1
            continue
        # frames_dir, NOT video_id: one video appears at several TTE/offset buckets,
        # so video_id is not a unique row key for this data (this exact assumption
        # has silently dropped ~35-40% of rows twice in this project).
        if fd in seen:
            skipped += 1
            continue
        seen.add(fd)
        label = int(r[args.gt_field])
        out_rows.append({
            "video_id": r["video_id"],
            "frames_dir": fd,
            "requested_time_to_event": r.get("group"),
            "gt_verdict": "YES" if label else "NO",
            "caption": PLACEHOLDER,
            "horizon_label": r.get("group"),
        })

    out_path = Path(args.out)      # same cwd-relative convention as man_path above
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lab = Counter(r["gt_verdict"] for r in out_rows)
    vids = {r["video_id"] for r in out_rows}
    print(f"[pool] {out_path}")
    print(f"  rows           : {len(out_rows)}  ({skipped} skipped: no frames_dir / duplicate)")
    print(f"  label balance  : YES={lab['YES']}  NO={lab['NO']}")
    print(f"  distinct clips : {len(vids)}")
    print(f"  frames_dir uniq: {len(seen)}  (== rows: {len(seen) == len(out_rows)})")
    print()
    print("  NOTE: captions are placeholders. Use with --semantic-weight 0.0 only.")


if __name__ == "__main__":
    main()

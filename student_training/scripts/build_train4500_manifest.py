"""
build_train4500_manifest.py
=============================
Emits the full 4,500-window train manifest (750 positives x 3 TTE horizons +
750 negatives x 3 midpoint offsets) in the Stage-A scorer schema, so
e4_stageA_badas_open_eval.py can score it unchanged. Nothing is extracted or
trained here - this is the manifest-planning step only (deliverable 1 of the
train4500-inference plan).

Window definition matches semsup_extract_promptbakeoff_frames.py exactly (same
STRIDE/WINDOW/T_FLOOR, same bucket labels/suffixes) so the frames_dir naming
this manifest expects is produced by that same extractor, just pointed at the
full 750/750 pool instead of its 250/250 excluded-pool sample.

Schema per row (matches test_manifest_hires.jsonl byte-for-byte on shared
keys, so evaluate_metrics.py / compute_group_metrics work unchanged):
    video_id, event_occurs, group, time_before_event_s, frame_indices [1..16],
    frames_dir, horizon_label, t_seconds, window_size, stride

`event_occurs` is the scorer's configured gt_field (student_training/configs/
e4_stageA.yaml: data.gt_field) - deliberately NOT named `target`, since
build_frames_dir_index()/load_training_examples() in semsup_common.py are a
different (caption-training) pipeline this manifest does not feed.

GROUP FIELD DESIGN NOTE (read before trusting compute_group_metrics' printed
labels on this manifest): for positives, group 0/1/2 <-> TTE 0.5/1.0/1.5s is
literal - the window really does end that many seconds before the real
time_of_event. For negatives there is no event, so group 0/1/2 is assigned by
matching bucket INDEX only (MID->0, MID-4->1, MID-8->2), mirroring how
test_manifest_hires.jsonl already assigns group 0/1/2 to its negatives (in
proportion to positives, not by real seconds-before-event - confirmed by
inspection: 677 distinct videos, one window each, negatives carry the same
group/time_before_event_s field shape as positives with no temporal meaning).
evaluate_metrics.compute_group_metrics() hardcodes the printed label "Xs
before event" per group regardless of class - that label is misleading for
the negative rows within a group and should be read as "bucket 0/1/2", not a
real time. mine_train_failures.py reports negatives by their own
`horizon_label` (MID/MID-4/MID-8), never by the group's cosmetic name, to
avoid this bleeding into the failure taxonomy.

Contamination guard: fails loudly (nonzero exit) if any planned video_id
overlaps test_manifest_hires.jsonl or val_e3a.jsonl - never silently drops
rows to "fix" an overlap.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "dataset" / "train.csv"
TEST_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "test_manifest_hires.jsonl"
VAL_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "val_e3a.jsonl"
OUT_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "train4500_hires.jsonl"

# Identical to semsup_extract_promptbakeoff_frames.py - the extractor this
# manifest is written to be consumed by.
WINDOW = 16
STRIDE = 4
T_FLOOR = 2.0

# (param, horizon_label, dir_suffix, group_index)
POS_BUCKETS = [(0.5, "TTE_0.5", "tte05", 0), (1.0, "TTE_1.0", "tte10", 1), (1.5, "TTE_1.5", "tte15", 2)]
NEG_BUCKETS = [(0.0, "MID", "mid0", 0), (-4.0, "MID-4", "neg4", 1), (-8.0, "MID-8", "neg8", 2)]


def load_train_labels() -> dict:
    out = {}
    with open(TRAIN_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vid = f"{int(r['id']):05d}"
            out[vid] = {"target": int(r["target"]),
                        "time_of_event": float(r["time_of_event"]) if r["time_of_event"] else None}
    return out


def load_video_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["video_id"])
    return out


def build_rows(labels: dict, excluded: set) -> list:
    rows = []
    pos_ids = sorted(vid for vid, d in labels.items()
                      if d["target"] == 1 and d["time_of_event"] is not None and vid not in excluded)
    neg_ids = sorted(vid for vid, d in labels.items() if d["target"] == 0 and vid not in excluded)

    for vid in pos_ids:
        t_event = labels[vid]["time_of_event"]
        for h, label, suffix, grp in POS_BUCKETS:
            t_new = max(T_FLOOR, t_event - h)
            rows.append({
                "video_id": vid,
                "event_occurs": 1,
                "group": grp,
                "time_before_event_s": h,
                "frame_indices": list(range(1, WINDOW + 1)),
                "frames_dir": f"{vid}_hires_{suffix}",
                "horizon_label": label,
                "t_seconds": round(t_new, 3),
                "window_size": WINDOW,
                "stride": STRIDE,
            })

    for vid in neg_ids:
        for off, label, suffix, grp in NEG_BUCKETS:
            rows.append({
                "video_id": vid,
                "event_occurs": 0,
                "group": grp,
                "time_before_event_s": None,
                "frame_indices": list(range(1, WINDOW + 1)),
                "frames_dir": f"{vid}_hires_{suffix}",
                "horizon_label": label,
                "t_seconds": None,  # depends on each video's own midpoint; filled by the extractor
                "window_size": WINDOW,
                "stride": STRIDE,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print preflight only, write nothing")
    ap.add_argument("--out", default=str(OUT_MANIFEST))
    args = ap.parse_args()

    labels = load_train_labels()
    n_pos_avail = sum(1 for d in labels.values() if d["target"] == 1 and d["time_of_event"] is not None)
    n_neg_avail = sum(1 for d in labels.values() if d["target"] == 0)
    print(f"train.csv: {len(labels)} labeled videos ({n_pos_avail} usable pos, {n_neg_avail} neg)")

    # Exclude val_e3a's 18 clips upstream (they are the checkpoint-selection val
    # set per e4_stageC.yaml, drawn FROM train.csv) - mirrors
    # semsup_extract_promptbakeoff_frames.py's load_excluded_video_ids(). The
    # guard below then re-checks by construction, not as the primary defense.
    test_ids = load_video_ids(TEST_MANIFEST)
    val_ids = load_video_ids(VAL_MANIFEST)
    excluded = test_ids | val_ids
    n_excluded_pos = sum(1 for vid, d in labels.items()
                          if d["target"] == 1 and d["time_of_event"] is not None and vid in excluded)
    n_excluded_neg = sum(1 for vid, d in labels.items() if d["target"] == 0 and vid in excluded)
    print(f"Excluding {len(excluded)} video_ids (test={len(test_ids)}, val={len(val_ids)}): "
          f"{n_excluded_pos} pos / {n_excluded_neg} neg removed from the candidate pool.")

    rows = build_rows(labels, excluded)
    n_pos_rows = sum(1 for r in rows if r["event_occurs"] == 1)
    n_neg_rows = sum(1 for r in rows if r["event_occurs"] == 0)
    print(f"Planned: {len(rows)} rows ({n_pos_rows} pos, {n_neg_rows} neg)")
    print("  bucket counts:", dict(Counter(r["horizon_label"] for r in rows)))

    # Contamination guard (defense in depth - should always pass now that
    # excluded videos never enter build_rows candidate pools).
    planned_ids = {r["video_id"] for r in rows}
    overlap_test = planned_ids & test_ids
    overlap_val = planned_ids & val_ids
    if overlap_test or overlap_val:
        print(f"CONTAMINATION (should be unreachable): {len(overlap_test)} video_ids also in "
              f"test_manifest_hires.jsonl, {len(overlap_val)} also in val_e3a.jsonl", file=sys.stderr)
        sys.exit(1)
    print(f"Contamination guard: 0 overlap with test ({len(test_ids)} ids) or val ({len(val_ids)} ids).")

    # How many frames_dirs already exist on disk (informs extraction time estimate).
    dst_root = PROJECT_ROOT / "dataset" / "train"
    n_exist = sum(1 for r in rows if (dst_root / r["frames_dir"]).exists()
                  and len(list((dst_root / r["frames_dir"]).glob("frame_*.jpg"))) == WINDOW)
    print(f"Already extracted (16/16 frames present): {n_exist}/{len(rows)}")

    if args.dry_run:
        print("[dry-run] stopping before write.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

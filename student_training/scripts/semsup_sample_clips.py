"""
semsup_sample_clips.py
=======================
Preflight + sampler for the prompt-bake-off manifest: a balanced, distinct-video
clip set for captioning with the new prompt (see PLAN: prompt-bakeoff-harness,
2026-07-27).

Key fact this script encodes (discovered during preflight, not assumed): positive
and negative clips use DIFFERENT windowing conventions in this dataset. Positives
are pre-extracted at TTE_0.5 / TTE_1.0 / TTE_1.5 (seconds before the real event).
Negatives have no event to count down to, so they are pre-extracted at MID /
MID-4 / MID-8 (offsets from the clip midpoint) instead - this is the SAME
convention the existing 267-caption set (teacher_dataset_e3b.jsonl) already uses.
"square TTE buckets" therefore means "3 buckets per class", not one shared axis.

Every (video_id, bucket) pair used here must already exist as a real 16-frame
extraction on disk - this script does NOT extract new frames (no raw video in
this repo). It only discovers, filters, and balances what has already been
extracted, across ALL files under dataset/teacher_labels/ (not just the e3b
default semsup_common.build_frames_dir_index() uses), because the videos needed
for a NEW distinct-video set are very likely to live in other teacher_labels
generations rather than the current 267-caption index.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = PROJECT_ROOT / "dataset" / "train.csv"
TEACHER_LABELS_DIR = PROJECT_ROOT / "dataset" / "teacher_labels"
TRAIN_FRAMES_ROOT = PROJECT_ROOT / "dataset" / "train"
INCUMBENT_CAPTIONS = PROJECT_ROOT / "outputs" / "semantic_captions" / "Caption_Train_All_Clips.jsonl"
TEST_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "test_manifest_hires.jsonl"
VAL_E3A_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "val_e3a.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "manifests" / "semsup_promptbakeoff.jsonl"

POS_BUCKETS = ("TTE_0.5", "TTE_1.0", "TTE_1.5")
NEG_BUCKETS = ("MID", "MID-4", "MID-8")


def load_train_labels() -> dict:
    """video_id (zero-padded 5-digit) -> 0/1, from the authoritative train.csv
    (covers all 1500 train videos; teacher_labels files only cover subsets used
    by specific past experiments)."""
    out = {}
    with open(TRAIN_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vid = f"{int(r['id']):05d}"
            out[vid] = int(r["target"])
    return out


def _classify_bucket(tte, horizon_label) -> str | None:
    """Prefer the row's own horizon_label (clean in teacher_dataset_e3b.jsonl);
    fall back to inferring from the raw tte value for files that lack it.
    Returns None for conventions this script doesn't know how to place
    (offset variants like '-5.0_offset' seen in a few older generations)."""
    if horizon_label:
        hl = str(horizon_label).strip().upper()
        if hl in POS_BUCKETS or hl in NEG_BUCKETS:
            return hl
    try:
        t = float(tte)
        for b, target in zip(POS_BUCKETS, (0.5, 1.0, 1.5)):
            if abs(t - target) < 0.01:
                return b
        return None
    except (TypeError, ValueError):
        s = str(tte).upper()
        if "MID" in s:
            if "-8" in s:
                return "MID-8"
            if "-4" in s:
                return "MID-4"
            if "MIDPOINT" in s or s == "MID":
                return "MID"
        return None


def _frames_present(frames_dir: str) -> bool:
    d = TRAIN_FRAMES_ROOT / frames_dir
    if not d.is_dir():
        return False
    return all((d / f"frame_{i:05d}.jpg").exists() for i in range(1, 17))


def discover_pool(labels: dict) -> list:
    """Scan every dataset/teacher_labels/*.jsonl file (tolerant of cross-file
    duplication - first-seen frames_dir wins per (video_id, bucket), since this
    is pool discovery for sampling, not the strict training-time index) and
    return usable (video_id, bucket, tte, frames_dir, label) candidates: frames
    physically present on disk, label known from train.csv, bucket recognized."""
    seen = {}
    for fp in sorted(glob.glob(str(TEACHER_LABELS_DIR / "*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid, fd = r.get("video_id"), r.get("frames_dir")
                if not vid or not fd:
                    continue
                bucket = _classify_bucket(r.get("requested_time_to_event"), r.get("horizon_label"))
                if bucket is None:
                    continue
                key = (vid, bucket)
                if key in seen:
                    continue  # first-seen wins; not asserting cross-file agreement here
                seen[key] = (r.get("requested_time_to_event"), fd)

    pool = []
    for (vid, bucket), (tte, fd) in seen.items():
        if vid not in labels:
            continue
        if not _frames_present(fd):
            continue
        pool.append({"video_id": vid, "bucket": bucket, "tte": tte,
                      "frames_dir": fd, "label": labels[vid]})
    return pool


def preflight_report(pool: list, excluded_ids: set) -> dict:
    by_label_bucket = Counter((r["label"], r["bucket"]) for r in pool)
    vids_by_label_bucket = defaultdict(set)
    for r in pool:
        vids_by_label_bucket[(r["label"], r["bucket"])].add(r["video_id"])

    distinct_videos = {r["video_id"] for r in pool}
    by_class_vids = defaultdict(set)
    for r in pool:
        by_class_vids[r["label"]].add(r["video_id"])

    print("=" * 70)
    print("PREFLIGHT: prompt-bakeoff clip pool")
    print("=" * 70)
    print(f"Candidate rows (frames present + labeled + recognized bucket): {len(pool)}")
    print(f"Distinct candidate videos: {len(distinct_videos)}")
    print(f"  positive (target=1): {len(by_class_vids.get(1, set()))} videos")
    print(f"  negative (target=0): {len(by_class_vids.get(0, set()))} videos")
    print(f"Excluded (test/val contamination guard): {len(excluded_ids)} video_ids removed from pool")
    print()
    print("Distinct videos per (class, bucket):")
    for label in (1, 0):
        buckets = POS_BUCKETS if label == 1 else NEG_BUCKETS
        for b in buckets:
            print(f"  label={label} bucket={b:8s} -> {len(vids_by_label_bucket.get((label, b), set()))} videos")

    max_balanced = 2 * min(len(by_class_vids.get(1, set())), len(by_class_vids.get(0, set())))
    print()
    print(f"MAX ACHIEVABLE BALANCED distinct-video set (one row/video, 1:1 class ratio): {max_balanced} rows")
    print("=" * 70)
    return {"max_balanced": max_balanced, "n_pos_videos": len(by_class_vids.get(1, set())),
            "n_neg_videos": len(by_class_vids.get(0, set()))}


def load_excluded_ids() -> set:
    """video_ids that must never enter the training sample - the 677-clip test
    set and the 18-clip val_e3a set. This is deliberately mechanical (not just
    careful sampling) so that whatever changes later, the guard still holds."""
    excluded = set()
    for fp in (TEST_MANIFEST, VAL_E3A_MANIFEST):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                vid = r.get("video_id")
                if vid:
                    excluded.add(vid)
    return excluded


def load_incumbent_video_ids() -> set:
    if not INCUMBENT_CAPTIONS.exists():
        return set()
    out = set()
    with open(INCUMBENT_CAPTIONS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.add(json.loads(line)["video_id"])
    return out


def sample_balanced(pool: list, n: int, seed: int, prefer_new: set) -> list:
    """One row per video_id, split evenly across the 3 buckets for each class,
    preferring videos NOT already in the incumbent 267-caption set."""
    rng = random.Random(seed)
    n_per_class = n // 2
    n_per_bucket = n_per_class // 3  # e.g. 300 -> 150 -> 50

    by_video = defaultdict(list)  # video_id -> list of candidate rows (one per bucket it has)
    for r in pool:
        by_video[r["video_id"]].append(r)

    selected = []
    for label, buckets in ((1, POS_BUCKETS), (0, NEG_BUCKETS)):
        videos = [v for v, rows in by_video.items() if rows[0]["label"] == label]
        # prefer videos not already captioned in the incumbent set
        videos.sort(key=lambda v: (v in prefer_new is False, v))  # new videos first (stable)
        videos_new = [v for v in videos if v in prefer_new]
        videos_old = [v for v in videos if v not in prefer_new]
        rng.shuffle(videos_new)
        rng.shuffle(videos_old)
        ordered_videos = videos_new + videos_old  # exhaust new pool before touching incumbent videos

        bucket_counts = {b: 0 for b in buckets}
        used_videos = set()
        for vid in ordered_videos:
            if sum(bucket_counts.values()) >= n_per_class:
                break
            options = {r["bucket"]: r for r in by_video[vid]}
            # pick the bucket this video can fill that currently has the fewest rows
            candidate_buckets = [b for b in buckets if b in options and bucket_counts[b] < n_per_bucket]
            if not candidate_buckets:
                continue
            best_bucket = min(candidate_buckets, key=lambda b: bucket_counts[b])
            selected.append(options[best_bucket])
            bucket_counts[best_bucket] += 1
            used_videos.add(vid)
        print(f"  class={label}: filled {dict(bucket_counts)} "
              f"({sum(bucket_counts.values())}/{n_per_class} rows, "
              f"{len(used_videos)} distinct videos, "
              f"{len(used_videos & prefer_new)} from new-video pool)")

    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300, help="target total rows (split 50/50 by class)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="preflight only, do not write the manifest")
    ap.add_argument("--allow-partial", action="store_true",
                     help="write the manifest even if the achievable n is below --n "
                          "(default: refuse and exit non-zero, per plan R3 lesson)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    labels = load_train_labels()
    pool = discover_pool(labels)
    excluded_ids = load_excluded_ids()
    n_before = len(pool)
    pool = [r for r in pool if r["video_id"] not in excluded_ids]
    n_removed = n_before - len(pool)

    stats = preflight_report(pool, excluded_ids)
    if n_removed:
        print(f"NOTE: {n_removed} candidate rows removed by the test/val contamination guard.")

    if stats["max_balanced"] < args.n and not args.allow_partial:
        print()
        print(f"REFUSING: requested --n {args.n} but only {stats['max_balanced']} rows are achievable "
              f"({stats['n_pos_videos']} pos videos, {stats['n_neg_videos']} neg videos). "
              f"Re-run with --allow-partial to sample the achievable amount instead, "
              f"or extend the candidate pool (more extracted frame windows).")
        raise SystemExit(1)

    if args.dry_run:
        print("\n[dry-run] stopping before sampling/writing.")
        return

    incumbent_ids = load_incumbent_video_ids()
    target_n = min(args.n, stats["max_balanced"]) if args.allow_partial else args.n
    print(f"\nSampling {target_n} rows (seed={args.seed}), preferring videos not in the "
          f"incumbent {len(incumbent_ids)}-video caption set...")
    selected = sample_balanced(pool, target_n, args.seed, prefer_new=set(labels) - incumbent_ids)

    # final defensive check - never trust the earlier filter alone
    sampled_ids = {r["video_id"] for r in selected}
    overlap = sampled_ids & excluded_ids
    if overlap:
        raise RuntimeError(f"contamination guard violated post-sampling: {overlap}")
    overlap_incumbent = sampled_ids & incumbent_ids
    print(f"Sampled {len(selected)} rows, {len(sampled_ids)} distinct videos, "
          f"{len(overlap_incumbent)} overlap the incumbent 267-caption set.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected:
            rec = {
                "video_id": r["video_id"],
                "frames_dir": r["frames_dir"],
                "frame_indices": list(range(1, 17)),
                "window_size": 16,
                "target": r["label"],
                "requested_time_to_event": r["tte"],
                "horizon_label": r["bucket"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} rows -> {out_path}")


if __name__ == "__main__":
    main()

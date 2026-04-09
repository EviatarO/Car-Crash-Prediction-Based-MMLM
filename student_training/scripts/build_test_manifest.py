"""
build_test_manifest.py
======================
Builds a JSONL manifest of test clips for zero-shot evaluation.

INPUT:
  - test.csv       : Nexar test labels (id, event_occurs, Usage, group)
  - test_frames_metadata.csv : per-video fps, total_frames, duration

OUTPUT:
  - test_manifest_private.jsonl : one record per Private video

CLIP DEFINITION:
  Nexar pre-cut each video to end at:
    group=0 → 0.5s before event/non-event
    group=1 → 1.0s before event/non-event
    group=2 → 1.5s before event/non-event
  So we always take the LAST 16 frames at stride=4 (last ~2 seconds of video).
  Frame indices: [N-1-60, N-1-56, ..., N-1]  where N = total_frames

GROUND TRUTH:
  event_occurs=1 → collision (TP candidate)
  event_occurs=0 → no collision (TN candidate)

Usage:
  python student_training/scripts/build_test_manifest.py \
    --test_csv      <path>/test.csv \
    --metadata_csv  <path>/test_frames_metadata.csv \
    --frames_root   <path>/test_frames256 \
    --output        outputs/test_manifest_private.jsonl \
    [--usage        Private]        # default: Private
    [--window_size  16]             # default: 16
    [--stride       4]              # default: 4
"""

import argparse
import json
import os
import sys

import pandas as pd


GROUP_TO_SECONDS = {0: 0.5, 1: 1.0, 2: 1.5}


def build_frame_indices(total_frames: int, window_size: int, stride: int) -> list[int]:
    """
    Return frame indices for the last window of a video.

    Takes the last `window_size` frames stepping back by `stride` from the
    final frame. E.g. total_frames=303, window_size=16, stride=4:
      last_frame = 302
      indices = [242, 246, 250, 254, 258, 262, 266, 270, 274, 278, 282, 286, 290, 294, 298, 302]
    """
    last_idx = total_frames - 1
    indices = [last_idx - (window_size - 1 - i) * stride for i in range(window_size)]
    # Clamp to valid range (in case video is shorter than expected)
    indices = [max(0, idx) for idx in indices]
    return indices


def verify_frames_exist(frames_root: str, video_id: str, indices: list[int],
                         pattern: str = "frame_{:05d}.jpg") -> tuple[bool, list[str]]:
    """Check that all required frame files exist on disk."""
    missing = []
    for idx in indices:
        path = os.path.join(frames_root, video_id, pattern.format(idx))
        if not os.path.exists(path):
            missing.append(path)
    return len(missing) == 0, missing


def main():
    parser = argparse.ArgumentParser(description="Build test clip manifest for zero-shot evaluation")
    parser.add_argument("--test_csv",     required=True, help="Path to test.csv")
    parser.add_argument("--metadata_csv", required=True, help="Path to test_frames_metadata.csv")
    parser.add_argument("--frames_root",  required=True, help="Root directory containing test_frames256/")
    parser.add_argument("--output",       required=True, help="Output JSONL path")
    parser.add_argument("--usage",        default="Private", help="Filter by Usage column (default: Private)")
    parser.add_argument("--window_size",  type=int, default=16)
    parser.add_argument("--stride",       type=int, default=4)
    parser.add_argument("--no_verify",    action="store_true", help="Skip frame file existence check")
    args = parser.parse_args()

    # ---- Load CSVs ----
    print(f"Loading test.csv from: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    # Strip trailing comma/whitespace from column names if present
    test_df.columns = [c.strip().rstrip(',') for c in test_df.columns]

    print(f"Loading metadata from: {args.metadata_csv}")
    meta_df = pd.read_csv(args.metadata_csv)
    meta_df.columns = [c.strip() for c in meta_df.columns]

    # Normalise video_id to zero-padded 5-digit string in both dataframes
    test_df['video_id'] = test_df['id'].apply(lambda x: f"{int(x):05d}")
    meta_df['video_id'] = meta_df['video_id'].apply(lambda x: f"{int(str(x).strip()):05d}")
    meta_lookup = meta_df.set_index('video_id').to_dict('index')

    # ---- Filter by Usage ----
    filtered = test_df[test_df['Usage'] == args.usage].copy()
    print(f"\nTotal videos in test.csv     : {len(test_df)}")
    print(f"Filtered (Usage='{args.usage}'): {len(filtered)}")

    # ---- Summary ----
    print("\nBreakdown (group × event_occurs):")
    print(filtered.groupby(['group', 'event_occurs']).size().unstack(fill_value=0).to_string())

    # ---- Build manifest ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    records = []
    skipped = 0

    for _, row in filtered.iterrows():
        vid = row['video_id']
        event_occurs = int(row['event_occurs'])
        group = int(row['group'])
        time_before_event = GROUP_TO_SECONDS.get(group, None)

        # Get frame count from metadata
        if vid not in meta_lookup:
            print(f"  WARNING: {vid} not in metadata, skipping")
            skipped += 1
            continue

        total_frames = int(meta_lookup[vid]['total_frames'])
        fps = float(meta_lookup[vid]['fps'])
        duration = float(meta_lookup[vid]['duration'])

        # Compute frame indices (last window)
        frame_indices = build_frame_indices(total_frames, args.window_size, args.stride)

        # Optionally verify files exist
        if not args.no_verify:
            ok, missing = verify_frames_exist(args.frames_root, vid, frame_indices)
            if not ok:
                print(f"  WARNING: {vid} missing {len(missing)} frames, skipping")
                skipped += 1
                continue

        record = {
            "video_id":            vid,
            "event_occurs":        event_occurs,      # 0=no collision, 1=collision
            "group":               group,             # 0=0.5s, 1=1.0s, 2=1.5s before event
            "time_before_event_s": time_before_event, # seconds before event video ends
            "total_frames":        total_frames,
            "fps":                 fps,
            "duration_s":          duration,
            "window_size":         args.window_size,
            "stride":              args.stride,
            "frame_indices":       frame_indices,     # 16 indices into frames folder
        }
        records.append(record)

    # ---- Write output ----
    with open(args.output, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"\n{'='*50}")
    print(f"Manifest saved  : {args.output}")
    print(f"Total records   : {len(records)}")
    print(f"Skipped         : {skipped}")
    print(f"  Collisions (event_occurs=1): {sum(1 for r in records if r['event_occurs']==1)}")
    print(f"  Safe      (event_occurs=0): {sum(1 for r in records if r['event_occurs']==0)}")
    print(f"\nGroup breakdown:")
    for g, s in GROUP_TO_SECONDS.items():
        n = sum(1 for r in records if r['group'] == g)
        print(f"  group={g} ({s}s before event): {n} clips")


if __name__ == "__main__":
    main()

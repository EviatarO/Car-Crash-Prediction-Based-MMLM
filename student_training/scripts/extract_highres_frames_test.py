"""Extract NATIVE-resolution (1280x720) frames for the 677-clip private test set.

Reads outputs/test_manifest_private.jsonl (677 records), each containing
video_id and frame_indices (stride-based positions in the source MP4).

For each clip, opens Nexar_DataSet/test/{vid}.mp4 with OpenCV, seeks to
each frame_index, and writes the frame at native resolution into
    dataset/test/{vid}_hires/frame_00001.jpg ... frame_00016.jpg
(sequential naming, matching the training HiRes convention).

Also generates outputs/test_manifest_hires.jsonl with:
  - frames_dir: "{vid}_hires"
  - frame_indices: [1, 2, ..., 16]
so that trained_eval.py can load them directly.

Idempotent: skips folders that already contain 16 frames.
Missing MP4s are logged to outputs/test_hires_extraction_failures.json.

Usage:
    python student_training/scripts/extract_highres_frames_test.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]

# Defaults reproduce the original Private-set behaviour; override via CLI for Public.
SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\test"
)
DST_ROOT = REPO_ROOT / "dataset" / "test"
MANIFEST_IN = REPO_ROOT / "outputs" / "test_manifest_private.jsonl"
MANIFEST_OUT = REPO_ROOT / "outputs" / "test_manifest_hires.jsonl"
FAILURES_JSON = REPO_ROOT / "outputs" / "test_hires_extraction_failures.json"

WINDOW = 16


def _extract_one(vid: str, frame_indices: list[int]) -> tuple[int, int, int]:
    """Extract frames at the given indices from the MP4, save with sequential naming.

    Returns (width, height, n_written). Raises FileNotFoundError if MP4 missing.
    """
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4}")
    out_dir = DST_ROOT / f"{vid}_hires"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_written = 0
    for seq_i, fr_idx in enumerate(frame_indices, start=1):
        dst = out_dir / f"frame_{seq_i:05d}.jpg"
        if dst.exists():
            n_written += 1
            continue
        # Clamp index to valid range
        clamped_idx = max(0, min(total - 1, fr_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {fr_idx} from {mp4}")
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_written += 1
    cap.release()
    return width, height, n_written


def main() -> None:
    global SRC_VIDEOS, DST_ROOT, MANIFEST_IN, MANIFEST_OUT, FAILURES_JSON
    ap = argparse.ArgumentParser(description="Extract native-res frames for a test manifest")
    ap.add_argument("--manifest_in", default=str(MANIFEST_IN))
    ap.add_argument("--dst_root", default=str(DST_ROOT))
    ap.add_argument("--manifest_out", default=str(MANIFEST_OUT))
    ap.add_argument("--failures_json", default=str(FAILURES_JSON))
    ap.add_argument("--src_videos", default=str(SRC_VIDEOS))
    args = ap.parse_args()
    SRC_VIDEOS = Path(args.src_videos)
    DST_ROOT = Path(args.dst_root)
    MANIFEST_IN = Path(args.manifest_in)
    MANIFEST_OUT = Path(args.manifest_out)
    FAILURES_JSON = Path(args.failures_json)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_IN.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST_IN}")
        sys.exit(1)
    if not SRC_VIDEOS.exists():
        print(f"[ERROR] Test MP4 directory not found: {SRC_VIDEOS}")
        sys.exit(1)

    # Load manifest
    with open(MANIFEST_IN, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Source MP4 root : {SRC_VIDEOS}")
    print(f"Destination     : {DST_ROOT}")
    print(f"Manifest        : {MANIFEST_IN}  ({len(records)} clips)")
    print()

    failures: list[dict] = []
    n_ok_existing = 0
    n_new = 0
    hires_records: list[dict] = []

    for i, rec in enumerate(records):
        vid = rec["video_id"]
        frame_indices = rec["frame_indices"]

        # Check if already extracted
        dst_dir = DST_ROOT / f"{vid}_hires"
        n_existing = len(list(dst_dir.glob("frame_*.jpg"))) if dst_dir.exists() else 0
        if n_existing == WINDOW:
            n_ok_existing += 1
            # Still add to HiRes manifest
            hires_rec = dict(rec)
            hires_rec["frames_dir"] = f"{vid}_hires"
            hires_rec["frame_indices"] = list(range(1, WINDOW + 1))
            hires_records.append(hires_rec)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(records)}] {vid}_hires: already present")
            continue

        try:
            w, h, n = _extract_one(vid, frame_indices)
            n_new += 1
            # Add to HiRes manifest
            hires_rec = dict(rec)
            hires_rec["frames_dir"] = f"{vid}_hires"
            hires_rec["frame_indices"] = list(range(1, WINDOW + 1))
            hires_records.append(hires_rec)
            if (i + 1) % 50 == 0 or n_new <= 5:
                print(f"  [{i+1}/{len(records)}] {vid}_hires: {n} frames @ {w}x{h}")
        except FileNotFoundError as e:
            print(f"  [MISS] {vid}: {e}")
            failures.append({"video_id": vid, "reason": "MP4 not found",
                             "path": str(SRC_VIDEOS / f"{vid}.mp4")})
        except Exception as e:
            print(f"  [ERR]  {vid}: {type(e).__name__}: {e}")
            failures.append({"video_id": vid, "reason": f"{type(e).__name__}: {e}"})

    # Write HiRes manifest
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        for rec in hires_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nHiRes manifest written: {MANIFEST_OUT}  ({len(hires_records)} records)")

    # Write failures
    FAILURES_JSON.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    # Summary
    print()
    print("=" * 60)
    print(f"Summary: {n_ok_existing} already had frames, {n_new} newly extracted, "
          f"{len(failures)} failed/skipped.")
    print(f"Total in HiRes manifest: {len(hires_records)}")
    if failures:
        print(f"Failures written to: {FAILURES_JSON}")
        for f_entry in failures:
            print(f"  - {f_entry['video_id']}: {f_entry['reason']}")
    print("=" * 60)

    # Spot check
    if hires_records:
        first = hires_records[0]
        print(f"\nSpot check (first record):")
        print(f"  video_id     : {first['video_id']}")
        print(f"  frames_dir   : {first['frames_dir']}")
        print(f"  frame_indices: {first['frame_indices']}")
        print(f"  event_occurs : {first.get('event_occurs')}")


if __name__ == "__main__":
    main()

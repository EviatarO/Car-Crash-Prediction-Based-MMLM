"""
extract_clip_frames.py
----------------------
Reads train + test manifests and copies ONLY the 16 needed frames per clip
into a compact output folder ready for upload to RunPod.

Output structure:
    outputs/extracted_clips/
        train/
            {video_id}/
                frame_{:05d}.jpg   (16 files)
        test/
            {video_id}/
                frame_{:05d}.jpg   (16 files)

Usage:
    python student_training/scripts/extract_clip_frames.py \
        --train_frames  "C:/path/to/train_frames256" \
        --test_frames   "C:/path/to/test_frames256" \
        --out_dir       outputs/extracted_clips
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults — edit here or override via CLI flags
# ---------------------------------------------------------------------------
DEFAULT_TRAIN_FRAMES = (
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train_frames256"
)
DEFAULT_TEST_FRAMES = (
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\test_frames256"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "extracted_clips"

TRAIN_MANIFEST = PROJECT_ROOT / "outputs" / "manifest_v11_100clips.jsonl"
TEST_MANIFEST  = PROJECT_ROOT / "outputs" / "test_manifest_private.jsonl"

FRAME_PATTERN = "frame_{:05d}.jpg"


# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_split(records: list[dict], src_root: Path, dst_root: Path, split: str):
    dst_root.mkdir(parents=True, exist_ok=True)
    missing_clips = []
    total_frames_copied = 0

    for rec in records:
        vid = rec["video_id"]
        frame_indices = rec["frame_indices"]

        src_clip_dir = src_root / vid
        dst_clip_dir = dst_root / vid
        dst_clip_dir.mkdir(parents=True, exist_ok=True)

        if not src_clip_dir.exists():
            missing_clips.append(vid)
            print(f"  [WARN] Missing source clip folder: {src_clip_dir}")
            continue

        for idx in frame_indices:
            fname = FRAME_PATTERN.format(idx)
            src_file = src_clip_dir / fname
            dst_file = dst_clip_dir / fname

            if not src_file.exists():
                print(f"  [WARN] Missing frame: {src_file}")
                continue

            shutil.copy2(src_file, dst_file)
            total_frames_copied += 1

    n_ok = len(records) - len(missing_clips)
    print(f"  [{split}] {n_ok}/{len(records)} clips extracted, "
          f"{total_frames_copied} frames copied")
    if missing_clips:
        print(f"  [{split}] Missing clip folders ({len(missing_clips)}): "
              f"{missing_clips[:10]}{'...' if len(missing_clips) > 10 else ''}")

    return missing_clips


def estimate_size(out_dir: Path) -> str:
    total_bytes = sum(f.stat().st_size for f in out_dir.rglob("*.jpg"))
    if total_bytes < 1024 ** 2:
        return f"{total_bytes / 1024:.1f} KB"
    elif total_bytes < 1024 ** 3:
        return f"{total_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{total_bytes / 1024 ** 3:.2f} GB"


def main():
    parser = argparse.ArgumentParser(description="Extract 16 frames per clip for RunPod upload")
    parser.add_argument("--train_frames", default=DEFAULT_TRAIN_FRAMES,
                        help="Root folder of train frame directories")
    parser.add_argument("--test_frames",  default=DEFAULT_TEST_FRAMES,
                        help="Root folder of test frame directories")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR),
                        help="Output directory for extracted clips")
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"],
                        help="Which splits to extract (default: both)")
    args = parser.parse_args()

    out_dir    = Path(args.out_dir)
    train_src  = Path(args.train_frames)
    test_src   = Path(args.test_frames)

    print(f"Output directory : {out_dir}")
    print(f"Train frames src : {train_src}")
    print(f"Test frames src  : {test_src}")
    print()

    if "train" in args.splits:
        print(f"Loading train manifest: {TRAIN_MANIFEST}")
        train_records = load_manifest(TRAIN_MANIFEST)
        print(f"  {len(train_records)} clips in train manifest")
        extract_split(train_records, train_src, out_dir / "train", "train")
        print()

    if "test" in args.splits:
        print(f"Loading test manifest : {TEST_MANIFEST}")
        test_records = load_manifest(TEST_MANIFEST)
        print(f"  {len(test_records)} clips in test manifest")
        extract_split(test_records, test_src, out_dir / "test", "test")
        print()

    print(f"Total size of extracted_clips/: {estimate_size(out_dir)}")
    print(f"Done. Ready to zip and upload:")
    print(f"  cd {out_dir.parent}")
    print(f"  tar -czf extracted_clips.tar.gz extracted_clips/")


if __name__ == "__main__":
    main()

"""Extract dense-sampled frame folders for the frame-density sweep.

For each of the 18 GT clips, creates:
- dataset/train/<vid>_32f/  (32 frames, stride=2, covering ~2 seconds)
- dataset/train/<vid>_64f/  (64 frames, stride=1, covering ~2 seconds)

Frames are copied from dataset/train/<vid>/frame_*.jpg and renumbered sequentially
starting at frame_00001.jpg.

Idempotent: skips folders that already have the expected number of frames.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "teacher_distillation" / "scripts"))

from teacher_prompt_bakeoff import _read_gt_excel_with_en  # noqa: E402

FPS = 30
# Source frames: the FULL Nexar extraction (1200+ frames per clip).
# The local dataset/train/<vid>/ only has the 16f window (~61 frames), insufficient for 64f.
SRC_ROOT = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train_frames256"
)
DST_ROOT = REPO_ROOT / "dataset" / "train"
GT_XLSX = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"

# (tag, window, stride)
CONFIGS = [
    ("8f", 8, 8),
    ("32f", 32, 2),
    ("64f", 64, 1),
]


def _frame_indices(t_seconds: float, fps: float, window: int, stride: int) -> list[int]:
    end = round(t_seconds * fps)
    return [end - (window - 1 - i) * stride for i in range(window)]


def _copy_frames(vid: str, indices: list[int], dst_dir: Path) -> None:
    src_dir = SRC_ROOT / vid
    if not src_dir.exists():
        raise FileNotFoundError(f"Source folder missing: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Source frames in train_frames256 are numbered starting from 0 (e.g. frame_00000.jpg)
    for new_idx, src_idx in enumerate(indices, start=1):
        src = src_dir / f"frame_{src_idx:05d}.jpg"
        dst = dst_dir / f"frame_{new_idx:05d}.jpg"
        if not src.exists():
            raise FileNotFoundError(f"Source frame missing: {src}")
        if not dst.exists():
            shutil.copy2(src, dst)


def main() -> None:
    clips = _read_gt_excel_with_en(GT_XLSX)
    print(f"Loaded {len(clips)} clips from {GT_XLSX.name}")
    print(f"Source frames root: {SRC_ROOT}")
    print(f"Destination root:   {DST_ROOT}")
    print()

    created = 0
    skipped = 0
    for clip in clips:
        vid = clip["video_id"]
        t = clip["t_seconds"]
        if t is None:
            print(f"  [SKIP] {vid}: t_seconds is None")
            skipped += 1
            continue
        for tag, window, stride in CONFIGS:
            dst_dir = DST_ROOT / f"{vid}_{tag}"
            n_existing = len(list(dst_dir.glob("frame_*.jpg"))) if dst_dir.exists() else 0
            if n_existing == window:
                print(f"  [OK]   {vid}_{tag} ({window} frames already present)")
                skipped += 1
                continue
            indices = _frame_indices(t, FPS, window, stride)
            _copy_frames(vid, indices, dst_dir)
            n_after = len(list(dst_dir.glob("frame_*.jpg")))
            assert n_after == window, f"{dst_dir}: expected {window}, got {n_after}"
            print(f"  [NEW]  {vid}_{tag}: {window} frames @ indices {indices[0]}..{indices[-1]}")
            created += 1

    print()
    print(f"Created: {created}  Skipped (already present): {skipped}")
    print("Done.")


if __name__ == "__main__":
    main()

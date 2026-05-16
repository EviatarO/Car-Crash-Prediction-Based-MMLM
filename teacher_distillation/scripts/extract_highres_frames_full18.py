"""Extract NATIVE-resolution frames from the Nexar MP4 source for the 12 GT
clips that were NOT included in the original hi-res 6-clip diagnostic.

For each clip in CLIPS_TO_RUN, opens <SRC_VIDEOS>/<vid>.mp4 with OpenCV,
decodes 16 frames at stride=4 ending at t_seconds*fps, writes them to
dataset/train/<vid>_hires/frame_00001.jpg ... frame_00016.jpg at native
resolution (typically 1280x720).

Idempotent: skips folders that already contain 16 frames.
Reuses the same logic as extract_highres_frames.py (different CLIPS list).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "teacher_distillation" / "scripts"))

from teacher_prompt_bakeoff import _read_gt_excel_with_en  # noqa: E402

# The 12 clips NOT already in highres_test.jsonl (18 GT clips minus the 6
# problem clips already extracted at hires by extract_highres_frames.py).
CLIPS_TO_RUN = [
    "00077", "00147", "00283", "00319", "00474", "00493",
    "00687", "01550", "01552", "01643", "02104", "02117",
]

SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train"
)
DST_ROOT = REPO_ROOT / "dataset" / "train"
GT_XLSX = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"

WINDOW = 16
STRIDE = 4


def _extract_one(vid: str, t_seconds: float) -> tuple[int, int, int]:
    """Returns (width, height, n_written)."""
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4}")
    out_dir = DST_ROOT / f"{vid}_hires"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end = round(t_seconds * fps)
    indices = [end - (WINDOW - 1 - i) * STRIDE for i in range(WINDOW)]

    # Clamp into [0, total-1]
    indices = [max(0, min(total - 1, ix)) for ix in indices]

    n_written = 0
    for i, fr_idx in enumerate(indices, start=1):
        dst = out_dir / f"frame_{i:05d}.jpg"
        if dst.exists():
            n_written += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {fr_idx} from {mp4}")
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_written += 1
    cap.release()
    return width, height, n_written


def main() -> None:
    clips = _read_gt_excel_with_en(GT_XLSX)
    t_map = {c["video_id"]: c["t_seconds"] for c in clips}

    print(f"Source MP4 root: {SRC_VIDEOS}")
    print(f"Destination:     {DST_ROOT}")
    print(f"Clips to extract: {CLIPS_TO_RUN}")
    print()

    for vid in CLIPS_TO_RUN:
        if vid not in t_map or t_map[vid] is None:
            print(f"  [SKIP] {vid}: no t_seconds in GT")
            continue
        dst_dir = DST_ROOT / f"{vid}_hires"
        n_existing = len(list(dst_dir.glob("frame_*.jpg"))) if dst_dir.exists() else 0
        if n_existing == WINDOW:
            print(f"  [OK]   {vid}_hires: {WINDOW} frames already present")
            continue
        w, h, n = _extract_one(vid, t_map[vid])
        print(f"  [NEW]  {vid}_hires: {n} frames @ native {w}x{h}  t={t_map[vid]:.2f}s")

    print()
    print("Done.")


if __name__ == "__main__":
    main()

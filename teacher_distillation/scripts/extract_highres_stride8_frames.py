"""Extract NATIVE-resolution (1280x720) 16-frame stride-8 windows for the 18 GT clips.

Anchored to the SAME final frame as the stride-4 extraction:
    last_frame_idx = round(t_seconds * fps)
    first_frame_idx = last - 15 * 8  (~4 seconds back at 30 fps)

Output dir: dataset/train/<vid>_hires_s8/frame_00001.jpg ... frame_00016.jpg
Separate from existing dataset/train/<vid>_hires/ so stride-4 frames stay intact.

For clips with short t_seconds, frame indices below 0 are clamped to 0 (early
frames repeat). This is logged but not skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "teacher_distillation" / "scripts"))

from teacher_prompt_bakeoff import _read_gt_excel_with_en  # noqa: E402

# All 18 GT clips
CLIPS_TO_RUN = [
    "00077", "00147", "00283", "00319", "00372", "00474",
    "00493", "00529", "00687", "01153", "01281", "01504",
    "01550", "01552", "01643", "01737", "02104", "02117",
]

SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train"
)
DST_ROOT = REPO_ROOT / "dataset" / "train"
GT_XLSX = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"

WINDOW = 16
STRIDE = 8       # ⇒ 15*8 = 120 frames ≈ 4 s at 30 fps


def _extract_one(vid: str, t_seconds: float) -> tuple[int, int, int, int]:
    """Returns (width, height, n_written, n_clamped).
    n_clamped = how many indices had to be clamped to frame 0.
    """
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4}")
    out_dir = DST_ROOT / f"{vid}_hires_s8"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end = round(t_seconds * fps)
    raw_indices = [end - (WINDOW - 1 - i) * STRIDE for i in range(WINDOW)]
    n_clamped = sum(1 for ix in raw_indices if ix < 0)
    indices = [max(0, min(total - 1, ix)) for ix in raw_indices]

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
    return width, height, n_written, n_clamped


def main() -> None:
    clips = _read_gt_excel_with_en(GT_XLSX)
    t_map = {c["video_id"]: c["t_seconds"] for c in clips}

    print(f"Source MP4 root: {SRC_VIDEOS}")
    print(f"Destination:     {DST_ROOT}\\<vid>_hires_s8\\")
    print(f"Window: {WINDOW} frames @ stride={STRIDE}  =>  "
          f"{(WINDOW-1)*STRIDE} frames back ~ {(WINDOW-1)*STRIDE/30:.1f}s history")
    print(f"Clips: {CLIPS_TO_RUN}\n")

    total_clamped_clips = 0
    for vid in CLIPS_TO_RUN:
        if vid not in t_map or t_map[vid] is None:
            print(f"  [SKIP] {vid}: no t_seconds in GT")
            continue
        dst_dir = DST_ROOT / f"{vid}_hires_s8"
        n_existing = len(list(dst_dir.glob("frame_*.jpg"))) if dst_dir.exists() else 0
        if n_existing == WINDOW:
            print(f"  [OK]   {vid}_hires_s8: {WINDOW} frames already present")
            continue
        w, h, n, n_clamped = _extract_one(vid, t_map[vid])
        flag = f"  [CLAMPED {n_clamped} frames to idx 0]" if n_clamped else ""
        print(f"  [NEW]  {vid}_hires_s8: {n} frames @ native {w}x{h}  "
              f"t={t_map[vid]:.2f}s{flag}")
        if n_clamped:
            total_clamped_clips += 1

    print()
    print(f"Done. {total_clamped_clips} clips had clamped (repeated) start frames "
          f"(t_seconds < ~4 s).")


if __name__ == "__main__":
    main()

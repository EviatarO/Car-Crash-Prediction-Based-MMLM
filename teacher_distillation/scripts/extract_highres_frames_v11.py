"""Extract NATIVE-resolution (1280x720) frames for the 100-clip v11 teacher set.

Reads outputs/teacher_dataset_v11.xlsx (100 rows, video_id + t_seconds),
zero-pads video_id to 5 digits, then for each clip opens
<SRC_VIDEOS>/<vid>.mp4 with OpenCV and writes 16 frames at stride=4 ending
at t_seconds*fps into dataset/train/<vid>_hires/frame_00001.jpg ... 00016.jpg.

Idempotent: skips folders that already contain 16 frames.
Missing MP4s are logged to extraction_failures.json (pipeline does NOT abort).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train"
)
DST_ROOT = REPO_ROOT / "dataset" / "train"
V11_XLSX = REPO_ROOT / "outputs" / "teacher_dataset_v11.xlsx"
OUT_DIR = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips"
FAILURES_JSON = OUT_DIR / "extraction_failures.json"

WINDOW = 16
STRIDE = 4


def _extract_one(vid: str, t_seconds: float) -> tuple[int, int, int]:
    """Returns (width, height, n_written). Raises FileNotFoundError if MP4 missing."""
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(V11_XLSX)
    if "video_id" not in df.columns or "t_seconds" not in df.columns:
        raise SystemExit(f"v11 xlsx missing video_id/t_seconds columns. Got: {list(df.columns)}")

    print(f"Source MP4 root: {SRC_VIDEOS}")
    print(f"Destination:     {DST_ROOT}")
    print(f"V11 clips:       {len(df)}")
    print()

    failures: list[dict] = []
    n_ok_existing = 0
    n_new = 0

    for _, row in df.iterrows():
        vid = f"{int(row['video_id']):05d}"
        t_seconds = row["t_seconds"]

        if pd.isna(t_seconds):
            print(f"  [SKIP] {vid}: no t_seconds")
            failures.append({"video_id": vid, "reason": "no t_seconds in v11.xlsx"})
            continue

        dst_dir = DST_ROOT / f"{vid}_hires"
        n_existing = len(list(dst_dir.glob("frame_*.jpg"))) if dst_dir.exists() else 0
        if n_existing == WINDOW:
            print(f"  [OK]   {vid}_hires: {WINDOW} frames already present")
            n_ok_existing += 1
            continue

        try:
            w, h, n = _extract_one(vid, float(t_seconds))
            print(f"  [NEW]  {vid}_hires: {n} frames @ native {w}x{h}  t={t_seconds:.2f}s")
            n_new += 1
        except FileNotFoundError as e:
            print(f"  [MISS] {vid}: {e}")
            failures.append({"video_id": vid, "reason": "MP4 not found", "path": str(SRC_VIDEOS / f"{vid}.mp4")})
        except Exception as e:
            print(f"  [ERR]  {vid}: {type(e).__name__}: {e}")
            failures.append({"video_id": vid, "reason": f"{type(e).__name__}: {e}"})

    FAILURES_JSON.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"Summary: {n_ok_existing} already had frames, {n_new} newly extracted, "
          f"{len(failures)} failed/skipped.")
    if failures:
        print(f"Failures written to: {FAILURES_JSON}")
        for f in failures:
            print(f"  - {f['video_id']}: {f['reason']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

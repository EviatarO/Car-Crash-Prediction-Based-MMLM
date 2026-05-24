"""Extract NATIVE-resolution frames for the 18 v11 FP clips at a NEW timestamp.

For each FP clip:
  - t_new = max(2.0, t_original - 4.0)  (4 seconds earlier than the original anchor)
  - extract 16 frames at stride=4 ending at t_new * fps
  - output to dataset/train/<vid>_hires_early/  (separate from existing <vid>_hires/)

Idempotent: skips folders that already contain 16 frames.
Writes resample_log.json with (video_id, t_original, t_new, source_mp4_duration_s, n_frames).
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
OUT_DIR = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips_resampled"
LOG_JSON = OUT_DIR / "resample_log.json"

WINDOW = 16
STRIDE = 4
T_OFFSET = -4.0    # 4 seconds earlier than t_original
T_FLOOR = 2.0      # never sample before this

# 18 FP video_ids from leaderboard_v6_debate_v11.md
FP_CLIPS = [
    "01045", "01144", "01225", "01261", "01305", "01307", "01400", "01420",
    "01470", "01508", "01539", "01569", "01614", "01655", "01771", "01817",
    "01904", "02064",
]


def _extract_one_resampled(vid: str, t_seconds: float) -> dict:
    """Extract 16 frames into <vid>_hires_early/. Returns log dict."""
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4}")
    out_dir = DST_ROOT / f"{vid}_hires_early"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total / fps if fps else 0.0

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
    return {
        "video_id": vid,
        "t_new": round(t_seconds, 3),
        "fps": round(fps, 3),
        "total_frames": total,
        "source_mp4_duration_s": round(duration_s, 2),
        "n_frames_written": n_written,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(V11_XLSX)
    if "video_id" not in df.columns or "t_seconds" not in df.columns:
        raise SystemExit(f"v11 xlsx missing video_id/t_seconds. Got: {list(df.columns)}")

    # Build map video_id -> t_original
    t_map: dict[str, float] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("video_id")) or pd.isna(row.get("t_seconds")):
            continue
        vid = f"{int(row['video_id']):05d}"
        t_map[vid] = float(row["t_seconds"])

    print(f"Source MP4 root:   {SRC_VIDEOS}")
    print(f"Destination root:  {DST_ROOT}")
    print(f"FP clips to resample: {len(FP_CLIPS)}")
    print(f"Offset rule: t_new = max({T_FLOOR}, t_original {T_OFFSET:+})")
    print()

    log: list[dict] = []
    n_skipped = n_new = n_failed = 0
    n_floored = 0

    for vid in FP_CLIPS:
        if vid not in t_map:
            print(f"  [MISS] {vid}: not found in v11.xlsx")
            log.append({"video_id": vid, "reason": "not in v11.xlsx"})
            n_failed += 1
            continue

        t_orig = t_map[vid]
        t_new_raw = t_orig + T_OFFSET
        t_new = max(T_FLOOR, t_new_raw)
        floored = t_new_raw < T_FLOOR
        if floored:
            n_floored += 1

        dst_dir = DST_ROOT / f"{vid}_hires_early"
        n_existing = len(list(dst_dir.glob("frame_*.jpg"))) if dst_dir.exists() else 0
        if n_existing == WINDOW:
            print(f"  [OK]   {vid}_hires_early: {WINDOW} frames already present "
                  f"(t_orig={t_orig:.2f}s -> t_new={t_new:.2f}s)")
            log.append({
                "video_id": vid,
                "t_original": round(t_orig, 3),
                "t_new": round(t_new, 3),
                "floored": floored,
                "status": "skipped_existing",
                "n_frames_written": WINDOW,
            })
            n_skipped += 1
            continue

        try:
            info = _extract_one_resampled(vid, t_new)
            info["t_original"] = round(t_orig, 3)
            info["floored"] = floored
            info["status"] = "new"
            log.append(info)
            flag = " [FLOORED]" if floored else ""
            print(f"  [NEW]  {vid}_hires_early: {info['n_frames_written']} frames "
                  f"t_orig={t_orig:.2f}s -> t_new={t_new:.2f}s{flag}")
            n_new += 1
        except Exception as e:
            print(f"  [ERR]  {vid}: {type(e).__name__}: {e}")
            log.append({
                "video_id": vid,
                "t_original": round(t_orig, 3),
                "t_new": round(t_new, 3),
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            })
            n_failed += 1

    LOG_JSON.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print()
    print("=" * 60)
    print(f"Summary: {n_skipped} existing, {n_new} newly extracted, "
          f"{n_failed} failed, {n_floored} floored to t={T_FLOOR}s")
    print(f"Log: {LOG_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()

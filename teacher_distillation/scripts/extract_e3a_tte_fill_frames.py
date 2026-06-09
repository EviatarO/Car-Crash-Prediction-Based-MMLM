"""E3a TTE-fill Stage 1: extract 178 NEW frame folders.

For every video in dataset/teacher_labels/teacher_dataset_e3a.jsonl, sample the
two horizons that are NOT yet covered:

  - YES videos (gt_verdict='YES'): each row has an existing requested_time_to_event
    in {0.5, 1.0, 1.5}. Compute t_event = t_seconds + TTE, then for each missing
    horizon h in {0.5, 1.0, 1.5} \\ {existing TTE} extract 16 frames stride-4
    ending at t_new = t_event - h. Output dir: <vid>_hires_tte{05|10|15}/.

  - NO videos (gt_verdict='NO'): currently at TN_MIDPOINT (t_seconds is the
    midpoint). Two new variants at t_mid - 4 s and t_mid - 8 s, floored at 2.0 s.
    Output dirs: <vid>_hires_neg4/ and <vid>_hires_neg8/.

Native 1280x720, sequential naming frame_00001..16.jpg, stride 4. Idempotent.

Writes outputs/prompt_bakeoff/e3a_tte_fill/extraction_log.json with one row per
new variant (video_id, gt_verdict, original_horizon, new_horizon_label, t_new,
floored, n_frames, source_mp4_duration_s, fps).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train"
)
DST_ROOT = REPO_ROOT / "dataset" / "train"

E3A_JSONL = REPO_ROOT / "dataset" / "teacher_labels" / "teacher_dataset_e3a.jsonl"
V11_XLSX  = REPO_ROOT / "dataset" / "teacher_labels" / "teacher_dataset_v11.xlsx"

OUT_DIR  = REPO_ROOT / "outputs" / "prompt_bakeoff" / "e3a_tte_fill"
LOG_JSON = OUT_DIR / "extraction_log.json"

WINDOW = 16
STRIDE = 4
T_FLOOR = 2.0
NEG_OFFSETS = [(-4.0, "MID-4", "neg4"), (-8.0, "MID-8", "neg8")]
YES_HORIZONS = [(0.5, "TTE_0.5", "tte05"),
                (1.0, "TTE_1.0", "tte10"),
                (1.5, "TTE_1.5", "tte15")]
MAX_MISSING_MP4_STREAK = 3  # stop-and-ask condition


def _extract_one(vid: str, t_new: float, suffix: str) -> Dict:
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4}")
    out_dir = DST_ROOT / f"{vid}_hires_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total / fps if fps else 0.0

    end = round(t_new * fps)
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
        "fps": round(fps, 3),
        "total_frames": total,
        "source_mp4_duration_s": round(duration_s, 2),
        "n_frames_written": n_written,
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
    }


def _load_e3a() -> List[Dict]:
    return [json.loads(l) for l in E3A_JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_v11_t() -> Dict[str, float]:
    df = pd.read_excel(V11_XLSX)
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("video_id")) or pd.isna(row.get("t_seconds")):
            continue
        vid = f"{int(row['video_id']):05d}"
        out[vid] = float(row["t_seconds"])
    return out


def _plan_variants(e3a: List[Dict], t_map: Dict[str, float]) -> List[Dict]:
    """Return list of variant dicts to extract."""
    plan: List[Dict] = []
    for r in e3a:
        vid = r["video_id"]
        gt  = r["gt_verdict"]
        t_anchor = t_map.get(vid)
        if t_anchor is None:
            print(f"  [WARN] {vid}: no t_seconds in v11 xlsx; skipping")
            continue

        if gt == "YES":
            existing_tte = float(r["requested_time_to_event"])
            t_event = t_anchor + existing_tte
            for h, label, suffix in YES_HORIZONS:
                if abs(h - existing_tte) < 1e-6:
                    continue  # already covered
                t_new = t_event - h
                plan.append({
                    "video_id": vid,
                    "gt_verdict": gt,
                    "original_horizon": f"TTE_{existing_tte}",
                    "new_horizon_label": label,
                    "t_anchor_seconds": round(t_anchor, 3),
                    "t_event_seconds": round(t_event, 3),
                    "horizon_s": h,
                    "t_new": round(t_new, 3),
                    "floored": False,
                    "frames_subdir": f"{vid}_hires_{suffix}",
                })
        else:  # NO
            for off, label, suffix in NEG_OFFSETS:
                t_new_raw = t_anchor + off
                floored = t_new_raw < T_FLOOR
                t_new = max(T_FLOOR, t_new_raw)
                plan.append({
                    "video_id": vid,
                    "gt_verdict": gt,
                    "original_horizon": "MID",
                    "new_horizon_label": label,
                    "t_anchor_seconds": round(t_anchor, 3),
                    "t_event_seconds": None,
                    "horizon_s": off,
                    "t_new": round(t_new, 3),
                    "floored": floored,
                    "frames_subdir": f"{vid}_hires_{suffix}",
                })
    return plan


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Source MP4 root:  {SRC_VIDEOS}")
    print(f"Destination root: {DST_ROOT}")
    print()

    e3a = _load_e3a()
    print(f"Loaded {len(e3a)} rows from {E3A_JSONL.name}")
    t_map = _load_v11_t()
    print(f"Loaded {len(t_map)} t_seconds anchors from {V11_XLSX.name}")

    plan = _plan_variants(e3a, t_map)
    n_yes = sum(1 for p in plan if p["gt_verdict"] == "YES")
    n_no  = sum(1 for p in plan if p["gt_verdict"] == "NO")
    print(f"Planned variants: {len(plan)}  (YES={n_yes}  NO={n_no})")
    print()

    log: List[Dict] = []
    n_skipped = n_new = n_failed = n_floored = 0
    consecutive_missing = 0
    t0 = time.time()

    for idx, p in enumerate(plan, start=1):
        vid = p["video_id"]
        suffix = p["frames_subdir"].split("_hires_")[-1]
        dst_dir = DST_ROOT / p["frames_subdir"]
        n_existing = len(list(dst_dir.glob("frame_*.jpg"))) if dst_dir.exists() else 0

        if n_existing == WINDOW:
            print(f"  [{idx:3d}/{len(plan)}] [OK]  {p['frames_subdir']}: 16 frames present (t_new={p['t_new']})")
            log.append({**p, "status": "skipped_existing", "n_frames_written": WINDOW})
            n_skipped += 1
            if p["floored"]:
                n_floored += 1
            consecutive_missing = 0
            continue

        try:
            info = _extract_one(vid, p["t_new"], suffix)
            log.append({**p, **info, "status": "new"})
            flag = " [FLOORED]" if p["floored"] else ""
            print(f"  [{idx:3d}/{len(plan)}] [NEW] {p['frames_subdir']}: {info['n_frames_written']} frames "
                  f"t_new={p['t_new']}{flag}")
            n_new += 1
            if p["floored"]:
                n_floored += 1
            consecutive_missing = 0
        except FileNotFoundError as e:
            print(f"  [{idx:3d}/{len(plan)}] [MISS] {vid}: {e}")
            log.append({**p, "status": "error", "error": f"FileNotFoundError: {e}"})
            n_failed += 1
            consecutive_missing += 1
            if consecutive_missing > MAX_MISSING_MP4_STREAK:
                print()
                print(f"  STOP-AND-ASK: {consecutive_missing} consecutive MP4-missing in a row.")
                print(f"  This suggests a path drift, not isolated corruption.")
                print(f"  Halting Stage 1. Investigate {SRC_VIDEOS} before resuming.")
                stop = OUT_DIR / "STOP_REASON.json"
                stop.write_text(json.dumps({
                    "stage": "extract_e3a_tte_fill_frames",
                    "reason": "consecutive_missing_mp4",
                    "count": consecutive_missing,
                    "last_vid": vid,
                }, indent=2), encoding="utf-8")
                LOG_JSON.write_text(json.dumps(log, indent=2), encoding="utf-8")
                sys.exit(2)
        except Exception as e:
            print(f"  [{idx:3d}/{len(plan)}] [ERR] {vid}: {type(e).__name__}: {e}")
            log.append({**p, "status": "error", "error": f"{type(e).__name__}: {e}"})
            n_failed += 1

    LOG_JSON.write_text(json.dumps(log, indent=2), encoding="utf-8")
    wall = time.time() - t0

    print()
    print("=" * 65)
    print("Stage 1 COMPLETE")
    print(f"  Planned variants: {len(plan)}")
    print(f"  Existing (skipped): {n_skipped}")
    print(f"  Newly extracted:    {n_new}")
    print(f"  Failed:             {n_failed}")
    print(f"  Floored to t={T_FLOOR}s: {n_floored}")
    print(f"  Wall time:          {wall:.1f}s")
    print(f"  Output:             {LOG_JSON}")
    print("=" * 65)


if __name__ == "__main__":
    main()

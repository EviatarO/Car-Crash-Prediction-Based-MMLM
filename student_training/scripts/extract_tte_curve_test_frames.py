"""
extract_tte_curve_test_frames.py
================================
Stage 1 of the TTE anticipation-curve experiment.

For each group-0 positive clip in BOTH test halves (Private 142 + Public 142 = 284
scenes total), re-cut the clip at 5 additional offsets:
  1.0 / 1.5 / 2.0 / 2.5 / 3.0 s before the collision event.

The 0.5 s point already exists in the standard test eval JSONLs and is NOT
re-extracted here; the analysis script pulls those scores directly.

Window formula (same as e3b extractor and train manifests):
  t_event = total_frames / fps - 0.5   (group-0 assumption: event 0.5 s before end)
  t_new   = t_event - offset
  end     = round(t_new * fps)
  indices = [end - (WINDOW - 1 - i) * STRIDE for i in range(WINDOW)]
  clamped to [0, total_frames - 1]

Input manifests:
  dataset/manifests/test_manifest_hires.jsonl        → Private 677 (142 group-0 pos)
  dataset/manifests/test_manifest_public_hires.jsonl → Public  667 (142 group-0 pos)

Source MP4s (both halves share the same directory):
  ../Data-Centric-Crash-Prediction-Using-3LC-and-MViT/src/Nexar_DataSet/test/

Outputs:
  dataset/test_tte_curve/private/{vid}_hires_tte{10,15,20,25,30}/  710 frame dirs
  dataset/test_tte_curve/public/{vid}_hires_tte{10,15,20,25,30}/   710 frame dirs
  dataset/manifests/test_tte_curve_private_manifest.jsonl           710 records
  dataset/manifests/test_tte_curve_public_manifest.jsonl            710 records
  outputs/e3b_student_267clips_tte/tte_curve/extraction_log.json

Run:
  python student_training/scripts/extract_tte_curve_test_frames.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]

SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\test"
)

MANIFESTS = {
    "private": REPO_ROOT / "dataset" / "manifests" / "test_manifest_hires.jsonl",
    "public":  REPO_ROOT / "dataset" / "manifests" / "test_manifest_public_hires.jsonl",
}

DST_ROOT   = REPO_ROOT / "dataset" / "test_tte_curve"
MANIF_DIR  = REPO_ROOT / "dataset" / "manifests"
LOG_DIR    = REPO_ROOT / "outputs" / "e3b_student_267clips_tte" / "tte_curve"
LOG_JSON   = LOG_DIR / "extraction_log.json"

WINDOW = 16
STRIDE = 4
# How many consecutive missing-MP4 errors trigger a hard stop
MAX_MISS_STREAK = 5

# Offsets to extract: (seconds, folder-suffix)
OFFSETS = [
    (1.0, "tte10"),
    (1.5, "tte15"),
    (2.0, "tte20"),
    (2.5, "tte25"),
    (3.0, "tte30"),
]


def _extract_all_offsets_for_clip(
    mp4: Path, t_event: float, dst_half: Path,
    vid: str, offsets: List[Tuple[float, str]]
) -> List[Dict]:
    """
    Open the MP4 once and extract all offset windows for a single clip.
    Returns list of per-offset result dicts (one per offset).
    Idempotent: skips frames that already exist.
    """
    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {mp4}")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total / fps if fps else 0.0

    results = []
    for offset_s, suffix in offsets:
        t_new = t_event - offset_s
        out_dir = dst_half / f"{vid}_hires_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)

        end = round(t_new * fps)
        raw_indices = [end - (WINDOW - 1 - i) * STRIDE for i in range(WINDOW)]
        clamped = any(ix < 0 or ix >= total for ix in raw_indices)
        indices = [max(0, min(total - 1, ix)) for ix in raw_indices]

        n_written = 0
        for seq, fr_idx in enumerate(indices, start=1):
            dst = out_dir / f"frame_{seq:05d}.jpg"
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

        results.append({
            "offset_s":         offset_s,
            "suffix":           suffix,
            "folder_name":      f"{vid}_hires_{suffix}",
            "t_new":            round(t_new, 4),
            "fps":              round(fps, 4),
            "total_frames":     total,
            "duration_s":       round(dur, 3),
            "end_frame_idx":    end,
            "clamped":          clamped,
            "n_frames_written": n_written,
        })

    cap.release()
    return results


def _load_group0_positives(manifest_path: Path) -> List[Dict]:
    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("group") == 0 and r.get("event_occurs") == 1]


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MANIF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source MP4 root : {SRC_VIDEOS}")
    if not SRC_VIDEOS.exists():
        print(f"  ERROR: source directory not found. Aborting.")
        sys.exit(1)
    print()

    all_log: List[Dict] = []
    all_manifests: Dict[str, List[Dict]] = {"private": [], "public": []}

    for half, manif_path in MANIFESTS.items():
        clips = _load_group0_positives(manif_path)
        print(f"[{half.upper()}] {len(clips)} group-0 positive clips")
        dst_half = DST_ROOT / half

        n_new = n_skip = n_fail = n_clamped = 0
        miss_streak = 0
        t0 = time.time()
        total_clips = len(clips)

        for ci, clip in enumerate(clips, start=1):
            vid       = clip["video_id"]
            total_f   = clip["total_frames"]
            fps_meta  = clip["fps"]
            mp4       = SRC_VIDEOS / f"{vid}.mp4"
            t_event   = total_f / fps_meta - 0.5

            # Determine which offsets need extraction vs. already exist
            need_extract: List[Tuple[float, str]] = []
            for offset_s, suffix in OFFSETS:
                folder_name = f"{vid}_hires_{suffix}"
                out_dir = dst_half / folder_name
                n_ex = len(list(out_dir.glob("frame_*.jpg"))) if out_dir.exists() else 0
                if n_ex == WINDOW:
                    # Already complete — log skip and add manifest entry
                    all_log.append({
                        "half": half, "video_id": vid, "offset_s": offset_s,
                        "folder_suffix": suffix, "t_event_s": round(t_event, 4),
                        "t_new_s": round(t_event - offset_s, 4),
                        "frames_dir": folder_name,
                        "status": "skipped_existing", "n_frames_written": WINDOW, "clamped": False,
                    })
                    all_manifests[half].append(_manifest_record(clip, half, offset_s, folder_name))
                    n_skip += 1
                else:
                    need_extract.append((offset_s, suffix))

            if not need_extract:
                print(f"  [{ci:4d}/{total_clips}] [SKIP] {vid}: all 5 offsets exist")
                miss_streak = 0
                continue

            if not mp4.exists():
                for offset_s, suffix in need_extract:
                    folder_name = f"{vid}_hires_{suffix}"
                    all_log.append({
                        "half": half, "video_id": vid, "offset_s": offset_s,
                        "frames_dir": folder_name, "status": "error_missing_mp4",
                    })
                    n_fail += 1
                miss_streak += 1
                print(f"  [{ci:4d}/{total_clips}] [MISS] {vid}: MP4 not found ({len(need_extract)} offsets skipped)")
                if miss_streak > MAX_MISS_STREAK:
                    _save_log(all_log)
                    print(f"\n  STOP: {miss_streak} consecutive missing MP4s — check SRC_VIDEOS path.")
                    sys.exit(2)
                continue

            # Open MP4 once; extract all needed offsets
            try:
                offset_results = _extract_all_offsets_for_clip(
                    mp4, t_event, dst_half, vid, need_extract
                )
                for res in offset_results:
                    offset_s    = res["offset_s"]
                    folder_name = res["folder_name"]
                    all_log.append({
                        "half": half, "video_id": vid, "offset_s": offset_s,
                        "folder_suffix": res["suffix"], "t_event_s": round(t_event, 4),
                        "t_new_s": res["t_new"], "frames_dir": folder_name,
                        "fps": res["fps"], "total_frames": res["total_frames"],
                        "duration_s": res["duration_s"], "end_frame_idx": res["end_frame_idx"],
                        "clamped": res["clamped"], "n_frames_written": res["n_frames_written"],
                        "status": "new",
                    })
                    all_manifests[half].append(_manifest_record(clip, half, offset_s, folder_name,
                                                                clamped=res["clamped"]))
                    n_new += 1
                    if res["clamped"]:
                        n_clamped += 1
                miss_streak = 0
                suffix_str = "+".join(s for _, s in need_extract)
                print(f"  [{ci:4d}/{total_clips}] [NEW]  {vid}: {suffix_str}")
            except Exception as e:
                for offset_s, suffix in need_extract:
                    folder_name = f"{vid}_hires_{suffix}"
                    all_log.append({
                        "half": half, "video_id": vid, "offset_s": offset_s,
                        "frames_dir": folder_name, "status": f"error:{type(e).__name__}:{e}",
                    })
                    n_fail += 1
                print(f"  [{ci:4d}/{total_clips}] [ERR]  {vid}: {type(e).__name__}: {e}")

        wall = time.time() - t0
        print(f"\n[{half.upper()}] Done: {total_clips} clips | new={n_new} skipped={n_skip} "
              f"failed={n_fail} clamped={n_clamped}  wall={wall:.1f}s\n")

    # Write output manifests
    for half, records in all_manifests.items():
        out_path = MANIF_DIR / f"test_tte_curve_{half}_manifest.jsonl"
        out_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8"
        )
        print(f"Manifest written: {out_path}  ({len(records)} records)")

    _save_log(all_log)

    # Summary
    total_new  = sum(1 for e in all_log if e.get("status") == "new")
    total_skip = sum(1 for e in all_log if e.get("status") == "skipped_existing")
    total_excl = sum(1 for e in all_log if e.get("status", "").startswith(("excluded", "error")))
    total_clmp = sum(1 for e in all_log if e.get("clamped"))
    print()
    print("=" * 65)
    print("COMPLETE")
    print(f"  Total variants planned : {len(OFFSETS) * 284}")
    print(f"  Newly extracted        : {total_new}")
    print(f"  Already existed (skip) : {total_skip}")
    print(f"  Excluded / failed      : {total_excl}")
    print(f"  Clamped (first frames) : {total_clmp}")
    print(f"  Manifests              : dataset/manifests/test_tte_curve_{{private,public}}_manifest.jsonl")
    print(f"  Extraction log         : {LOG_JSON}")
    print("=" * 65)


def _manifest_record(clip: Dict, half: str, offset_s: float, folder_name: str,
                     clamped: bool = False) -> Dict:
    return {
        "video_id":            clip["video_id"],
        "half":                half,
        "event_occurs":        1,
        "group":               0,
        "requested_tte_s":     offset_s,
        "time_before_event_s": offset_s,   # trained_eval.py reads this → time_before_s
        "frames_dir":          folder_name,
        "frame_indices":       list(range(1, WINDOW + 1)),
        "window_size":         WINDOW,
        "t_event_s":           round(clip["total_frames"] / clip["fps"] - 0.5, 4),
        "fps":                 clip["fps"],
        "total_frames":        clip["total_frames"],
        "clamped":             clamped,
    }


def _save_log(log: List[Dict]) -> None:
    LOG_JSON.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Extraction log   : {LOG_JSON}  ({len(log)} entries)")


if __name__ == "__main__":
    main()

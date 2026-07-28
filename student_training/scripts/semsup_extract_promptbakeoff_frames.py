"""
semsup_extract_promptbakeoff_frames.py
========================================
Extracts fresh 16-frame windows for the prompt-bakeoff sample, from the raw
Nexar MP4s (discovered 2026-07-27 in a sibling project folder, NOT previously
known to be available - see docs_agents/DECISIONS.md). Unlocks the sampler
from the 89-video local ceiling (`semsup_sample_clips.py`) to the full
750/750-balanced train.csv pool.

Timing convention (matches the existing extraction scripts exactly - see
teacher_distillation/scripts/extract_e3a_tte_fill_frames.py):
  - Positives (target=1): dataset/train.csv's `time_of_event` IS the crash
    timestamp in seconds. Window for horizon h in {0.5, 1.0, 1.5} ends at
    t_event - h. Output: <vid>_hires_tte{05,10,15}/
  - Negatives (target=0): train.csv has no timestamp (no event exists), so the
    anchor is the CLIP'S OWN MIDPOINT (duration/2, read from the mp4 itself).
    Window for offset in {0, -4, -8} ends at t_anchor + offset, floored at 2.0s
    (a window needs (16-1)*4/fps ~ 2s of prior video to exist at all).
    Output: <vid>_hires_mid0/, <vid>_hires_neg4/, <vid>_hires_neg8/
    ("mid0" instead of reusing bare "_hires" for the zero-offset case - the
    existing convention's bare "_hires" is ambiguous, shared with an old
    positive-TTE backfill scheme; new extractions always get an explicit tag).

Candidate selection: excludes the incumbent 89-video caption set, the 677-clip
test set, and the 18-clip val_e3a set. Balances 250 pos / 250 neg, round-robin
across each class's 3 buckets (~83-84 each). Native 1280x720, stride 4,
sequential frame_00001..16.jpg naming - identical format to every existing
extraction, so downstream code (BADAS preprocessing, semsup_sample_clips.py's
pool discovery) needs zero changes.

Writes a new teacher_labels-style JSONL
(dataset/teacher_labels/teacher_dataset_promptbakeoff_500.jsonl) with the
fields semsup_sample_clips.py's discover_pool() already knows how to read -
after this runs, `semsup_sample_clips.py --n 500 --dry-run` should just work.

Idempotent (skips a (video_id, bucket) pair that already has 16 frames on
disk) and has a stop-and-ask safety net if too many consecutive MP4s are
missing/corrupt, matching the existing extraction scripts' convention.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "student_training" / "scripts"))

TRAIN_CSV = PROJECT_ROOT / "dataset" / "train.csv"
SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train"
)
DST_ROOT = PROJECT_ROOT / "dataset" / "train"
OUT_LABEL_FILE = PROJECT_ROOT / "dataset" / "teacher_labels" / "teacher_dataset_promptbakeoff_500.jsonl"
LOG_JSON = PROJECT_ROOT / "outputs" / "semantic_captions" / "promptbakeoff" / "extraction_log.json"

WINDOW = 16
STRIDE = 4
T_FLOOR = 2.0
MAX_CONSECUTIVE_FAILURES = 5

POS_BUCKETS = [(0.5, "TTE_0.5", "tte05"), (1.0, "TTE_1.0", "tte10"), (1.5, "TTE_1.5", "tte15")]
NEG_BUCKETS = [(0.0, "MID", "mid0"), (-4.0, "MID-4", "neg4"), (-8.0, "MID-8", "neg8")]


def load_train_labels() -> dict:
    out = {}
    with open(TRAIN_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vid = f"{int(r['id']):05d}"
            out[vid] = {"target": int(r["target"]),
                        "time_of_event": float(r["time_of_event"]) if r["time_of_event"] else None}
    return out


def load_excluded_video_ids() -> set:
    excluded = set()
    incumbent = PROJECT_ROOT / "outputs" / "semantic_captions" / "Caption_Train_All_Clips.jsonl"
    for fp in (incumbent,
               PROJECT_ROOT / "dataset" / "manifests" / "test_manifest_hires.jsonl",
               PROJECT_ROOT / "dataset" / "manifests" / "val_e3a.jsonl"):
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    excluded.add(json.loads(line)["video_id"])
    return excluded


def plan_candidates(labels: dict, excluded: set, n_per_class: int, seed: int) -> list:
    rng = random.Random(seed)
    pos_pool = sorted(vid for vid, d in labels.items()
                       if d["target"] == 1 and vid not in excluded and d["time_of_event"] is not None)
    neg_pool = sorted(vid for vid, d in labels.items() if d["target"] == 0 and vid not in excluded)
    rng.shuffle(pos_pool)
    rng.shuffle(neg_pool)

    plan = []
    # round-robin bucket assignment done per-class so counts stay balanced
    for i, vid in enumerate(pos_pool[:n_per_class]):
        h, label, suffix = POS_BUCKETS[i % 3]
        plan.append({"video_id": vid, "target": 1, "bucket_label": label, "suffix": suffix, "param": h})
    for i, vid in enumerate(neg_pool[:n_per_class]):
        off, label, suffix = NEG_BUCKETS[i % 3]
        plan.append({"video_id": vid, "target": 0, "bucket_label": label, "suffix": suffix, "param": off})

    if len(pos_pool) < n_per_class or len(neg_pool) < n_per_class:
        print(f"WARNING: pool smaller than requested - pos available={len(pos_pool)}, "
              f"neg available={len(neg_pool)}, requested {n_per_class} each")
    return plan


def _extract_one(vid: str, t_new: float) -> dict:
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4}")
    cap = cv2.VideoCapture(str(mp4))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or not total:
            raise RuntimeError(f"Unreadable video metadata (fps={fps}, total={total}): {mp4}")
        end = round(t_new * fps)
        indices = [end - (WINDOW - 1 - i) * STRIDE for i in range(WINDOW)]
        indices = [max(0, min(total - 1, ix)) for ix in indices]
        frames = []
        for fr_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {fr_idx} from {mp4}")
            frames.append(frame)
        return {"frames": frames, "fps": round(fps, 3), "total_frames": total}
    finally:
        cap.release()


def get_midpoint_seconds(vid: str) -> float:
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    cap = cv2.VideoCapture(str(mp4))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or not total:
            raise RuntimeError(f"Unreadable video metadata: {mp4}")
        return (total / fps) / 2.0
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-class", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap total plan size (mechanics test)")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no extraction/writes")
    args = ap.parse_args()

    labels = load_train_labels()
    excluded = load_excluded_video_ids()
    print(f"train.csv: {len(labels)} labeled videos. Excluded (incumbent+test+val): {len(excluded)}.")

    plan = plan_candidates(labels, excluded, args.n_per_class, args.seed)
    if args.limit:
        # keep it balanced even when capped, for a meaningful mechanics test
        pos = [p for p in plan if p["target"] == 1][:args.limit // 2]
        neg = [p for p in plan if p["target"] == 0][:args.limit // 2]
        plan = pos + neg
    n_pos = sum(1 for p in plan if p["target"] == 1)
    n_neg = sum(1 for p in plan if p["target"] == 0)
    print(f"Planned: {len(plan)} rows ({n_pos} pos, {n_neg} neg)")
    from collections import Counter
    print("  bucket counts:", dict(Counter(p["bucket_label"] for p in plan)))

    if args.dry_run:
        print("[dry-run] stopping before extraction.")
        return

    OUT_LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_JSON.parent.mkdir(parents=True, exist_ok=True)

    log, n_new, n_skipped, n_failed, consecutive_failures = [], 0, 0, 0, 0
    t0 = time.time()
    midpoint_cache = {}

    with open(OUT_LABEL_FILE, "w", encoding="utf-8") as out_f:
        for idx, p in enumerate(plan, start=1):
            vid, suffix = p["video_id"], p["suffix"]
            out_dir = DST_ROOT / f"{vid}_hires_{suffix}"
            existing = len(list(out_dir.glob("frame_*.jpg"))) if out_dir.exists() else 0

            if existing == WINDOW:
                status, t_new, floored = "skipped_existing", None, False
                n_skipped += 1
            else:
                try:
                    if p["target"] == 1:
                        t_event = labels[vid]["time_of_event"]
                        t_new_raw = t_event - p["param"]
                    else:
                        if vid not in midpoint_cache:
                            midpoint_cache[vid] = get_midpoint_seconds(vid)
                        t_new_raw = midpoint_cache[vid] + p["param"]
                    floored = t_new_raw < T_FLOOR
                    t_new = max(T_FLOOR, t_new_raw)

                    info = _extract_one(vid, t_new)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for i, frame in enumerate(info["frames"], start=1):
                        cv2.imwrite(str(out_dir / f"frame_{i:05d}.jpg"), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])
                    status = "new"
                    n_new += 1
                    consecutive_failures = 0
                except Exception as e:
                    status = f"error: {type(e).__name__}: {e}"
                    n_failed += 1
                    consecutive_failures += 1
                    t_new, floored = None, False
                    print(f"  [{idx:4d}/{len(plan)}] [ERR] {vid} ({p['bucket_label']}): {e}")
                    if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                        print(f"\nSTOP-AND-ASK: {consecutive_failures} consecutive failures. "
                              f"Halting - investigate before resuming.")
                        break

            row = {
                "video_id": vid, "frames_dir": f"{vid}_hires_{suffix}",
                "requested_time_to_event": p["param"] if p["target"] == 1 else f"{p['param']}_offset",
                "horizon_label": p["bucket_label"], "gt_verdict": "YES" if p["target"] == 1 else "NO",
                "target": p["target"], "row_origin": "promptbakeoff_500_extraction",
            }
            if status in ("new", "skipped_existing"):
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.append({**row, "status": status, "t_new": t_new, "floored": floored})

            if idx % 50 == 0 or idx == len(plan):
                print(f"  [{idx:4d}/{len(plan)}] new={n_new} skipped={n_skipped} failed={n_failed} "
                      f"({time.time()-t0:.0f}s)")

    LOG_JSON.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print()
    print("=" * 70)
    print(f"DONE. new={n_new} skipped_existing={n_skipped} failed={n_failed} "
          f"wall={time.time()-t0:.0f}s")
    print(f"Label file: {OUT_LABEL_FILE}")
    print(f"Log:        {LOG_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()

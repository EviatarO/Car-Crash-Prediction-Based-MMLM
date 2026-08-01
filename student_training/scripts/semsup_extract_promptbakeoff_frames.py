"""
semsup_extract_promptbakeoff_frames.py
========================================
Extracts fresh 16-frame windows for the prompt-bakeoff sample, from the raw
Nexar MP4s (discovered 2026-07-27 in a sibling project folder, NOT previously
known to be available - see docs_agents/DECISIONS.md). Unlocks the sampler
from the 89-video local ceiling (`semsup_sample_clips.py`) to the full
750/750-balanced train.csv pool.

Two input modes:
  1. Default (no --manifest): the original random-sampling mode - excludes
     the incumbent 89-video caption set / test / val, balances --n-per-class
     pos/neg, round-robins buckets. Unchanged from the original script.
  2. --manifest <path.jsonl>: drive extraction from an explicit manifest (e.g.
     dataset/manifests/train4500_hires.jsonl from build_train4500_manifest.py)
     instead of sampling. Every (video_id, horizon_label) row in the manifest
     becomes one extraction target; `frames_dir` in the manifest determines
     the output directory name directly (no bucket-index guessing needed).

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

SEQUENTIAL DECODE (added for the train4500 run - see
~/.claude/plans/but-if-b-a1-it-woolly-metcalfe.md "Frame-sharing... analysed
and rejected" / "sequential decode (adopted)"): per-frame `cap.set(POS_FRAMES)`
seeking costs ~160ms/seek and dominates extraction time (measured ~1.1s per
16-frame window = 16 seeks). Each window's own 16 frames span only
(16-1)*STRIDE = 60 source frames, so ONE seek to the window's start index,
followed by sequential .read() through to its end index (keeping only the
STRIDE-selected frames), replaces 16 seeks with 1. Benchmarked on 6 real train
videos: 3.9x faster, byte-identical JPEGs to the old per-frame-seek method
(verified via np.array_equal). This does NOT merge frames across a video's 3
TTE/offset buckets (those remain 3 independent single-seek reads) - buckets
are far enough apart (60-300 source frames) that merging them would trade a
cheap extra seek for an expensive long sequential read across frames that are
mostly unused; see the plan doc for the arithmetic.

Candidate selection (default mode): excludes the incumbent 89-video caption
set, the 677-clip test set, and the 18-clip val_e3a set. Balances 250 pos /
250 neg, round-robin across each class's 3 buckets (~83-84 each). Native
1280x720, stride 4, sequential frame_00001..16.jpg naming - identical format
to every existing extraction, so downstream code (BADAS preprocessing,
semsup_sample_clips.py's pool discovery) needs zero changes.

Writes a new teacher_labels-style JSONL (default mode:
dataset/teacher_labels/teacher_dataset_promptbakeoff_500.jsonl; --manifest
mode: alongside --out-label, default dataset/teacher_labels/
teacher_dataset_train4500.jsonl) with the fields semsup_sample_clips.py's
discover_pool() already knows how to read.

Idempotent (skips a (video_id, bucket) pair that already has 16 frames on
disk) and has a stop-and-ask safety net if too many consecutive MP4s are
missing/corrupt, matching the existing extraction scripts' convention. The
label JSONL is now written via upsert-append (read existing rows first, only
append genuinely new ones) rather than truncate-on-open - a run halted by the
consecutive-failure break no longer loses previously-recorded rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
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
OUT_LABEL_FILE_MANIFEST_MODE = PROJECT_ROOT / "dataset" / "teacher_labels" / "teacher_dataset_train4500.jsonl"
LOG_JSON = PROJECT_ROOT / "outputs" / "semantic_captions" / "promptbakeoff" / "extraction_log.json"

WINDOW = 16
STRIDE = 4
T_FLOOR = 2.0
MAX_CONSECUTIVE_FAILURES = 5

POS_BUCKETS = [(0.5, "TTE_0.5", "tte05"), (1.0, "TTE_1.0", "tte10"), (1.5, "TTE_1.5", "tte15")]
# MID moved from offset 0.0 to -10.0 (see build_train4500_manifest.py's NEG_BUCKETS
# comment) - the clip-midpoint window produced 42.8% high-confidence false positives
# in real scoring, isolated to that one bucket, vs 14.4% for MID-4/MID-8.
NEG_BUCKETS = [(-10.0, "MID-10", "mid10"), (-4.0, "MID-4", "neg4"), (-8.0, "MID-8", "neg8")]


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


def load_manifest_plan(manifest_path: Path) -> list:
    """Convert a build_train4500_manifest.py-style JSONL into the same plan-dict
    shape used by the sampler mode, so the rest of the pipeline is unforked."""
    plan = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            suffix = r["frames_dir"].split("_hires_", 1)[1]
            target = r["event_occurs"]
            if target == 1:
                param = r["time_before_event_s"]
            else:
                # negatives: recover the offset from the suffix (mid10/neg4/neg8),
                # since time_before_event_s is None for negatives by design.
                param = {"mid10": -10.0, "neg4": -4.0, "neg8": -8.0}[suffix]
            plan.append({"video_id": r["video_id"], "target": target,
                         "bucket_label": r["horizon_label"], "suffix": suffix, "param": param})
    return plan


def _read_window_sequential(cap, indices: list) -> list:
    """One seek to indices[0]/min, then sequential .read() through max(indices),
    keeping only the frames actually needed. Replaces WINDOW separate seeks
    with 1 seek + (span) sequential reads - see module docstring for the
    measured 3.9x speedup and the byte-identity verification."""
    start, stop = min(indices), max(indices)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    needed = set(indices)
    captured = {}
    cur = start
    while cur <= stop:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {cur} (span {start}-{stop})")
        if cur in needed:
            captured.setdefault(cur, frame)
        cur += 1
    return [captured[i] for i in indices]


def get_video_meta(vid: str) -> tuple:
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4}")
    cap = cv2.VideoCapture(str(mp4))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or not total:
            raise RuntimeError(f"Unreadable video metadata (fps={fps}, total={total}): {mp4}")
        return fps, total
    finally:
        cap.release()


def _indices_for(t_new: float, fps: float, total: int) -> list:
    end = round(t_new * fps)
    indices = [end - (WINDOW - 1 - i) * STRIDE for i in range(WINDOW)]
    return [max(0, min(total - 1, ix)) for ix in indices]


def process_video(args_tuple) -> list:
    """Worker unit: extract every bucket for ONE video_id (own mp4 open, own
    sequential-decode reads - one seek per bucket, not shared across buckets;
    see module docstring for why buckets aren't merged). Returns a list of
    per-bucket result dicts. Safe to run under multiprocessing.Pool since all
    state is local to this call."""
    vid, bucket_rows = args_tuple
    results = []
    try:
        fps, total = get_video_meta(vid)
    except Exception as e:
        for p in bucket_rows:
            results.append({**p, "status": f"error: {type(e).__name__}: {e}", "t_new": None, "floored": False})
        return results

    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    cap = cv2.VideoCapture(str(mp4))
    midpoint = None
    try:
        for p in bucket_rows:
            out_dir = DST_ROOT / f"{vid}_hires_{p['suffix']}"
            existing = len(list(out_dir.glob("frame_*.jpg"))) if out_dir.exists() else 0
            if existing == WINDOW:
                results.append({**p, "status": "skipped_existing", "t_new": None, "floored": False})
                continue
            try:
                if p["target"] == 1:
                    t_new_raw = None  # positives already carry an absolute anchor via param below
                    t_event = p.get("_t_event")
                    t_new_raw = t_event - p["param"] if t_event is not None else p["param"]
                else:
                    if midpoint is None:
                        midpoint = (total / fps) / 2.0
                    t_new_raw = midpoint + p["param"]
                floored = t_new_raw < T_FLOOR
                t_new = max(T_FLOOR, t_new_raw)

                indices = _indices_for(t_new, fps, total)
                frames = _read_window_sequential(cap, indices)
                out_dir.mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(frames, start=1):
                    cv2.imwrite(str(out_dir / f"frame_{i:05d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                results.append({**p, "status": "new", "t_new": t_new, "floored": floored})
            except Exception as e:
                results.append({**p, "status": f"error: {type(e).__name__}: {e}",
                                 "t_new": None, "floored": False})
    finally:
        cap.release()
    return results


def load_existing_label_keys(out_label_file: Path) -> set:
    """(video_id, suffix) pairs already recorded, for upsert-append."""
    if not out_label_file.exists():
        return set()
    keys = set()
    with open(out_label_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                suffix = r["frames_dir"].split("_hires_", 1)[1] if "_hires_" in r["frames_dir"] else ""
                keys.add((r["video_id"], suffix))
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-class", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap total plan size (mechanics test)")
    ap.add_argument("--dry-run", action="store_true", help="plan only, no extraction/writes")
    ap.add_argument("--manifest", default=None,
                     help="drive extraction from a build_train4500_manifest.py-style JSONL "
                          "instead of the random sampler")
    ap.add_argument("--out-label", default=None,
                     help="override the output label JSONL path")
    ap.add_argument("--workers", type=int, default=1, help="multiprocessing pool size (by video_id)")
    ap.add_argument("--chunk-size", type=int, default=0,
                     help="process only this many DISTINCT VIDEOS per invocation (0 = all); "
                          "chunks are cut on video_id boundaries so a video's buckets never split")
    ap.add_argument("--chunk-index", type=int, default=0,
                     help="which chunk to process (0-based), used with --chunk-size")
    args = ap.parse_args()

    out_label_file = Path(args.out_label) if args.out_label else (
        OUT_LABEL_FILE_MANIFEST_MODE if args.manifest else OUT_LABEL_FILE)

    if args.manifest:
        plan = load_manifest_plan(Path(args.manifest))
        labels = load_train_labels()  # for t_event lookup on positives
        for p in plan:
            if p["target"] == 1:
                p["_t_event"] = labels[p["video_id"]]["time_of_event"]
        print(f"Loaded manifest plan: {len(plan)} rows from {args.manifest}")
    else:
        labels = load_train_labels()
        excluded = load_excluded_video_ids()
        print(f"train.csv: {len(labels)} labeled videos. Excluded (incumbent+test+val): {len(excluded)}.")
        plan = plan_candidates(labels, excluded, args.n_per_class, args.seed)

    if args.limit:
        pos = [p for p in plan if p["target"] == 1][:args.limit // 2]
        neg = [p for p in plan if p["target"] == 0][:args.limit // 2]
        plan = pos + neg

    # group by video_id (preserves the "one mp4 open per video" property needed
    # for sequential decode, and is the natural chunk-boundary unit)
    by_video = defaultdict(list)
    for p in plan:
        by_video[p["video_id"]].append(p)

    # Class-interleaved ordering, NOT a plain sort. train.csv's raw ids put
    # every positive at 0-1039 and every negative at 1040-2139 (verified) - a
    # plain sorted(by_video.keys()) puts entire classes in separate chunks,
    # which breaks the pipeline's early-abort checkpoint (deliverable 2b): a
    # single-class chunk has an undefined/degenerate confusion matrix and
    # AP, so its comparison against A0's 23.6% test error rate means nothing.
    # Round-robin pos/neg (each class independently sorted first, for
    # determinism) so every contiguous slice is class-balanced by construction.
    pos_ids = sorted(v for v in by_video if by_video[v][0]["target"] == 1)
    neg_ids = sorted(v for v in by_video if by_video[v][0]["target"] == 0)
    video_ids = []
    for a, b in zip(pos_ids, neg_ids):
        video_ids.append(a)
        video_ids.append(b)
    video_ids.extend(pos_ids[len(neg_ids):])
    video_ids.extend(neg_ids[len(pos_ids):])

    if args.chunk_size:
        start = args.chunk_index * args.chunk_size
        video_ids = video_ids[start:start + args.chunk_size]
        n_chunks = (len(by_video) + args.chunk_size - 1) // args.chunk_size
        print(f"Chunk {args.chunk_index}/{n_chunks - 1}: {len(video_ids)} videos "
              f"({sum(len(by_video[v]) for v in video_ids)} rows)")

    n_pos = sum(1 for v in video_ids for p in by_video[v] if p["target"] == 1)
    n_neg = sum(1 for v in video_ids for p in by_video[v] if p["target"] == 0)
    n_rows = sum(len(by_video[v]) for v in video_ids)
    print(f"Planned: {n_rows} rows across {len(video_ids)} videos ({n_pos} pos, {n_neg} neg)")
    from collections import Counter
    print("  bucket counts:", dict(Counter(p["bucket_label"] for v in video_ids for p in by_video[v])))

    if args.dry_run:
        print("[dry-run] stopping before extraction.")
        return

    out_label_file.parent.mkdir(parents=True, exist_ok=True)
    LOG_JSON.parent.mkdir(parents=True, exist_ok=True)

    already_written = load_existing_label_keys(out_label_file)
    print(f"Label file already has {len(already_written)} rows recorded (upsert-append mode).")

    work_items = [(vid, by_video[vid]) for vid in video_ids]

    log, n_new, n_skipped, n_failed, n_appended = [], 0, 0, 0, 0
    t0 = time.time()
    consecutive_failures = 0

    with open(out_label_file, "a", encoding="utf-8") as out_f:
        if args.workers > 1:
            pool = Pool(args.workers)
            iterator = pool.imap(process_video, work_items, chunksize=1)
        else:
            pool = None
            iterator = (process_video(item) for item in work_items)

        try:
            for vid_idx, results in enumerate(iterator, start=1):
                for r in results:
                    status = r["status"]
                    if status == "new":
                        n_new += 1
                        consecutive_failures = 0
                    elif status == "skipped_existing":
                        n_skipped += 1
                    else:
                        n_failed += 1
                        consecutive_failures += 1
                        print(f"  [{vid_idx:4d}/{len(video_ids)}] [ERR] {r['video_id']} "
                              f"({r['bucket_label']}): {status}")

                    row = {
                        "video_id": r["video_id"], "frames_dir": f"{r['video_id']}_hires_{r['suffix']}",
                        "requested_time_to_event": r["param"] if r["target"] == 1 else f"{r['param']}_offset",
                        "horizon_label": r["bucket_label"], "gt_verdict": "YES" if r["target"] == 1 else "NO",
                        "target": r["target"], "row_origin": "promptbakeoff_500_extraction"
                        if not args.manifest else "train4500_extraction",
                    }
                    key = (r["video_id"], r["suffix"])
                    if status in ("new", "skipped_existing") and key not in already_written:
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        already_written.add(key)
                        n_appended += 1
                    log.append({**row, "status": status, "t_new": r.get("t_new"), "floored": r.get("floored")})

                if vid_idx % 25 == 0 or vid_idx == len(video_ids):
                    out_f.flush()
                    print(f"  [{vid_idx:4d}/{len(video_ids)}] new={n_new} skipped={n_skipped} "
                          f"failed={n_failed} ({time.time()-t0:.0f}s)")

                if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                    print(f"\nSTOP-AND-ASK: {consecutive_failures} consecutive failures. "
                          f"Halting - investigate before resuming.")
                    break
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()

    LOG_JSON.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print()
    print("=" * 70)
    print(f"DONE. new={n_new} skipped_existing={n_skipped} failed={n_failed} "
          f"label_rows_appended={n_appended} wall={time.time()-t0:.0f}s")
    print(f"Label file: {out_label_file}")
    print(f"Log:        {LOG_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()

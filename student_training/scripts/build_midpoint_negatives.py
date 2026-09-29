"""
build_midpoint_negatives.py
============================
Re-extracts the 1,761-pool's 905 negative windows using the Nexar TEST set's own
negative-sampling protocol instead of MID-10/-8/-4, to test whether the mismatch
between how our training negatives and the Nexar test negatives are cut explains
part of the train/test AP gap (docs_agents/EXPERIMENTS.md, 2026-09-26 re-analysis).

Nexar test protocol (dataset paper, arXiv 2503.03848 sec 4.1): for a negative
video, a fake event time is placed at the clip's own midpoint plus Gaussian
noise; three windows are then cut ending 0.5/1.0/1.5s before that fake event -
structurally identical to how positive windows are cut before the REAL event.
Our existing MID-10/-4/-8 buckets instead cut at three fixed, deliberately
off-midpoint offsets (moved there in 2026 because the literal midpoint produced
43% FP - see build_train4500_manifest.py's own comment on NEG_BUCKETS).

Only the 1,761 pool's EXISTING negative rows are touched: same 564 videos, same
905-window count, same per-video bucket multiplicity (a video that contributed
one MID-8 window gets exactly one new window back, at the same group index) -
so this is a pure resampling-protocol swap, not a pool-size change. Positives
are untouched (copied through as-is).

One fake-event offset is drawn PER VIDEO (not per window), so a video's 1-3
windows share one consistent fake event, matching how the real protocol works.

Usage:
    python build_midpoint_negatives.py [--noise-std 1.0] [--seed 0] [--dry-run]

Writes:
    outputs/semantic_captions/Caption_Train4500_MidpointNeg_1761.jsonl
    dataset/train/<vid>_hires_midtest{05,10,15}/frame_*.jpg  (905 windows)
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_CAPTIONS = PROJECT_ROOT / "outputs" / "semantic_captions" / "Caption_Train4500_Mixed_1761.jsonl"
OUT_CAPTIONS = PROJECT_ROOT / "outputs" / "semantic_captions" / "Caption_Train4500_MidpointNeg_1761.jsonl"
SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train"
)
DST_ROOT = PROJECT_ROOT / "dataset" / "train"

WINDOW = 16
STRIDE = 4
T_FLOOR = 2.0

# old bucket -> group index, matching build_train4500_manifest.py's NEG_BUCKETS exactly
# (MID-10=grp0, MID-4=grp1, MID-8=grp2) so a comparison by group stays aligned.
OLD_GRP = {"MID-10": 0, "MID-4": 1, "MID-8": 2}
# group -> (new horizon label, new frames_dir suffix, seconds before the fake event) -
# same 0.5/1.0/1.5s spacing the POSITIVE TTE buckets use, and the same spacing the real
# Nexar test protocol uses for its negatives.
NEW_BUCKET = {0: ("MIDTEST-0.5", "midtest05", 0.5),
              1: ("MIDTEST-1.0", "midtest10", 1.0),
              2: ("MIDTEST-1.5", "midtest15", 1.5)}


def video_noise(vid: str, std: float, seed: int) -> float:
    """Deterministic per-video Gaussian noise (seconds), so re-running this script
    (or --dry-run then a real run) reproduces the exact same fake event times."""
    h = hashlib.sha256(f"{seed}:{vid}".encode()).hexdigest()
    r = random.Random(int(h[:16], 16))
    return r.gauss(0.0, std)


def get_video_meta(vid: str):
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


def indices_for(t_new: float, fps: float, total: int):
    end = round(t_new * fps)
    idx = [end - (WINDOW - 1 - i) * STRIDE for i in range(WINDOW)]
    return [max(0, min(total - 1, ix)) for ix in idx]


def read_window_sequential(cap, indices):
    start, stop = min(indices), max(indices)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    needed = set(indices)
    captured, cur = {}, start
    while cur <= stop:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {cur} (span {start}-{stop})")
        if cur in needed:
            captured.setdefault(cur, frame)
        cur += 1
    return [captured[i] for i in indices]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-std", type=float, default=1.0,
                     help="stddev (s) of the per-video fake-event offset from the true "
                          "midpoint - the Nexar paper doesn't publish this value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="plan + print only, extract nothing")
    ap.add_argument("--complete-horizons", action="store_true",
                     help="cut ALL of 0.5/1.0/1.5s for every negative video (existing windows are "
                          "skipped, same seeded fake event so they are reproduced exactly) and write "
                          "a separate LookAhead_NegWindows manifest; the training captions file is "
                          "NOT touched")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(SRC_CAPTIONS, encoding="utf-8")]
    pos_rows = [r for r in rows if r["gt_verdict"] == "YES"]
    neg_rows = [r for r in rows if r["gt_verdict"] == "NO"]
    print(f"source pool: {len(rows)} rows ({len(pos_rows)} pos, {len(neg_rows)} neg, "
          f"{len({r['video_id'] for r in neg_rows})} unique negative videos)")

    unmapped = {r["horizon_label"] for r in neg_rows} - set(OLD_GRP)
    assert not unmapped, f"unrecognized negative horizon_label(s): {unmapped}"

    video_ids = sorted({r["video_id"] for r in neg_rows})
    noise = {vid: video_noise(vid, args.noise_std, args.seed) for vid in video_ids}

    out_rows, plan, errors = [], [], []
    meta_cache = {}
    if args.complete_horizons:
        work = [({"video_id": v, "gt_verdict": "NO"}, g) for v in video_ids for g in (0, 1, 2)]
    else:
        work = [(r, OLD_GRP[r["horizon_label"]]) for r in neg_rows]
    for r, grp in work:
        vid = r["video_id"]
        label, suffix, h = NEW_BUCKET[grp]
        try:
            if vid not in meta_cache:
                meta_cache[vid] = get_video_meta(vid)
            fps, total = meta_cache[vid]
        except Exception as e:
            errors.append((vid, str(e)))
            continue
        midpoint = (total / fps) / 2.0
        t_new_raw = midpoint + noise[vid] - h
        floored = t_new_raw < T_FLOOR
        t_new = max(T_FLOOR, t_new_raw)
        frames_dir = f"{vid}_hires_{suffix}"
        plan.append({"video_id": vid, "grp": grp, "suffix": suffix, "t_new": t_new,
                     "floored": floored, "frames_dir": frames_dir})
        out_rows.append({**r, "frames_dir": frames_dir, "horizon_label": label,
                         "t_seconds": round(t_new, 3)})

    print(f"planned {len(plan)} negative windows across {len(meta_cache)} videos "
          f"({len(errors)} video(s) failed metadata read)")
    n_floored = sum(1 for p in plan if p["floored"])
    if n_floored:
        print(f"  {n_floored} window(s) hit the {T_FLOOR}s floor (short video + large "
              f"negative noise) - same fallback the original MID buckets already rely on")
    for vid, e in errors[:10]:
        print(f"  [error] {vid}: {e}")

    if args.dry_run:
        print("[dry-run] wrote nothing")
        return

    n_new, n_skip, n_err = 0, 0, 0
    for i, p in enumerate(plan):
        out_dir = DST_ROOT / p["frames_dir"]
        if out_dir.exists() and len(list(out_dir.glob("frame_*.jpg"))) == WINDOW:
            n_skip += 1
            continue
        vid = p["video_id"]
        fps, total = meta_cache[vid]
        mp4 = SRC_VIDEOS / f"{vid}.mp4"
        cap = cv2.VideoCapture(str(mp4))
        try:
            idx = indices_for(p["t_new"], fps, total)
            frames = read_window_sequential(cap, idx)
            out_dir.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(frames, start=1):
                cv2.imwrite(str(out_dir / f"frame_{j:05d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_new += 1
        except Exception as e:
            print(f"  [error] {vid} ({p['frames_dir']}): {e}")
            n_err += 1
        finally:
            cap.release()
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(plan)} (new={n_new} skip={n_skip} err={n_err})")

    print(f"extraction done: new={n_new} skip={n_skip} err={n_err}")
    assert n_err == 0, f"{n_err} window(s) failed to extract - fix before training on this pool"

    if args.complete_horizons:
        manifest = PROJECT_ROOT / "outputs" / "semantic_captions" / "LookAhead_NegWindows_564x3.jsonl"
        with open(manifest, "w", encoding="utf-8") as f:
            for p in plan:
                f.write(json.dumps({"video_id": p["video_id"], "frames_dir": p["frames_dir"],
                                    "horizon_label": NEW_BUCKET[p["grp"]][0],
                                    "t_seconds": round(p["t_new"], 3)}) + "\n")
        print(f"[wrote] {manifest} ({len(plan)} windows; captions file untouched)")
        return

    final_rows = pos_rows + out_rows
    assert len(final_rows) == len(rows), \
        f"row count drifted: {len(final_rows)} vs source {len(rows)}"
    with open(OUT_CAPTIONS, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[wrote] {OUT_CAPTIONS} ({len(final_rows)} rows: {len(pos_rows)} pos unchanged, "
          f"{len(out_rows)} neg re-cut)")


if __name__ == "__main__":
    sys.exit(main())

"""
aa1_yolop_cache.py
===================
Stage 0/1 (child plan 2026-09-15-AA1-v2b): run YOLOPv2 on all 18 val_e3a clips using the SAME
decoded spans as the v1 G-DINO run, and cache raw per-frame outputs (boxes/scores + drivable
and lane masks, RLE-compressed). Kept separate from detection-track-rank.py's cache so tracking
and geometry changes never require re-running the model (~3 min for all 18 clips, vs G-DINO's
~50 min).

Also runs the Stage 1 detector comparison against the v1 G-DINO tracks cached in
`outputs/aa1_smoke_18clips/<vid>_tracks.json`: per clip, IoU-matches YOLOPv2 boxes (per frame)
against G-DINO's tracked boxes at the same timestamp, to see whether YOLOPv2 finds the same
vehicles (and specifically the known partner objects) at comparable recall.

Usage:
  python aa1_yolop_cache.py --all --out-dir ../../outputs/aa1_v2_18clips
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aa1_detect_track_rank import decode_frames, decode_span, load_cache  # noqa: E402
from aa1_scene import YOLOPv2, CONF_THRES  # noqa: E402

VAL_E3A_IDS = ["00319", "00077", "00687", "00283", "00147", "00529", "00493", "00474",
               "00372", "01153", "01504", "01643", "01281", "01550", "01737", "02104",
               "02117", "01552"]
V1_DIR = REPO / "outputs" / "aa1_smoke_18clips"


def rle_encode(mask: np.ndarray) -> dict:
    """Run-length encode a boolean mask (row-major), for compact JSON storage.
    BUG FIXED 2026-09-15: the original version used `prepend=1-flat[0]`, which always
    registers a transition at index 0 and therefore always emits a spurious zero-length first
    run. rle_decode's val-flip is tied to run position, so that phantom run silently inverted
    every real segment's value (confirmed: cached lane masks decoded ~97-99% True, when the
    single-frame sanity check - same mask, no RLE round-trip - showed a normal thin lane
    overlay). Fixed here; all 18 clips' YOLOPv2 cache must be regenerated (see child plan
    2026-09-15-AA1-v2b Stage 2 notes)."""
    flat = mask.reshape(-1).astype(np.uint8)
    if flat.size == 0:
        return dict(shape=list(mask.shape), start=0, runs=[])
    changes = np.flatnonzero(np.diff(flat)) + 1  # index right after each real transition
    boundaries = np.concatenate([[0], changes, [flat.size]])
    runs = np.diff(boundaries)
    return dict(shape=list(mask.shape), start=int(flat[0]), runs=runs.tolist())


def run_clip(yp: YOLOPv2, video_id: str, out_dir: Path):
    t_start, t_end, is_pos = decode_span(video_id)
    frames, timestamps, fps = decode_frames(video_id, t_start, t_end)
    t0 = time.time()
    per_frame = []
    for frame, t in zip(frames, timestamps):
        out = yp.infer(frame)
        per_frame.append(dict(
            t=round(t, 4), boxes=out["boxes"].round(1).tolist(), scores=out["scores"].round(4).tolist(),
            drivable_rle=rle_encode(out["drivable"]), lane_rle=rle_encode(out["lane"])))
    dt = time.time() - t0
    n_boxes = sum(len(p["boxes"]) for p in per_frame)
    print(f"[{video_id}] YOLOPv2: {len(frames)} frames, {n_boxes} boxes "
         f"({n_boxes/len(frames):.1f}/frame), {dt:.1f}s ({dt/len(frames)*1000:.0f}ms/frame)")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{video_id}_yolop.json", "w", encoding="utf-8") as f:
        json.dump(dict(video_id=video_id, is_positive=is_pos, span=[t_start, t_end], fps=fps,
                       n_frames=len(frames), timing_s=round(dt, 2), frames=per_frame), f)
    return per_frame, timestamps


def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def compare_to_gdino(video_id: str, yolop_frames: list, iou_thresh: float = 0.4):
    """Per-frame IoU match: for each G-DINO box (from the v1 tracked cache), is there a YOLOPv2
    box with IoU >= iou_thresh at the same timestamp? Recall = matched / total G-DINO boxes."""
    try:
        gd_cache = load_cache(V1_DIR, video_id)
    except FileNotFoundError:
        return None
    gd_by_t = {}
    for tid, pts in gd_cache["tracks"].items():
        for t, box, conf in pts:
            gd_by_t.setdefault(round(t, 4), []).append((tid, box))
    yp_by_t = {p["t"]: p["boxes"] for p in yolop_frames}

    matched, total = 0, 0
    partner_track_id = None
    for t, gd_boxes in gd_by_t.items():
        yp_boxes = yp_by_t.get(t, [])
        for tid, gbox in gd_boxes:
            total += 1
            best_iou = max((iou_xyxy(gbox, ybox) for ybox in yp_boxes), default=0.0)
            if best_iou >= iou_thresh:
                matched += 1
    recall = matched / total if total else None
    return dict(video_id=video_id, gdino_boxes=total, matched=matched,
               recall=round(recall, 3) if recall is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default=str(REPO / "outputs" / "aa1_v2_18clips"))
    args = ap.parse_args()
    clips = [args.clip] if args.clip else (VAL_E3A_IDS if args.all else [])
    if not clips:
        raise SystemExit("pass --clip VIDEO_ID or --all")

    out_dir = Path(args.out_dir)
    print("[setup] loading YOLOPv2 ...")
    yp = YOLOPv2()
    print("[setup] ready")

    comparisons = []
    for vid in clips:
        yolop_frames, _ = run_clip(yp, vid, out_dir)
        cmp = compare_to_gdino(vid, yolop_frames)
        if cmp:
            comparisons.append(cmp)
            print(f"[{vid}] vs G-DINO: recall={cmp['recall']} ({cmp['matched']}/{cmp['gdino_boxes']} boxes matched)")

    if comparisons:
        recalls = [c["recall"] for c in comparisons if c["recall"] is not None]
        print(f"\n[summary] mean recall across {len(recalls)} clips: {np.mean(recalls):.3f}  "
             f"min={min(recalls):.3f}  max={max(recalls):.3f}")
        with open(out_dir / "yolop_vs_gdino_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparisons, f, indent=2)


if __name__ == "__main__":
    main()

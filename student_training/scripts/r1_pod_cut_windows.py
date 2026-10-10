"""
r1_pod_cut_windows.py - cut the window frames of the teacher dataset from the raw Nexar videos (pod side; resumable, sharded).

Reads windows.jsonl (the contract written locally by r1_teacher_windows.py) and writes, for every window,
$R1_TRAIN_DIR/<frames_dir>/frame_00001..00016.jpg at the video's native resolution, JPEG quality 95 (same cutter as the local windows).
Videos come from $NEXAR_VIDEOS/<video_id>.mp4.

    NEXAR_VIDEOS=/root/nexar/train R1_TRAIN_DIR=/root/frames python r1_pod_cut_windows.py --windows /root/teacher_dataset/windows.jsonl [--shard K/N]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_midpoint_negatives import SRC_VIDEOS, WINDOW, read_window_sequential  # noqa: E402
import os  # noqa: E402

TRAIN = Path(os.environ.get("R1_TRAIN_DIR") or "/root/frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--shard", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.windows, encoding="utf-8")]
    byv = collections.defaultdict(list)
    for r in rows:
        byv[r["video_id"]].append(r)
    vids = sorted(byv)
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        vids = [v for i, v in enumerate(vids) if i % n == k]
    print(f"[cut] {len(vids)} videos, source {SRC_VIDEOS}, target {TRAIN}", flush=True)
    t0, done = time.time(), 0
    for i, vid in enumerate(vids):
        todo = [r for r in byv[vid] if len(list((TRAIN / r["frames_dir"]).glob("frame_*.jpg"))) != WINDOW]
        if not todo:
            continue
        cap = cv2.VideoCapture(str(SRC_VIDEOS / f"{vid}.mp4"))
        assert cap.isOpened(), f"cannot open {SRC_VIDEOS / (vid + '.mp4')}"
        try:
            for r in todo:
                frames = read_window_sequential(cap, r["frame_idx"])
                d = TRAIN / r["frames_dir"]
                d.mkdir(parents=True, exist_ok=True)
                for j, fr in enumerate(frames, 1):
                    assert fr.shape[1] == 1280 and fr.shape[0] == 720, (vid, fr.shape)
                    cv2.imwrite(str(d / f"frame_{j:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                done += 1
        finally:
            cap.release()
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(vids)}] windows cut {done}, {time.time() - t0:.0f}s", flush=True)
    bad = [r["frames_dir"] for v in vids for r in byv[v] if len(list((TRAIN / r["frames_dir"]).glob("frame_*.jpg"))) != WINDOW]
    print(f"[cut] done: {done} windows cut, {len(bad)} incomplete", flush=True)
    assert not bad, bad[:5]


if __name__ == "__main__":
    main()

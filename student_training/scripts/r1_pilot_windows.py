"""
r1_pilot_windows.py - select the windows of the boxed-teacher pilot (plan 2026-10-08_Plan-BoxedTeacher-Pilot-rev3)
and cut the missing no-crash windows.

Rules
* Videos come from the Nexar TRAIN videos of the 4,446-window set that are NOT in the 1,761 pool (the pool is enriched with
  A1 failures) and not in the 18 val_e3a clips. Test videos are never touched (train.csv only).
* Crash videos: time-to-alert >= 1.5 s + 8 frames (0.27 s), so all three windows (TTE 1.5 / 1.0 / 0.5) pass the window
  visibility rule; the existing dataset/train/<vid>_hires_tte{05,10,15}/ frames are used.
* Normal videos: three windows ending 0.5 / 1.0 / 1.5 s before a fake event at the video's midpoint + Gaussian noise
  (Nexar test-set protocol, functions reused from build_midpoint_negatives.py); written as dataset/train/<vid>_hires_midtest{05,10,15}/.
* Two disjoint sets A and B, each 9 crash videos x 3 + 8 normal videos x 3 = 51 windows; random with --seed.

    python r1_pilot_windows.py [--dry-run]
      -> outputs/teacher_pilot_2026-10/windows.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_midpoint_negatives import (NEW_BUCKET, SRC_VIDEOS, T_FLOOR, WINDOW, get_video_meta,  # noqa: E402
                                      indices_for, read_window_sequential, video_noise)

OUT = ROOT / "outputs" / "teacher_pilot_2026-10"
TRAIN = ROOT / "dataset" / "train"
MARGIN_S = 8 / 30
N_CRASH, N_NORMAL = 9, 8                       # videos per set


def load_lists():
    t45 = [json.loads(l) for l in open(ROOT / "dataset" / "manifests" / "train4500_hires.jsonl", encoding="utf-8")]
    vids = {str(r["video_id"]).zfill(5) for r in t45}
    pool = {str(json.loads(l)["video_id"]).zfill(5) for l in open(ROOT / "dataset" / "manifests" / "recap_v12_1761.jsonl", encoding="utf-8")}
    t = pd.read_csv(ROOT / "dataset" / "train.csv")
    t["vid"] = t.id.map(lambda x: f"{int(x):05d}")
    t = t[t.vid.isin(vids - pool)]
    crash = t[(t.target == 1) & ((t.time_of_event - t.time_of_alert) >= 1.5 + MARGIN_S)]
    crash = [r for r in crash.itertuples() if all(len(list((TRAIN / f"{r.vid}_hires_tte{s}").glob("frame_*.jpg"))) == WINDOW
                                                   for s in ("05", "10", "15"))]
    normal = [r for r in t[t.target == 0].itertuples() if (SRC_VIDEOS / f"{r.vid}.mp4").exists()]
    return crash, normal, len(vids - pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-std", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    crash, normal, n_out = load_lists()
    print(f"[select] {n_out} videos outside the pool: {len(crash)} crash videos with all 3 windows valid, {len(normal)} normal videos")
    rng = random.Random(args.seed)
    rng.shuffle(crash)
    rng.shuffle(normal)
    assert len(crash) >= 2 * N_CRASH and len(normal) >= 2 * N_NORMAL
    rows = []
    for k, s in enumerate(("A", "B")):
        for r in crash[k * N_CRASH:(k + 1) * N_CRASH]:
            fps, total = get_video_meta(r.vid)
            for tte, suf in ((1.5, "15"), (1.0, "10"), (0.5, "05")):
                end = r.time_of_event - tte
                rows.append({"set": s, "video_id": r.vid, "label": 1, "horizon": tte, "frames_dir": f"{r.vid}_hires_tte{suf}",
                             "t_end_s": round(end, 3), "fps": fps, "frame_idx": indices_for(end, fps, total),
                             "time_of_alert_s": r.time_of_alert, "time_of_event_s": r.time_of_event,
                             "time_to_alert_s": round(r.time_of_event - r.time_of_alert, 3)})
        for r in normal[k * N_NORMAL:(k + 1) * N_NORMAL]:
            fps, total = get_video_meta(r.vid)
            fake = (total / fps) / 2.0 + video_noise(r.vid, args.noise_std, args.seed)
            for grp in (2, 1, 0):                      # 1.5, 1.0, 0.5 s before the fake event
                _, suf, h = NEW_BUCKET[grp]
                end = max(T_FLOOR, fake - h)
                rows.append({"set": s, "video_id": r.vid, "label": 0, "horizon": h, "frames_dir": f"{r.vid}_hires_{suf}",
                             "t_end_s": round(end, 3), "fps": fps, "frame_idx": indices_for(end, fps, total),
                             "fake_event_s": round(fake, 3), "floored": fake - h < T_FLOOR})
    print(f"[select] {len(rows)} windows: " + ", ".join(f"set {s}: {sum(r['set'] == s for r in rows)}" for s in ("A", "B")))
    if args.dry_run:
        for r in rows[:4]:
            print(r)
        return
    n_cut = 0
    for r in rows:
        d = TRAIN / r["frames_dir"]
        if d.exists() and len(list(d.glob("frame_*.jpg"))) == WINDOW:
            continue
        assert r["label"] == 0, f"missing crash frames {d}"
        cap = cv2.VideoCapture(str(SRC_VIDEOS / f"{r['video_id']}.mp4"))
        try:
            frames = read_window_sequential(cap, r["frame_idx"])
        finally:
            cap.release()
        d.mkdir(parents=True, exist_ok=True)
        for j, fr in enumerate(frames, 1):
            cv2.imwrite(str(d / f"frame_{j:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_cut += 1
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "windows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[done] cut {n_cut} new no-crash windows; wrote {OUT / 'windows.jsonl'}")


if __name__ == "__main__":
    main()

"""
r1_teacher_windows.py - Step A of the full teacher run (plan 2026-10-10_Plan-Teacher-Full-Run.md).

Writes the 4,203-window manifest of the teacher dataset and cuts the window frames that are missing locally.

Pool: all 1,500 Nexar training videos (750 crash + 750 normal) minus the crash videos in which NO window passes the visibility rule
(alert-to-collision lead < 0.5 s + 0.27 s; 99 videos incl. 00319 of the 18 former `val_e3a` clips). 1,401 videos x 3 windows.
  crash videos : windows end 1.5 / 1.0 / 0.5 s before time_of_event   -> dataset/train/<vid>_hires_tte{15,10,05}
  normal videos: Nexar test-set protocol: fake event = video midpoint + per-video Gaussian noise (std 1.0, seed 0, sha256-seeded), windows
                 end 1.5 / 1.0 / 0.5 s before it                       -> dataset/train/<vid>_hires_midtest{15,10,05}
Per window: label (Nexar event label), valid (crash window passes the rule: end >= time_of_alert + 8/30 s), pre_alert (= crash and not
valid), label_vis default (normal 0, crash valid 1, pre-alert 0; E7-pre may raise a pre-alert one to 1 later), frame_idx, fps, t_end_s.
Checks: crash frame_idx equal to dataset/manifests/train4500_hires.jsonl wherever the pool has the window; existing folders have 16 frames;
a random sample of existing folders passes the pixel alignment check against the raw video.

    python r1_teacher_windows.py --dry-run     # counts + checks, nothing written
    python r1_teacher_windows.py               # writes outputs/teacher_dataset_2026-10/windows.jsonl and cuts the missing windows
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_midpoint_negatives import (NEW_BUCKET, SRC_VIDEOS, T_FLOOR, WINDOW, get_video_meta, indices_for,  # noqa: E402
                                      read_window_sequential, video_noise)

TRAIN = ROOT / "dataset" / "train"
OUT = ROOT / "outputs" / "teacher_dataset_2026-10"
MARGIN_S = 8 / 30
SUF = {1.5: "15", 1.0: "10", 0.5: "05"}


def jpg_count(d):
    return len(list(d.glob("frame_*.jpg"))) if d.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--align-sample", type=int, default=40)
    args = ap.parse_args()
    t = pd.read_csv(ROOT / "dataset" / "train.csv")
    t["vid"] = t.id.map(lambda x: f"{int(x):05d}")
    pool_manifest = {}
    for l in open(ROOT / "dataset" / "manifests" / "train4500_hires.jsonl", encoding="utf-8"):
        r = json.loads(l)
        pool_manifest[r["frames_dir"]] = r.get("t_seconds")            # window end time (frame_indices there are positions 1..16)

    rows, drop = [], []
    for r in t.itertuples():
        fps, total = get_video_meta(r.vid)
        if r.target == 1:
            if r.time_of_event - r.time_of_alert < 0.5 + MARGIN_S:
                drop.append(r.vid)
                continue
            for tte in (1.5, 1.0, 0.5):
                end = max(T_FLOOR, r.time_of_event - tte)                  # a 2-s window needs 2 s of video (same floor as the old pool)
                valid = bool(end >= r.time_of_alert + MARGIN_S)
                rows.append({"set": "full", "video_id": r.vid, "label": 1, "horizon": tte, "frames_dir": f"{r.vid}_hires_tte{SUF[tte]}",
                             "t_end_s": round(end, 3), "fps": fps, "frame_idx": indices_for(end, fps, total),
                             "time_of_alert_s": r.time_of_alert, "time_of_event_s": r.time_of_event,
                             "time_to_alert_s": round(r.time_of_event - r.time_of_alert, 3), "valid": valid, "pre_alert": not valid,
                             "floored": bool(r.time_of_event - tte < T_FLOOR), "tte_effective_s": round(r.time_of_event - end, 3),
                             "label_vis": 1 if valid else 0})
        else:
            fake = (total / fps) / 2.0 + video_noise(r.vid, args.noise_std, args.seed)
            for grp in (2, 1, 0):
                _, suf, h = NEW_BUCKET[grp]
                end = max(T_FLOOR, fake - h)
                rows.append({"set": "full", "video_id": r.vid, "label": 0, "horizon": h, "frames_dir": f"{r.vid}_hires_{suf}",
                             "t_end_s": round(end, 3), "fps": fps, "frame_idx": indices_for(end, fps, total),
                             "fake_event_s": round(fake, 3), "floored": bool(fake - h < T_FLOOR), "valid": True, "pre_alert": False,
                             "label_vis": 0})

    # ---- counts asserted against the plan
    c = collections.Counter((r["label"], r["pre_alert"]) for r in rows)
    kept = {h: sum(1 for r in rows if r["label"] == 1 and r["valid"] and r["horizon"] == h) for h in (0.5, 1.0, 1.5)}
    print(f"[windows] {len(rows)} windows | crash kept {c[(1, False)]} | crash pre-alert {c[(1, True)]} | normal {c[(0, False)]} | "
          f"videos dropped (no kept window): {len(drop)} | kept per TTE 0.5/1.0/1.5 = {kept[0.5]}/{kept[1.0]}/{kept[1.5]}")
    assert len(rows) == 4203 and c[(1, False)] == 1343 and c[(1, True)] == 610 and c[(0, False)] == 2250 and len(drop) == 99, "plan counts changed"
    assert kept == {0.5: 651, 1.0: 439, 1.5: 253}
    assert len({r["frames_dir"] for r in rows}) == len(rows)
    print(f"[windows] windows floored at {T_FLOOR}s: normal {sum(r.get('floored', False) for r in rows if r['label'] == 0)}, crash {sum(r.get('floored', False) for r in rows if r['label'] == 1)}")

    # ---- consistency checks on what already exists
    mism = [r["frames_dir"] for r in rows if r["frames_dir"] in pool_manifest and pool_manifest[r["frames_dir"]] is not None
            and abs(pool_manifest[r["frames_dir"]] - r["t_end_s"]) > 0.006]
    n_pool = sum(r["frames_dir"] in pool_manifest for r in rows)
    print(f"[check] {n_pool} windows are in the old pool manifest (crash TTE windows); window end-time mismatches: {len(mism)} {mism[:3]}")
    assert not mism, "crash window end times differ from train4500_hires.jsonl"
    have = [r for r in rows if jpg_count(TRAIN / r["frames_dir"]) == WINDOW]
    bad = [r["frames_dir"] for r in rows if 0 < jpg_count(TRAIN / r["frames_dir"]) < WINDOW]
    missing = [r for r in rows if jpg_count(TRAIN / r["frames_dir"]) == 0]
    print(f"[check] frame folders: {len(have)} complete, {len(missing)} missing, {len(bad)} incomplete {bad[:3]}")
    assert not bad
    rng = random.Random(0)
    worst = 0.0
    for r in rng.sample(have, min(args.align_sample, len(have))):
        cap = cv2.VideoCapture(str(SRC_VIDEOS / f"{r['video_id']}.mp4"))
        for pos, k in ((r["frame_idx"][-1], 15), (r["frame_idx"][0], 0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, raw = cap.read()
            jp = cv2.imread(str(TRAIN / r["frames_dir"] / f"frame_{k + 1:05d}.jpg"))
            d = float(np.abs(raw.astype(np.float32) - jp.astype(np.float32)).mean()) if ok else 99.0
            worst = max(worst, d)
            assert d < 4.0, f"{r['frames_dir']} frame {k + 1}: alignment diff {d:.1f}"
        cap.release()
    print(f"[check] pixel alignment of {min(args.align_sample, len(have))} existing windows against the raw video: worst mean diff {worst:.2f}")
    kinds = collections.Counter(("crash" if r["label"] else "normal") for r in missing)
    print(f"[cut] {len(missing)} windows to cut locally: {dict(kinds)}")
    if args.dry_run:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    byv = collections.defaultdict(list)
    for r in missing:
        byv[r["video_id"]].append(r)
    for vid, ws in byv.items():
        cap = cv2.VideoCapture(str(SRC_VIDEOS / f"{vid}.mp4"))
        try:
            for r in ws:
                frames = read_window_sequential(cap, r["frame_idx"])
                d = TRAIN / r["frames_dir"]
                d.mkdir(parents=True, exist_ok=True)
                for j, fr in enumerate(frames, 1):
                    cv2.imwrite(str(d / f"frame_{j:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
        finally:
            cap.release()
        if n % 200 < len(ws):
            print(f"  cut {n}/{len(missing)}", flush=True)
    assert all(jpg_count(TRAIN / r["frames_dir"]) == WINDOW for r in rows), "some windows still incomplete"
    with open(OUT / "windows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (OUT / "dropped_videos.json").write_text(json.dumps({"reason": "crash video with no window passing the visibility rule "
                                                         "(alert-to-collision lead < 0.77 s)", "videos": sorted(drop)}, indent=1), encoding="utf-8")
    print(f"[done] wrote {len(rows)} windows -> {OUT / 'windows.jsonl'}; cut {n} windows; dropped videos -> dropped_videos.json")


if __name__ == "__main__":
    main()

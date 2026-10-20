"""
r1_extra_windows.py - extra crash windows at TTE 0.75 s and 1.25 s (plan 2026-10-19_Plan-Extra-TTE-Windows-Retrain, step 1).

Same rules as the existing windows (r1_teacher_windows.py): a crash video is kept if the alert-to-event lead is >= 0.5 s + 8/30 s (651 of 750); a crash window ends
max(2.0 s, time_of_event - TTE) seconds into the video, 16 frames at stride 4 (build_midpoint_negatives.indices_for); it is VALID (the hazard is visible) if
its end >= time_of_alert + 8/30 s. Only valid windows are written (pre-alert ones are not used). A new window inherits the split of its video in the reasoning manifest
(r1_nexar_reasoning_windows.jsonl): no video crosses splits; videos outside that manifest are skipped.

  python r1_extra_windows.py --dry-run     counts + the check that the rule reproduces the existing 651 / 439 / 253 windows (nothing written)
  python r1_extra_windows.py               writes outputs/teacher_dataset_2026-10/windows_extra.jsonl and cuts the frames to dataset/train/<vid>_hires_tte075 / _tte125
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
from build_midpoint_negatives import SRC_VIDEOS, T_FLOOR, WINDOW, get_video_meta, indices_for, read_window_sequential  # noqa: E402

TRAIN = ROOT / "dataset" / "train"
OUT = ROOT / "outputs" / "teacher_dataset_2026-10"
MARGIN_S = 8 / 30
NEW_TTES = {0.75: "075", 1.25: "125"}
EXISTING_KEPT = {0.5: 651, 1.0: 439, 1.5: 253}                  # the plan's counts of the existing windows


def jpg_count(d):
    return len(list(d.glob("frame_*.jpg"))) if d.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--align-sample", type=int, default=30)
    a = ap.parse_args()
    t = pd.read_csv(ROOT / "dataset" / "train.csv")
    t["vid"] = t.id.map(lambda x: f"{int(x):05d}")
    split_of = {}
    for l in open(ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl", encoding="utf-8"):
        r = json.loads(l)
        split_of[r["video_key"]] = r["split"]
    crash = t[(t.target == 1) & (t.time_of_event - t.time_of_alert >= 0.5 + MARGIN_S)]
    print(f"[videos] {len(crash)} crash videos pass the lead rule (expect 651)")
    assert len(crash) == 651

    def valid(r, tte):
        return max(T_FLOOR, r.time_of_event - tte) >= r.time_of_alert + MARGIN_S

    got = {tte: sum(valid(r, tte) for r in crash.itertuples()) for tte in EXISTING_KEPT}
    print(f"[check] the rule reproduces the existing windows: {got} (expect {EXISTING_KEPT})")
    assert got == EXISTING_KEPT, "visibility rule differs from the one that built the existing windows"

    rows, skipped = [], collections.Counter()
    for r in crash.itertuples():
        sp = split_of.get(r.vid)
        for tte, suf in NEW_TTES.items():
            if not valid(r, tte):
                skipped["pre-alert"] += 1
                continue
            if sp is None:
                skipped["video not in the reasoning manifest"] += 1
                continue
            fps, total = get_video_meta(r.vid)
            end = max(T_FLOOR, r.time_of_event - tte)
            rows.append({"set": "extra", "split": sp, "video_id": r.vid, "label": 1, "horizon": tte, "frames_dir": f"{r.vid}_hires_tte{suf}", "t_end_s": round(end, 3),
                         "fps": fps, "frame_idx": indices_for(end, fps, total), "time_of_alert_s": r.time_of_alert, "time_of_event_s": r.time_of_event,
                         "time_to_alert_s": round(r.time_of_event - r.time_of_alert, 3), "valid": True, "pre_alert": False,
                         "floored": bool(r.time_of_event - tte < T_FLOOR), "tte_effective_s": round(r.time_of_event - end, 3), "label_vis": 1})
    c = collections.Counter((r["horizon"], r["split"]) for r in rows)
    print(f"[windows] {len(rows)} new windows: " + ", ".join(f"TTE {k[0]} {k[1]} {v}" for k, v in sorted(c.items())) + f" | skipped {dict(skipped)}")
    assert c[(0.75, "train")] == 467 and c[(1.25, "train")] == 290 and c[(0.75, "val")] == 74 and c[(1.25, "val")] == 43, "counts differ from the plan"
    assert len({r["frames_dir"] for r in rows}) == len(rows)
    print(f"[windows] floored at {T_FLOOR} s: {sum(r['floored'] for r in rows)}")
    have = [r for r in rows if jpg_count(TRAIN / r["frames_dir"]) == WINDOW]
    missing = [r for r in rows if jpg_count(TRAIN / r["frames_dir"]) == 0]
    bad = [r["frames_dir"] for r in rows if 0 < jpg_count(TRAIN / r["frames_dir"]) < WINDOW]
    print(f"[check] frame folders: {len(have)} complete, {len(missing)} to cut, {len(bad)} incomplete {bad[:3]}")
    assert not bad
    if a.dry_run:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    byv = collections.defaultdict(list)
    for r in missing:
        byv[r["video_id"]].append(r)
    n = 0
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
        if n % 100 < len(ws):
            print(f"  cut {n}/{len(missing)}", flush=True)
    assert all(jpg_count(TRAIN / r["frames_dir"]) == WINDOW for r in rows), "some windows are incomplete"
    rng, worst = random.Random(0), 0.0                                     # pixel alignment against the raw video (same check as the existing windows)
    for r in rng.sample(rows, min(a.align_sample, len(rows))):
        cap = cv2.VideoCapture(str(SRC_VIDEOS / f"{r['video_id']}.mp4"))
        for pos, k in ((r["frame_idx"][-1], 15), (r["frame_idx"][0], 0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, raw = cap.read()
            jp = cv2.imread(str(TRAIN / r["frames_dir"] / f"frame_{k + 1:05d}.jpg"))
            d = float(np.abs(raw.astype(np.float32) - jp.astype(np.float32)).mean()) if ok else 99.0
            worst = max(worst, d)
            assert d < 4.0, f"{r['frames_dir']} frame {k + 1}: alignment diff {d:.1f}"
        cap.release()
    (OUT / "windows_extra.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"[done] {len(rows)} windows -> {OUT / 'windows_extra.jsonl'}; cut {n}; pixel alignment of {min(a.align_sample, len(rows))} windows: worst mean diff {worst:.2f}")


if __name__ == "__main__":
    main()

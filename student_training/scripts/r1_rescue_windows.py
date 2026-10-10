"""
r1_rescue_windows.py - pick the videos for the E6 rescue test and write their windows (set "R") to a separate folder.

Rescue test: can the teacher flow (E3 -> E6 / E7 -> chain) give grounded explanations for crash windows that the visibility rule
removes (window end < time_of_alert + 0.27 s)? Groups by alert-to-event lead:
  only TTE 0.5 kept  : lead in [0.77, 1.27) s  -> TTE 1.0 and 1.5 windows are "removed"
  TTE 0.5 + 1.0 kept : lead in [1.27, 1.77) s  -> TTE 1.5 window is "removed"
All 3 windows of each video are written; `valid` marks whether the visibility rule keeps the window (anchors must be valid).
Videos come from the 4,446-window pool, are not pilot videos, and must have all three frame folders in dataset/train.

    python r1_rescue_windows.py --per-group 5 --seed 0      # -> outputs/teacher_rescue_2026-10/windows.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_midpoint_negatives import get_video_meta, indices_for  # noqa: E402

TRAIN = ROOT / "dataset" / "train"
PILOT = ROOT / "outputs" / "teacher_pilot_2026-10"
OUT = ROOT / "outputs" / "teacher_rescue_2026-10"
MARGIN_S = 8 / 30
GROUPS = {"only_0.5_kept": (0.5 + MARGIN_S, 1.0 + MARGIN_S), "0.5_and_1.0_kept": (1.0 + MARGIN_S, 1.5 + MARGIN_S)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pool = {json.loads(l)["video_id"] for l in open(ROOT / "dataset" / "manifests" / "train4500_hires.jsonl", encoding="utf-8")}
    pilot = {json.loads(l)["video_id"] for l in open(PILOT / "windows.jsonl", encoding="utf-8")}
    t = pd.read_csv(ROOT / "dataset" / "train.csv")
    t["vid"] = t.id.map(lambda x: f"{int(x):05d}")
    t = t[(t.target == 1) & t.vid.isin(pool) & ~t.vid.isin(pilot)].copy()
    t["lead"] = t.time_of_event - t.time_of_alert
    rng = random.Random(args.seed)
    rows = []
    for g, (lo, hi) in GROUPS.items():
        cand = [r for r in t[(t.lead >= lo) & (t.lead < hi)].itertuples()
                if all(len(list((TRAIN / f"{r.vid}_hires_tte{s}").glob("frame_*.jpg"))) == 16 for s in ("05", "10", "15"))]
        rng.shuffle(cand)
        for r in sorted(cand[: args.per_group], key=lambda r: r.vid):
            fps, total = get_video_meta(r.vid)
            for tte, suf in ((1.5, "15"), (1.0, "10"), (0.5, "05")):
                end = r.time_of_event - tte
                rows.append({"set": "R", "group": g, "video_id": r.vid, "label": 1, "horizon": tte, "frames_dir": f"{r.vid}_hires_tte{suf}",
                             "t_end_s": round(end, 3), "fps": fps, "frame_idx": indices_for(end, fps, total),
                             "time_of_alert_s": r.time_of_alert, "time_of_event_s": r.time_of_event,
                             "time_to_alert_s": round(r.lead, 3), "valid": bool(end >= r.time_of_alert + MARGIN_S)})
        print(f"[{g}] {len(cand)} candidates; picked {[r.vid for r in sorted(cand[: args.per_group], key=lambda r: r.vid)]}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "windows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[done] {len(rows)} windows ({sum(r['valid'] for r in rows)} valid, {sum(not r['valid'] for r in rows)} removed by the rule) -> {OUT}")
    for r in rows:
        print(f"  {r['frames_dir']}  lead {r['time_to_alert_s']:.2f}s  valid={r['valid']}")


if __name__ == "__main__":
    main()

"""
r1_shift_windows.py - shifted copies of the Nexar reasoning windows (plan 2026-10-14_Plan-Phase1-Enlarged-Dataset, section 1b).

For every window of the reasoning dataset (train and val) make up to 3 copies whose 16 source-frame indices are all moved EARLIER by s = 1, 2, 3
frames (<= 0.1 s). The frames are every 4th frame, so each copy is a different 16-frame sample of the same scene; the label, the facts and every
text version stay valid. Rules:
  * normal windows and pre-alert windows with label_vis 0: always kept (earlier = still not visible / still normal)
  * crash windows with label_vis 1: kept only if the NEW end time is still >= time_of_alert + 8/30 s (the visibility rule); else dropped
  * pre-alert windows whose label_vis was raised to 1 by hindsight: no shifts
  * first frame index must stay >= 0
Writes (outputs/teacher_dataset_2026-10):
  windows_shifted.jsonl   one row per copy for the cutter (video_id, frames_dir = <orig>_s<k>, frame_idx, ...), same fields as windows.jsonl
and the training manifest dataset/manifests/r1_nexar_reasoning_shifted.jsonl (same format as r1_nexar_reasoning_windows.jsonl; id = <orig id>_s<k>,
extra field `source_id`, `shift`). Prints the kept / dropped counts per reason.

    python r1_shift_windows.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "teacher_dataset_2026-10"
MAN = ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl"
OUT_MAN = ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_shifted.jsonl"
CTRL_MAN = ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_control.jsonl"      # 24 ORIGINAL windows re-cut on the pod: their crash-model P must reproduce
MARGIN_S = 8 / 30
SHIFTS = (1, 2, 3)


def main():
    wins = {}
    for l in open(D / "windows.jsonl", encoding="utf-8"):
        w = json.loads(l)
        wins[w["frames_dir"]] = w
    man = [json.loads(l) for l in open(MAN, encoding="utf-8")]
    rows, mrows, drops = [], [], collections.Counter()
    for m in man:
        w = wins[m["frames_dir"]]
        g = m["gt"]
        for s in SHIFTS:
            if m["label"] == 1 and g["pre_alert"] and g["label_vis"] == 1:
                drops["pre-alert raised to 1 by hindsight"] += 1
                continue
            idx = [i - s for i in w["frame_idx"]]
            if idx[0] < 0:
                drops["first frame < 0"] += 1
                continue
            t_end = round(w["t_end_s"] - s / w["fps"], 4)
            if m["label"] == 1 and g["label_vis"] == 1 and t_end < w["time_of_alert_s"] + MARGIN_S - 1e-9:
                drops["visibility rule (crash, label_vis 1)"] += 1
                continue
            fd = f"{m['frames_dir']}_s{s}"
            rows.append({**{k: v for k, v in w.items() if k not in ("frame_idx", "t_end_s", "frames_dir")}, "frames_dir": fd, "source_frames_dir": m["frames_dir"],
                         "shift": s, "t_end_s": t_end, "frame_idx": idx})
            mrows.append({**m, "id": f"{m['id']}_s{s}", "frames_dir": fd, "source_id": m["id"], "shift": s})
    (D / "windows_shifted.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    OUT_MAN.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mrows), encoding="utf-8")
    pick = ([m for m in man if m["split"] == "train" and m["label"] == 1][:8] + [m for m in man if m["split"] == "train" and m["label"] == 0][:8]
            + [m for m in man if m["split"] == "val"][:8])
    crow = [{**{k: v for k, v in wins[m["frames_dir"]].items() if k != "frames_dir"}, "frames_dir": m["frames_dir"] + "_c", "source_frames_dir": m["frames_dir"],
             "shift": 0} for m in pick]
    cman = [{**m, "id": m["id"] + "_c", "frames_dir": m["frames_dir"] + "_c", "source_id": m["id"], "shift": 0} for m in pick]
    CTRL_MAN.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cman), encoding="utf-8")
    (D / "windows_control.jsonl").write_text("".join(json.dumps(r) + "\n" for r in crow), encoding="utf-8")
    per = collections.Counter(r["source_id"] for r in mrows)
    full = sum(1 for m in man if per[m["id"]] == len(SHIFTS))
    sp = collections.Counter(r["split"] for r in mrows)
    print(f"[shift] {len(man)} windows -> {len(mrows)} shifted copies ({dict(sp)}); windows with all {len(SHIFTS)} shifts: {full}; "
          f"with none: {sum(1 for m in man if per[m['id']] == 0)}")
    print("        dropped:", dict(drops))


if __name__ == "__main__":
    main()

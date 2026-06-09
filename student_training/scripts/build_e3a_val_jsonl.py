"""
build_e3a_val_jsonl.py
======================
Build the E3a validation JSONL from the 18 hand-annotated GT clips.

Source : dataset/teacher_dataset_GT_self_imply.xlsx
Output : outputs/val_e3a.jsonl

These 18 clips carry HUMAN ground-truth reasoning (column verdict_reasoning_en),
which makes them the held-out reference for the LLM-as-judge reasoning metric
(cause-match / object-match / hallucination). They are excluded from the train
set by build_e3a_train_jsonl.py (leakage guard).

Each output record mirrors the train schema:

  video_id          clean id, e.g. "00319"
  frames_dir        "{video_id}_hires"     (1280x720 source; downsampled to 448)
  frame_indices     [1..16]                (hires dirs renumbered 1..16)
  window_size       16
  target            int 0/1
  gt_verdict        "YES"/"NO"
  final_verdict     == gt_verdict          (GT clips are correct by definition)
  requested_time_to_event   0.5 / 1.0 / 1.5 / "TN_MIDPOINT"
  gt_reasoning_en   human GT reasoning (English) — reference for LLM-as-judge
  assistant_target  JSON string: {"verdict": <gt_verdict>, "reason": <gt_reasoning_en>}

Run:
  python student_training/scripts/build_e3a_val_jsonl.py
"""

import json
import os
import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC  = REPO / "dataset/teacher_dataset_GT_self_imply.xlsx"
FRAMES_BASE = REPO / "dataset/train"
OUT  = REPO / "dataset/manifests/val_e3a.jsonl"

WINDOW_SIZE = 16


def _norm_id(v) -> str:
    s = str(v).strip()
    return s.zfill(5) if s.isdigit() else s


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    df = pd.read_excel(SRC)
    print(f"Source rows: {len(df)}")

    out_records = []
    missing_frames = []
    for _, row in df.iterrows():
        video_id = _norm_id(row["video_id"])
        gt = str(row["gt_verdict"]).strip()
        reason = str(row.get("verdict_reasoning_en") or "").strip()

        if gt not in ("YES", "NO"):
            raise ValueError(f"{video_id}: unexpected gt_verdict={gt!r}")
        if not reason:
            raise ValueError(f"{video_id}: empty verdict_reasoning_en")

        # Verify hi-res frames exist
        hires_dir = FRAMES_BASE / f"{video_id}_hires"
        n = len(glob.glob(str(hires_dir / "frame_*.jpg"))) if hires_dir.is_dir() else 0
        if n < WINDOW_SIZE:
            missing_frames.append((video_id, n))

        assistant_target = json.dumps(
            {"verdict": gt, "reason": reason},
            ensure_ascii=False,
        )

        out_records.append({
            "video_id":                video_id,
            "frames_dir":              f"{video_id}_hires",
            "frame_indices":           list(range(1, WINDOW_SIZE + 1)),
            "window_size":             WINDOW_SIZE,
            "target":                  1 if gt == "YES" else 0,
            "gt_verdict":              gt,
            "final_verdict":           gt,
            "requested_time_to_event": row.get("requested_time_to_event"),
            "gt_reasoning_en":         reason,
            "assistant_target":        assistant_target,
        })

    if missing_frames:
        raise FileNotFoundError(
            f"{len(missing_frames)} val clip(s) missing hi-res frames: {missing_frames}"
        )

    n_pos = sum(x["target"] for x in out_records)
    n_neg = len(out_records) - n_pos

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for x in out_records:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_records)} records -> {OUT}")
    print(f"  class balance : {n_pos} YES / {n_neg} NO")


if __name__ == "__main__":
    main()

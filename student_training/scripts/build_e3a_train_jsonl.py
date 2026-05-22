"""
build_e3a_train_jsonl.py
========================
Build the E3a student-training JSONL from the resampled teacher bake-off.

Source : outputs/prompt_bakeoff/v11_100clips_resampled/final_combined.jsonl
Filter : keep only records with passes_for_student_training == True  (→ 90 clips)
Output : outputs/teacher_dataset_e3a.jsonl

Each output record carries exactly what student_training/data/collision_dataset.py
needs, plus the minimal PROMPT_S assistant target:

  video_id          clean id, e.g. "01045"      (identity / leakage checks)
  frames_dir        "{video_id}_hires"          (the AUTHORITATIVE 1280x720 window;
                                                 the plain 256px dirs are STALE for
                                                 resampled-FP clips — see plan)
  frame_indices     [1..16]                     (hires dirs are renumbered 1..16)
  window_size       16
  target            1 if gt_verdict=="YES" else 0
  gt_verdict        "YES"/"NO"
  final_verdict     teacher final verdict (== gt_verdict for passing clips)
  requested_time_to_event   0.5 / 1.0 / 1.5 / "TN_MIDPOINT"
  source            v11_preserved / v11_resampled_fp / v11_fn_v7_1
  assistant_target  JSON string: {"verdict": <final_verdict>, "reason": <teacher_reasoning_final>}

Why no reasoning-compression step: teacher_reasoning_final already fits the
PROMPT_S budget (mean ~80 tokens; only 1/90 clips marginally over at ~152),
so we use it verbatim as the `reason`.

Run:
  python student_training/scripts/build_e3a_train_jsonl.py
"""

import json
from pathlib import Path

import pandas as pd

REPO    = Path(__file__).resolve().parents[2]
SRC     = REPO / "outputs/prompt_bakeoff/v11_100clips_resampled/final_combined.jsonl"
VAL_XLSX = REPO / "dataset/teacher_dataset_GT_self_imply.xlsx"
OUT     = REPO / "outputs/teacher_dataset_e3a.jsonl"

WINDOW_SIZE = 16


def _norm_id(v) -> str:
    """Normalise a video id to the 5-digit zero-padded string used on disk."""
    s = str(v).strip()
    return s.zfill(5) if s.isdigit() else s


def load_val_ids() -> set:
    """Video ids reserved for the validation set — must be excluded from train
    to avoid leakage (the 18 GT clips are the held-out val set)."""
    if not VAL_XLSX.exists():
        raise FileNotFoundError(VAL_XLSX)
    df = pd.read_excel(VAL_XLSX)
    return {_norm_id(v) for v in df["video_id"]}


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    records = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    passing = [r for r in records if r.get("passes_for_student_training")]

    # Leakage guard: drop any passing clip whose video_id is reserved for val
    val_ids = load_val_ids()
    before = len(passing)
    excluded = [str(r["video_id"]) for r in passing if _norm_id(r["video_id"]) in val_ids]
    passing = [r for r in passing if _norm_id(r["video_id"]) not in val_ids]
    print(f"Source records: {len(records)}  |  passing: {before}")
    if excluded:
        print(f"  LEAKAGE GUARD: excluded {len(excluded)} val-overlap clip(s) from train: {excluded}")
    print(f"  train after guard: {len(passing)}")

    out_records = []
    over_budget = []
    for r in passing:
        video_id = str(r["video_id"])
        gt = r.get("gt_verdict")
        final_verdict = r.get("final_verdict")
        reason = (r.get("teacher_reasoning_final") or "").strip()

        if gt not in ("YES", "NO"):
            raise ValueError(f"{video_id}: unexpected gt_verdict={gt!r}")
        if final_verdict not in ("YES", "NO"):
            raise ValueError(f"{video_id}: unexpected final_verdict={final_verdict!r}")
        if not reason:
            raise ValueError(f"{video_id}: empty teacher_reasoning_final")

        # Flag (but keep) any clip whose reason exceeds ~150 tokens
        if len(reason) // 4 > 150:
            over_budget.append((video_id, len(reason) // 4))

        assistant_target = json.dumps(
            {"verdict": final_verdict, "reason": reason},
            ensure_ascii=False,
        )

        out_records.append({
            "video_id":                video_id,
            "frames_dir":              f"{video_id}_hires",
            "frame_indices":           list(range(1, WINDOW_SIZE + 1)),
            "window_size":             WINDOW_SIZE,
            "target":                  1 if gt == "YES" else 0,
            "gt_verdict":              gt,
            "final_verdict":           final_verdict,
            "requested_time_to_event": r.get("requested_time_to_event"),
            "source":                  r.get("source"),
            "assistant_target":        assistant_target,
        })

    # Sanity report
    n_pos = sum(x["target"] for x in out_records)
    n_neg = len(out_records) - n_pos
    from collections import Counter
    src_counts = Counter(x["source"] for x in out_records)
    tte_counts = Counter(str(x["requested_time_to_event"]) for x in out_records)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for x in out_records:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_records)} records -> {OUT}")
    print(f"  class balance : {n_pos} YES / {n_neg} NO")
    print(f"  by source     : {dict(src_counts)}")
    print(f"  by TTE        : {dict(tte_counts)}")
    if over_budget:
        print(f"  NOTE: {len(over_budget)} clip(s) over ~150 tokens (kept verbatim): {over_budget}")


if __name__ == "__main__":
    main()

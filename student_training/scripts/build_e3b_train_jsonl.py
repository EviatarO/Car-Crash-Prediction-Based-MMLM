"""
build_e3b_train_jsonl.py
========================
Build the e3b multi-horizon student-training JSONL (267 clips) from the TTE-fill
distillation, mirroring the schema of build_e3a_train_jsonl.py.

Sources:
  - results_e3a_tte_fill.xlsx / sheet `per_clip` (267 rows): the authoritative table.
      A=video_id, B=gt_verdict, E=horizon_label, U=final_verdict,
      Z=final_reasoning2 (SFT reason), X=row_origin.
  - extraction_log.json (178 new clips): (video_id, new_horizon_label) -> frames_subdir.
      Existing-89 clips reuse the original "{video_id}_hires" dir.

Output: dataset/teacher_labels/teacher_dataset_e3b.jsonl

Each record carries exactly what collision_dataset.py needs plus the minimal
PROMPT_S assistant target:
  video_id, frames_dir, frame_indices[1..16], window_size, target(0/1),
  gt_verdict, final_verdict, horizon_label, requested_time_to_event, row_origin,
  assistant_target = {"verdict": final_verdict, "reason": final_reasoning2}

Run:
  python student_training/scripts/build_e3b_train_jsonl.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
TTE = REPO / "outputs/prompt_bakeoff/e3a_tte_fill"
XLSX = TTE / "results_e3a_tte_fill.xlsx"
EXTRACT_LOG = TTE / "extraction_log.json"
FRAMES_ROOT = REPO / "dataset/train"
VAL_JSONL = REPO / "dataset/manifests/val_e3a.jsonl"
OUT = REPO / "dataset/teacher_labels/teacher_dataset_e3b.jsonl"

WINDOW_SIZE = 16


def norm_id(v) -> str:
    s = str(v).strip()
    return s.zfill(5) if s.replace(".0", "").isdigit() else s


def load_val_ids() -> set:
    ids = set()
    for line in open(VAL_JSONL, encoding="utf-8"):
        line = line.strip()
        if line:
            ids.add(norm_id(json.loads(line)["video_id"]))
    return ids


def main():
    # ---- per_clip ----
    df = pd.read_excel(XLSX, sheet_name="per_clip")
    df["vid"] = df["video_id"].apply(norm_id)
    assert len(df) == 267, f"expected 267 rows, got {len(df)}"

    # ---- extraction_log: (vid, horizon) -> frames_subdir ----
    el = json.load(open(EXTRACT_LOG, encoding="utf-8"))
    sub_lookup = {}
    for r in el:
        key = (norm_id(r["video_id"]), r["new_horizon_label"])
        sub_lookup[key] = r["frames_subdir"]
    print(f"extraction_log: {len(el)} new clips, {len(sub_lookup)} unique keys")

    out_records = []
    missing_frames = []
    missing_join = []
    for _, row in df.iterrows():
        vid = row["vid"]
        horizon = row["horizon_label"]
        origin = row["row_origin"]
        gt = str(row["gt_verdict"]).strip().upper()
        final_verdict = str(row["final_verdict"]).strip().upper()
        reason = str(row["final_reasoning2"]).strip()

        if gt not in ("YES", "NO"):
            raise ValueError(f"{vid}/{horizon}: bad gt_verdict={gt!r}")
        if final_verdict not in ("YES", "NO"):
            raise ValueError(f"{vid}/{horizon}: bad final_verdict={final_verdict!r}")
        if not reason or reason.lower() == "nan":
            raise ValueError(f"{vid}/{horizon}: empty final_reasoning2")

        # frames_dir
        if origin == "existing_89":
            frames_dir = f"{vid}_hires"
        else:  # new_178
            key = (vid, horizon)
            if key not in sub_lookup:
                missing_join.append(key)
                continue
            frames_dir = sub_lookup[key]

        if not (FRAMES_ROOT / frames_dir).is_dir():
            missing_frames.append(frames_dir)

        assistant_target = json.dumps(
            {"verdict": final_verdict, "reason": reason}, ensure_ascii=False
        )
        out_records.append({
            "video_id": vid,
            "frames_dir": frames_dir,
            "frame_indices": list(range(1, WINDOW_SIZE + 1)),
            "window_size": WINDOW_SIZE,
            "target": 1 if gt == "YES" else 0,
            "gt_verdict": gt,
            "final_verdict": final_verdict,
            "horizon_label": horizon,
            "requested_time_to_event": row.get("requested_time_to_event"),
            "row_origin": origin,
            "assistant_target": assistant_target,
        })

    # ---- leakage guard (train vs VAL only; train/test are different Nexar pools) ----
    val_ids = load_val_ids()
    train_ids = {r["video_id"] for r in out_records}
    leak = train_ids & val_ids
    print(f"\nLeakage guard: {len(train_ids)} unique train videos; "
          f"train ∩ val = {len(leak)} {sorted(leak) if leak else ''}")

    # ---- write ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in out_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report ----
    n_pos = sum(r["target"] for r in out_records)
    print(f"\nWrote {len(out_records)} records -> {OUT}")
    print(f"  class balance : {n_pos} YES / {len(out_records) - n_pos} NO")
    print(f"  horizons      : {dict(Counter(r['horizon_label'] for r in out_records))}")
    print(f"  row_origin    : {dict(Counter(r['row_origin'] for r in out_records))}")
    if missing_join:
        print(f"  !! MISSING extraction_log join for {len(missing_join)}: {missing_join[:5]}")
    if missing_frames:
        print(f"  !! MISSING frame dirs on disk for {len(missing_frames)}: {missing_frames[:5]}")
    else:
        print(f"  all {len(out_records)} frames_dir exist under {FRAMES_ROOT}")
    if leak:
        print("  !! LEAKAGE: train videos overlap val — investigate before training")


if __name__ == "__main__":
    main()

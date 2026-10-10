"""
r1_merge_pilot_boxes.py - keep the boxes of the windows the teacher already answered (pilot + rescue) in the full teacher dataset.

The pod recomputed tracks and attention for all 4,203 windows. For the windows that already have boxes and E3 answers (pilot set A/B, rescue test)
the existing boxed frames, review grids and boxes.jsonl rows are used instead (A1 scores agree within 0.008; 6 of 7 top-track differences are
track numbers shifted by one on the same object; 00709 TTE 1.0 differs for real), so their teacher answers stay valid.
Also copies their E3 answers into the full run's runs/E3.jsonl (resumable teacher runs skip them).

    python r1_merge_pilot_boxes.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FULL = ROOT / "outputs" / "teacher_dataset_2026-10"
SOURCES = [ROOT / "outputs" / "teacher_pilot_2026-10", ROOT / "outputs" / "teacher_rescue_2026-10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(FULL / "boxes.jsonl", encoding="utf-8")}
    n_box = n_e3 = 0
    e3_out = {}
    for src in SOURCES:
        old = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(src / "boxes.jsonl", encoding="utf-8")}
        e3 = ({json.loads(l)["frames_dir"]: json.loads(l) for l in open(src / "runs" / "E3.jsonl", encoding="utf-8")}
              if (src / "runs" / "E3.jsonl").exists() else {})
        for fd, r in old.items():
            if fd not in rows:
                continue
            if not args.dry_run:
                for kind in ("boxed", "review"):
                    a = src / kind / (fd if kind == "boxed" else fd + ".jpg")
                    b = FULL / kind / (fd if kind == "boxed" else fd + ".jpg")
                    if a.is_dir():
                        shutil.rmtree(b, ignore_errors=True)
                        shutil.copytree(a, b)
                    elif a.exists():
                        shutil.copy2(a, b)
            rows[fd] = dict(r, set="full", source="pilot/rescue (kept)")
            n_box += 1
            if fd in e3:
                e3_out[fd] = e3[fd]
                n_e3 += 1
    print(f"[merge] boxes taken from the pilot / rescue folders: {n_box}; E3 answers carried over: {n_e3}")
    if args.dry_run:
        return
    with open(FULL / "boxes.jsonl", "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r) + "\n")
    (FULL / "runs").mkdir(exist_ok=True)
    p = FULL / "runs" / "E3.jsonl"
    have = {json.loads(l)["frames_dir"] for l in open(p, encoding="utf-8")} if p.exists() else set()
    with open(p, "a", encoding="utf-8") as f:
        for fd, r in e3_out.items():
            if fd not in have:
                f.write(json.dumps(dict(r, set="full"), ensure_ascii=False) + "\n")
    print("[merge] done")


if __name__ == "__main__":
    main()

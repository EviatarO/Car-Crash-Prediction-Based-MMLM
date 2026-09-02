"""
merge_semtest200_v2.py
=======================
Combine the original 200-clip selection.jsonl with the new 100-clip
selection_easy100.jsonl (select_semtest200_easy.py) into one 300-row pool,
selection_v2.jsonl / .xlsx / val_vids.txt - written in the exact schema
select_semtest200_recovery.py already uses, so every downstream script
(make_semtest200_folds.py, build_semtest200_comparison.py, semsup_train.py
--val-video-ids) needs no format-specific changes.

Also asserts the two inputs are video-disjoint (redundant with
select_semtest200_easy.py's own exclusion, but cheap and catches a stale
--base-selection pointed at the wrong file).

Usage:
  python merge_semtest200_v2.py \
      --base ../../outputs/semtest200/selection.jsonl \
      --easy ../../outputs/semtest200_v2/selection_easy100.jsonl \
      --out-dir ../../outputs/semtest200_v2
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from openpyxl import Workbook

from select_semtest200_recovery import HEADER_FILL, HEADER_FONT


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--easy", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    base = load(args.base)
    easy = load(args.easy)
    base_vids = {r["video_id"] for r in base}
    easy_vids = {r["video_id"] for r in easy}
    overlap = base_vids & easy_vids
    if overlap:
        raise SystemExit(f"{len(overlap)} video_ids appear in BOTH pools: "
                          f"{sorted(overlap)[:10]}... - not disjoint, refusing to merge")

    combined = base + easy
    print(f"[merge] base={len(base)}  easy={len(easy)}  combined={len(combined)}  "
          f"unique_videos={len({r['video_id'] for r in combined})}")
    print("[source]", dict(Counter(r["source"] for r in combined)))
    print("[split] ", dict(Counter(r["split"] for r in combined)))
    print("[gt]    ", dict(Counter(r["gt_verdict"] for r in combined)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "selection_v2.jsonl", "w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")

    headers = ["video_id", "frames_dir", "gt_verdict", "horizon_label", "source",
               "a0_score", "response_time", "split"]
    wb = Workbook()
    ws = wb.active
    ws.title = "selection_v2"
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    for r in combined:
        ws.append([r["video_id"], r["frames_dir"], r["gt_verdict"], r["horizon_label"],
                   r["source"], round(r["a0_score"], 4), r.get("response_time"), r["split"]])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:H{len(combined) + 1}"
    wb.save(out_dir / "selection_v2.xlsx")

    with open(out_dir / "val_vids.txt", "w", encoding="utf-8", newline="\n") as f:
        for r in combined:
            if r["split"] == "val":
                f.write(r["video_id"] + "\n")

    print(f"[wrote] {out_dir / 'selection_v2.jsonl'}")
    print(f"[wrote] {out_dir / 'selection_v2.xlsx'}")
    print(f"[wrote] {out_dir / 'val_vids.txt'}")


if __name__ == "__main__":
    main()

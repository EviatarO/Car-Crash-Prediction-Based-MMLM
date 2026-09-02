"""
build_semtest200_v2_a0_scores.py
==================================
A0 (frozen baseline) needs no training or pod time - its score for every clip already
exists in the full 4,446-window pool scoring run. This just filters that file down to
the v2 300-clip pool and re-writes it in the scores/<arm>.jsonl schema
build_semtest200_comparison.py expects, so the A0 column and the hard-gate re-score
consistency check work immediately, before any of the 4 trained arms exist.

Usage:
  python build_semtest200_v2_a0_scores.py \
      --a0-full ../../outputs/semtest200/A0_full4446.jsonl \
      --selection ../../outputs/semtest200_v2/selection_v2.jsonl \
      --out ../../outputs/semtest200_v2/scores/A0.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0-full", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sel = {json.loads(l)["frames_dir"] for l in open(args.selection, encoding="utf-8") if l.strip()}
    out_rows = []
    for line in open(args.a0_full, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["frames_dir"] in sel:
            out_rows.append({
                "arm": "A0", "frames_dir": r["frames_dir"], "video_id": r["video_id"],
                "score": r["score"], "gt_verdict": r["gt_verdict"],
                "requested_time_to_event": r.get("requested_time_to_event"),
            })

    missing = sel - {r["frames_dir"] for r in out_rows}
    if missing:
        raise SystemExit(f"{len(missing)} selection rows have no A0 score in "
                          f"{args.a0_full}, e.g. {sorted(missing)[:5]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[wrote] {out_path} ({len(out_rows)} rows, matching {len(sel)}-row selection)")


if __name__ == "__main__":
    main()

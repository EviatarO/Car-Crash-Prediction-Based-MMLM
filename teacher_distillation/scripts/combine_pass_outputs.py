"""Combine e3a-style pass-1 + pass-2(recovery) JSONLs into the aggregator schema.

The e3a runners emit pass-1 rows (fields: verdict/reasoning) and recovery rows
(fields: recovery_verdict/recovery_reasoning). teacher_reasoning_aggregate.py expects
the debate schema (collision_verdict / verdict_reasoning / p2_collision_verdict /
p2_verdict_reasoning / final_verdict / final_reasoning). This maps one to the other.

  final_* = recovery result when a clip was recovered, else the pass-1 result.

Usage:
  python teacher_distillation/scripts/combine_pass_outputs.py \
      --pass1 outputs/teacher_reasoning/stages/private_stage1/pass1.jsonl \
      --recovery outputs/teacher_reasoning/stages/private_stage1/recovery.jsonl \
      --split private \
      --out outputs/teacher_reasoning/stages/private_stage1/combined.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(p: Path):
    if not p or not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _dedupe(rows, verdict_field):
    """Keep the row per (video_id, horizon_label) that actually has a verdict."""
    out = {}
    for r in rows:
        k = (r["video_id"], r.get("horizon_label"))
        if k not in out or (out[k].get(verdict_field) is None and r.get(verdict_field) is not None):
            out[k] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1", required=True)
    ap.add_argument("--recovery", default="")
    ap.add_argument("--split", default="", help="optional split tag (private/public)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    p1 = _dedupe(_load(Path(args.pass1)), "verdict")
    rec = _dedupe(_load(Path(args.recovery)) if args.recovery else [], "recovery_verdict")

    recs = []
    for k, r in p1.items():
        rv = rec.get(k)
        recovered = rv is not None and rv.get("recovery_verdict") is not None
        p1_v = r.get("verdict")
        p1_reason = r.get("reasoning")
        p2_v = rv.get("recovery_verdict") if rv else None
        p2_reason = rv.get("recovery_reasoning") if rv else None
        recs.append({
            "video_id": r["video_id"],
            "gt_verdict": r.get("gt_verdict"),
            "horizon_label": r.get("horizon_label"),
            "requested_time_to_event": r.get("horizon_s"),
            "t_seconds": None,  # not meaningful for pre-cut test clips; horizon carries the info
            "collision_verdict": p1_v,
            "verdict_reasoning": p1_reason,
            "p2_collision_verdict": p2_v,
            "p2_verdict_reasoning": p2_reason,
            "final_verdict": p2_v if recovered else p1_v,
            "final_reasoning": p2_reason if recovered else p1_reason,
            "frames_subdir": r.get("frames_subdir"),
            "split": args.split or r.get("split", ""),
        })

    out = Path(args.out if Path(args.out).is_absolute() else REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_rec = sum(1 for r in recs if r["p2_collision_verdict"] is not None)
    print(f"combined {len(recs)} clips ({n_rec} recovered) -> {out}")


if __name__ == "__main__":
    main()

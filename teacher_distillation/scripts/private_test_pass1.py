"""Private-test teacher pass-1 — reuse the PROVEN e3a runner, repointed at the test set.

We do NOT re-implement the teacher call or its cost-safety rig. We import
`e3a_tte_fill_pass1.py` as a module, monkeypatch only its path/ceiling constants, and
call its `main()` verbatim — so the $-ceiling, cost-anomaly guard, heartbeat, resume,
and stop-and-ask behaviour are exactly the ones battle-tested on e3a.

Differences from e3a (train):
  * frames already exist -> NO extraction stage. Each private-test clip has 16 native
    1280x720 frames at dataset/test/<id>_hires/frame_00001..16.jpg.
  * one fixed TTE per clip (from the manifest), not 3 re-cut horizons.

Input : outputs/teacher_reasoning/stages/private_stage1.todo.jsonl  (from teacher_next_batch.py)
Output: outputs/teacher_reasoning/stages/private_stage1/pass1.jsonl

Usage (set OPENROUTER_API_KEY first; nothing is spent until you run this):
  python teacher_distillation/scripts/private_test_pass1.py --budget 30
Dry check (build the plan, print the roster, spend nothing):
  python teacher_distillation/scripts/private_test_pass1.py --plan_only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "teacher_distillation" / "scripts"

STAGES = REPO_ROOT / "outputs" / "teacher_reasoning" / "stages"
FRAMES_ROOT = REPO_ROOT / "dataset" / "test"


def _tte_label(x) -> str:
    try:
        return f"TTE_{float(x):.1f}"
    except (ValueError, TypeError):
        return str(x)


def build_plan(batch: Path) -> list:
    """Convert the balanced batch (.todo) into the e3a extraction_log plan schema."""
    rows = [json.loads(l) for l in batch.read_text(encoding="utf-8").splitlines() if l.strip()]
    plan = []
    for r in rows:
        vid = r["video_id"]
        horizon_s = r.get("requested_time_to_event")
        plan.append({
            "video_id": vid,
            "gt_verdict": r["gt_verdict"],
            "new_horizon_label": _tte_label(horizon_s),
            "frames_subdir": r.get("frames_dir", f"{vid}_hires"),
            "t_new": r.get("t_seconds"),
            "horizon_s": horizon_s,
            "split": r.get("split", "private"),
            "fps": r.get("fps", 30.0),
        })
    return plan


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=30.0, help="USD hard-stop ceiling for pass-1")
    ap.add_argument("--plan_only", action="store_true", help="build plan + print roster, no API calls")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: process only a balanced N (front TP + back TN); 0 = full plan. "
                         "Rows land in the same pass1.jsonl, so a later full run resumes over them.")
    ap.add_argument("--stage", default="private_stage1",
                    help="stage name -> batch stages/<stage>.todo.jsonl, outputs stages/<stage>/")
    args = ap.parse_args()

    batch = STAGES / f"{args.stage}.todo.jsonl"
    stage_dir = STAGES / args.stage
    PLAN = stage_dir / "plan.json"
    OUT_JSONL = stage_dir / "pass1.jsonl"
    STOP_FILE = stage_dir / "STOP_REASON.json"

    if not batch.exists():
        raise SystemExit(f"batch not found: {batch}\n"
                         f"run teacher_next_batch.py --dataset test --split private --n N "
                         f"--out outputs/teacher_reasoning/stages/{args.stage}.todo.jsonl first")

    stage_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(batch)
    if args.limit and args.limit < len(plan):
        n_tp = (args.limit + 1) // 2          # front of plan is TP
        n_tn = args.limit - n_tp              # back of plan is TN
        plan = plan[:n_tp] + (plan[-n_tn:] if n_tn else [])
        print(f"[smoke] limiting to {len(plan)} clips ({n_tp} TP + {n_tn} TN)")
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    # sanity: every frame folder present
    missing = [p["video_id"] for p in plan
               if not (FRAMES_ROOT / p["frames_subdir"] / "frame_00001.jpg").exists()]
    n_yes = sum(1 for p in plan if p["gt_verdict"] == "YES")
    print(f"Plan: {len(plan)} clips  (TP={n_yes}, TN={len(plan)-n_yes})  -> {PLAN}")
    from collections import Counter
    print(f"  TP by TTE: {dict(sorted(Counter(p['new_horizon_label'] for p in plan if p['gt_verdict']=='YES').items()))}")
    print(f"  frame folders missing: {len(missing)}"
          + (f"  !! {missing[:5]}" if missing else ""))
    est = len(plan) * 0.0594
    print(f"  est pass-1 cost @ $0.0594/clip: ${est:.2f}  (ceiling ${args.budget:.2f})")
    if missing:
        raise SystemExit("Refusing to run: some frame folders are missing.")
    if args.plan_only:
        print("plan_only: no API calls made.")
        return

    # ---- reuse the proven e3a pass-1 runner, repointed ----
    m = _load_module(SCRIPTS / "e3a_tte_fill_pass1.py", "e3a_pass1")
    m.EXTRACTION_LOG = PLAN
    m.OUT_JSONL = OUT_JSONL
    m.STOP_FILE = STOP_FILE
    m.DEFAULT_FRAMES_ROOT = FRAMES_ROOT
    m.STAGE_BUDGET_CEILING = args.budget
    m.main()


if __name__ == "__main__":
    main()

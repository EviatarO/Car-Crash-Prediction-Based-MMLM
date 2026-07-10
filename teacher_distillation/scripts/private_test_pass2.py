"""Private-test teacher pass-2 (recovery / "debate") — reuse the PROVEN e3a pass-2 runner.

Reads the pass-1 output, queues only the mismatches (verdict != gt), and re-asks with the
TP_RECOVERY (FN) / TN_RECOVERY (FP) prompts. Same monkeypatch approach as pass-1: import
e3a_tte_fill_pass2.py, repoint its path/ceiling constants, call main() verbatim.

Input : outputs/teacher_reasoning/stages/private_stage1/pass1.jsonl
Output: outputs/teacher_reasoning/stages/private_stage1/recovery.jsonl

Usage:
  python teacher_distillation/scripts/private_test_pass2.py --budget 12
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "teacher_distillation" / "scripts"

STAGES = REPO_ROOT / "outputs" / "teacher_reasoning" / "stages"
FRAMES_ROOT = REPO_ROOT / "dataset" / "test"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=12.0, help="USD hard-stop ceiling for recovery")
    ap.add_argument("--stage", default="private_stage1", help="stage name -> stages/<stage>/")
    args = ap.parse_args()

    stage_dir = STAGES / args.stage
    pass1_jsonl = stage_dir / "pass1.jsonl"
    if not pass1_jsonl.exists():
        raise SystemExit(f"pass-1 output not found: {pass1_jsonl}\nrun private_test_pass1.py first")

    m = _load_module(SCRIPTS / "e3a_tte_fill_pass2.py", "e3a_pass2")
    m.PASS1_JSONL = pass1_jsonl
    m.OUT_JSONL = stage_dir / "recovery.jsonl"
    m.STOP_FILE = stage_dir / "STOP_REASON_pass2.json"
    m.DEFAULT_FRAMES_ROOT = FRAMES_ROOT
    m.STAGE_BUDGET_CEILING = args.budget
    m.main()


if __name__ == "__main__":
    main()

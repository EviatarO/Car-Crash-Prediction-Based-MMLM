"""v7.1 A/B test: run v7.1 recovery prompts on stride-4 frames.

Purpose: Isolate the PROMPT variable from the STRIDE variable.
         v6 debate used inverted prompts on stride-4 frames.
         v7 debate used correct prompts on stride-8 frames.
         This test uses v7.1 (correct) prompts on stride-4 frames.

Reads v6_hires_full18.jsonl to identify the 5 first-pass failures at stride-4:
  00474  GT=YES  pred=NO  (FN) -> v7.1 TP_RECOVERY
  01153  GT=NO   pred=YES (FP) -> v7.1 TN_RECOVERY
  01504  GT=NO   pred=YES (FP) -> v7.1 TN_RECOVERY
  02104  GT=NO   pred=YES (FP) -> v7.1 TN_RECOVERY
  02117  GT=NO   pred=YES (FP) -> v7.1 TN_RECOVERY

Uses stride-4 hires frames: dataset/train/<vid>_hires/ (same as v6 experiment).

Outputs:
  outputs/prompt_bakeoff/v7_1_s4_ab/v7_1_debate.jsonl
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "teacher_distillation" / "scripts"))

from teacher_bakeoff import (  # noqa: E402
    _build_messages, _calc_cost, _call_model, _load_clip_frames, _parse_response,
)
from teacher_prompt_bakeoff import _read_gt_excel_with_en  # noqa: E402
from apo_metric import score_one, warmup_bertscore  # noqa: E402

MODEL_SLUG = "google/gemini-3.1-pro-preview"
PRICE_IN   = 2.00
PRICE_OUT  = 12.00
TEMPERATURE = 0.1

# -- Timeout / retry settings ------------------------------------------------
DEFAULT_TIMEOUT   = 90.0
MAX_RETRIES       = 2
RETRY_DELAY       = 5.0
CLIP_BUDGET_SECS  = 300
INTER_CALL_DELAY  = 1.5
PROMPT_TOKEN_HARD_CAP = 100_000
# -----------------------------------------------------------------------------

DEFAULT_GT_XLSX     = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"

# First-pass results from v6@stride-4
FIRST_JSONL = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v6_hires_full18.jsonl"

# Output for this A/B test
OUT_DIR     = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v7_1_s4_ab"
DEBATE_JSONL = OUT_DIR / "v7_1_debate.jsonl"

# v7.1 prompts (2-second duration)
PROMPT_TP_FILE = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v7_1_TP_RECOVERY.py"
PROMPT_TN_FILE = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v7_1_TN_RECOVERY.py"


def _load_prompt(path: Path, var_name: str) -> str:
    spec = importlib.util.spec_from_file_location(f"_p_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, var_name)


def _load_jsonl_dict(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            out[rec["video_id"]] = rec
        except Exception:
            pass
    return out


def _append(path: Path, rec: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _call_with_budget(client, prompt: str, vid: str, frames_subdir: str) -> Tuple[str, Dict, float]:
    """Run a single call with wall-clock budget guard."""
    def _inner():
        indices = list(range(1, 17))
        b64s = _load_clip_frames(DEFAULT_FRAMES_ROOT, frames_subdir, indices, frame_size=0)
        messages = _build_messages(prompt, b64s, detail="high")
        t0 = time.time()
        raw, usage = _call_model(
            client, MODEL_SLUG, messages,
            timeout=DEFAULT_TIMEOUT, max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY, temperature=TEMPERATURE,
        )
        latency = time.time() - t0
        prompt_tok = usage.get("prompt_tokens", 0) if usage else 0
        if prompt_tok and prompt_tok > PROMPT_TOKEN_HARD_CAP:
            raise RuntimeError(
                f"Prompt token count {prompt_tok} > hard cap {PROMPT_TOKEN_HARD_CAP}"
            )
        return raw, usage, latency

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_inner)
        try:
            return future.result(timeout=CLIP_BUDGET_SECS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise RuntimeError(
                f"Wall-clock budget {CLIP_BUDGET_SECS}s exceeded for {vid}"
            )


def main() -> None:
    tp_prompt = _load_prompt(PROMPT_TP_FILE, "PROMPT_G_OPT_v7_1_TP_RECOVERY")
    tn_prompt = _load_prompt(PROMPT_TN_FILE, "PROMPT_G_OPT_v7_1_TN_RECOVERY")
    print(f"Loaded v7.1 TP_RECOVERY ({len(tp_prompt)} chars)")
    print(f"Loaded v7.1 TN_RECOVERY ({len(tn_prompt)} chars)")
    print(f"Timeout: {DEFAULT_TIMEOUT}s/attempt, {MAX_RETRIES} retries, "
          f"{CLIP_BUDGET_SECS}s wall-clock budget per clip")

    # Load first-pass results
    first = _load_jsonl_dict(FIRST_JSONL)
    print(f"\nFirst-pass records loaded: {len(first)} from {FIRST_JSONL.name}")

    # Identify failures
    failures = []
    for vid, rec in sorted(first.items()):
        if rec.get("verdict") is None:
            continue
        if rec["verdict"] != rec["gt_verdict"]:
            failures.append((vid, rec))

    print(f"\nStride-4 first-pass failures to debate: {len(failures)}")
    for vid, rec in failures:
        kind = "FN" if rec["gt_verdict"] == "YES" else "FP"
        print(f"  {vid}  GT={rec['gt_verdict']}  v6={rec['verdict']}  [{kind}]")

    if not failures:
        print("No failures found. Nothing to do.")
        return

    # Load GT for BERTScore
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_v7_1_ab"),
        },
    )

    clips_all = _read_gt_excel_with_en(DEFAULT_GT_XLSX)
    clip_map = {c["video_id"]: c for c in clips_all}

    print("\nWarming up BERTScore...")
    warmup_bertscore()
    print("BERTScore ready.\n")

    # Check for resume
    existing = _load_jsonl_dict(DEBATE_JSONL)
    total_cost = 0.0
    n_fixed = n_still = n_err = 0

    print(f"=== v7.1 A/B TEST: recovery on stride-4 frames ===\n")

    for idx, (vid, first_rec) in enumerate(failures, start=1):
        clip = clip_map.get(vid)
        if not clip:
            print(f"[{idx}/{len(failures)}] {vid} -- skip (not in GT)")
            continue

        gt = first_rec["gt_verdict"]

        if vid in existing and existing[vid].get("recovery_verdict") is not None:
            r = existing[vid]
            flipped = "FIXED" if r["recovery_verdict"] == gt else "still-wrong"
            print(f"[{idx}/{len(failures)}] {vid} -- skip (already done, "
                  f"recovery={r['recovery_verdict']} {flipped})")
            total_cost += r.get("cost_usd", 0.0)
            if flipped == "FIXED":
                n_fixed += 1
            else:
                n_still += 1
            continue

        if gt == "YES":
            recovery_name = "PROMPT_G_OPT_v7_1_TP_RECOVERY"
            prompt = tp_prompt
        else:
            recovery_name = "PROMPT_G_OPT_v7_1_TN_RECOVERY"
            prompt = tn_prompt

        # Use stride-4 hires frames: <vid>_hires/
        frames_subdir = f"{vid}_hires"

        print(f"[{idx}/{len(failures)}] {vid}  GT={gt}  v6={first_rec['verdict']}  "
              f"-> {recovery_name}", end="  ", flush=True)

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        clip_start = time.time()
        try:
            raw, usage, latency = _call_with_budget(client, prompt, vid, frames_subdir)
            parsed, recovery_verdict = _parse_response(raw)
            cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)
            reasoning = parsed.get("verdict_reasoning") if parsed else None
            sb = score_one(recovery_verdict, reasoning, gt, clip["gt_reasoning_en"])

            flipped = "FIXED" if recovery_verdict == gt else "still-wrong"
            print(f"recovery={recovery_verdict or '??':3s}  {flipped}  "
                  f"BERT={sb.alignment:.3f}  cost=${cost:.4f}  "
                  f"in_tok={usage.get('prompt_tokens')}  "
                  f"out_tok={usage.get('completion_tokens')}  {latency:.1f}s")
            total_cost += cost
            if flipped == "FIXED":
                n_fixed += 1
            else:
                n_still += 1

            rec = {
                "video_id": vid,
                "recovery_prompt": recovery_name,
                "resolution": "native_1280x720",
                "stride": 4,
                "detail": "high",
                "gt_verdict": gt,
                "target": clip["target"],
                "t_seconds": clip["t_seconds"],
                "v6_verdict": first_rec["verdict"],
                "recovery_verdict": recovery_verdict,
                "recovery_reasoning": reasoning,
                "full_json": parsed or {},
                "scores": sb.to_dict(),
                "raw": raw,
                "usage": usage,
                "cost_usd": cost,
                "latency_s": round(latency, 2),
                "error": None,
            }
        except Exception as exc:
            elapsed = time.time() - clip_start
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[ERR] {msg}")
            n_err += 1
            rec = {
                "video_id": vid, "recovery_prompt": recovery_name if gt == "YES" else recovery_name,
                "resolution": "native_1280x720", "stride": 4, "detail": "high",
                "gt_verdict": gt, "target": clip.get("target"),
                "t_seconds": clip.get("t_seconds"),
                "v6_verdict": first_rec["verdict"],
                "recovery_verdict": None, "recovery_reasoning": None,
                "full_json": {}, "scores": {"composite": 0, "verdict": 0, "alignment": 0, "length": 0, "word_count": 0},
                "raw": "", "usage": {}, "cost_usd": 0.0,
                "latency_s": round(elapsed, 2), "error": msg,
            }
        _append(DEBATE_JSONL, rec)

    print()
    print("-" * 65)
    print(f"v7.1 A/B test done. FIXED={n_fixed}  still-wrong={n_still}  errors={n_err}")
    print(f"Total cost: ${total_cost:.4f}")
    print("-" * 65)

    # Quick comparison with v6 debate
    v6_debate = _load_jsonl_dict(REPO_ROOT / "outputs" / "prompt_bakeoff" / "v6_debate.jsonl")
    v7_1_debate = _load_jsonl_dict(DEBATE_JSONL)
    print("\n=== COMPARISON: v6 (inverted) vs v7.1 (correct) on stride-4 ===")
    print(f"{'Clip':>6}  {'GT':>3}  {'v6_rec':>6}  {'v7.1_rec':>8}  v6_outcome  v7.1_outcome")
    for vid, first_rec in failures:
        gt = first_rec["gt_verdict"]
        v6_r = v6_debate.get(vid, {})
        v71_r = v7_1_debate.get(vid, {})
        v6_rec = v6_r.get("recovery_verdict", "N/A")
        v71_rec = v71_r.get("recovery_verdict", "N/A")
        v6_out = "FIXED" if v6_rec == gt else ("still-wrong" if v6_rec != "N/A" else "N/A")
        v71_out = "FIXED" if v71_rec == gt else ("still-wrong" if v71_rec != "N/A" else "N/A")
        print(f"{vid:>6}  {gt:>3}  {v6_rec:>6}  {v71_rec:>8}  {v6_out:>10}  {v71_out:>12}")

    # Summary
    v6_fixed = sum(1 for vid, _ in failures if v6_debate.get(vid, {}).get("recovery_verdict") == first[vid]["gt_verdict"])
    v71_fixed = n_fixed
    print(f"\nv6@stride-4 (inverted prompts): {v6_fixed}/{len(failures)} FIXED")
    print(f"v7.1@stride-4 (correct prompts): {v71_fixed}/{len(failures)} FIXED")
    first_acc = sum(1 for r in first.values() if r.get("verdict") == r.get("gt_verdict")) / len(first)
    v6_final = first_acc * len(first) + v6_fixed
    v71_final = first_acc * len(first) + v71_fixed
    print(f"\nFinal accuracy after debate:")
    print(f"  v6 (inverted): {v6_final:.0f}/{len(first)} = {v6_final/len(first)*100:.1f}%")
    print(f"  v7.1 (correct): {v71_final:.0f}/{len(first)} = {v71_final/len(first)*100:.1f}%")

    print(f"\nOutput: {DEBATE_JSONL}")


if __name__ == "__main__":
    main()

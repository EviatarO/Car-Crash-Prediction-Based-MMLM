"""v7 debate experiment on 18 GT clips with stride-8 hi-res frames.

Stage 1: PROMPT_G_OPT_v6_balanced_s8 on dataset/train/<vid>_hires_s8/ (16 frames,
         ~4 second window, native 1280x720).
Stage 2: For each clip where stage-1 verdict != gt_verdict, run the appropriate
         v7 recovery prompt:
            GT=YES (FN) -> PROMPT_G_OPT_v7_TP_RECOVERY (proactive hazard analyst)
            GT=NO  (FP) -> PROMPT_G_OPT_v7_TN_RECOVERY (conservative auditor)

Outputs (both resume-safe, keyed by video_id):
  outputs/prompt_bakeoff/v7_stride8/v6_s8_hires.jsonl
  outputs/prompt_bakeoff/v7_stride8/v7_debate.jsonl

Timeout safety:
  DEFAULT_TIMEOUT  = 90s per HTTP attempt
  MAX_RETRIES      = 2
  CLIP_BUDGET_SECS = 300s wall-clock per clip (concurrent.futures)
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

# ── Timeout / retry settings ─────────────────────────────────────────────────
DEFAULT_TIMEOUT   = 90.0
MAX_RETRIES       = 2
RETRY_DELAY       = 5.0
CLIP_BUDGET_SECS  = 300
INTER_CALL_DELAY  = 1.5
PROMPT_TOKEN_HARD_CAP = 100_000
# ─────────────────────────────────────────────────────────────────────────────

CLIPS_TO_RUN = [
    "00077", "00147", "00283", "00319", "00372", "00474",
    "00493", "00529", "00687", "01153", "01281", "01504",
    "01550", "01552", "01643", "01737", "02104", "02117",
]

DEFAULT_GT_XLSX     = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"

OUT_DIR             = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v7_stride8"
FIRST_JSONL         = OUT_DIR / "v6_s8_hires.jsonl"
DEBATE_JSONL        = OUT_DIR / "v7_debate.jsonl"

PROMPT_FIRST_FILE   = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_balanced_s8.py"
PROMPT_FIRST_VAR    = "PROMPT_G_OPT_v6_balanced_s8"
PROMPT_TP_FILE      = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v7_TP_RECOVERY.py"
PROMPT_TN_FILE      = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v7_TN_RECOVERY.py"


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
    """Run a single call with the wall-clock budget guard. Returns (raw, usage, latency)."""
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


def _run_first_pass(client, prompt: str, clips: List[Dict]) -> None:
    """Stage 1: PROMPT_G_OPT_v6_balanced_s8 on stride-8 frames for all 18 clips."""
    existing = _load_jsonl_dict(FIRST_JSONL)
    done = sum(1 for r in existing.values() if r.get("verdict") is not None)
    print(f"\n=== STAGE 1: first pass (v6_balanced_s8) ===")
    print(f"Resume: {done}/{len(clips)} clips already done\n")

    total_cost = 0.0
    n_ok = n_xx = n_err = 0
    for idx, clip in enumerate(clips, start=1):
        vid = clip["video_id"]
        gt = clip["gt_verdict"]

        if vid in existing and existing[vid].get("verdict") is not None:
            r = existing[vid]
            ok = "[OK]" if r["verdict"] == gt else "[XX]"
            print(f"[{idx:2d}/{len(clips)}] {vid} -- skip (already done, verdict={r['verdict']} {ok})")
            total_cost += r.get("cost_usd", 0.0)
            if r["verdict"] == gt:
                n_ok += 1
            else:
                n_xx += 1
            continue

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        print(f"[{idx:2d}/{len(clips)}] {vid}  GT={gt}  t={clip['t_seconds']:.2f}s",
              end="  ", flush=True)

        clip_start = time.time()
        try:
            raw, usage, latency = _call_with_budget(client, prompt, vid, f"{vid}_hires_s8")
            parsed, verdict = _parse_response(raw)
            cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)
            reasoning = parsed.get("verdict_reasoning") if parsed else None
            sb = score_one(verdict, reasoning, gt, clip["gt_reasoning_en"])

            ok_str = "[OK]" if verdict == gt else "[XX]"
            print(f"verdict={verdict or '??':3s} {ok_str}  "
                  f"BERT={sb.alignment:.3f}  cost=${cost:.4f}  "
                  f"in_tok={usage.get('prompt_tokens')}  "
                  f"out_tok={usage.get('completion_tokens')}  {latency:.1f}s")
            total_cost += cost
            if verdict == gt:
                n_ok += 1
            else:
                n_xx += 1

            rec = {
                "video_id": vid,
                "prompt_name": "PROMPT_G_OPT_v6_balanced_s8",
                "resolution": "native_1280x720",
                "stride": 8,
                "detail": "high",
                "gt_verdict": gt,
                "target": clip["target"],
                "t_seconds": clip["t_seconds"],
                "verdict": verdict,
                "reasoning": reasoning,
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
                "video_id": vid,
                "prompt_name": "PROMPT_G_OPT_v6_balanced_s8",
                "resolution": "native_1280x720",
                "stride": 8,
                "detail": "high",
                "gt_verdict": gt,
                "target": clip["target"],
                "t_seconds": clip["t_seconds"],
                "verdict": None,
                "reasoning": None,
                "full_json": {},
                "scores": {"composite": 0, "verdict": 0, "alignment": 0, "length": 0, "word_count": 0},
                "raw": "",
                "usage": {},
                "cost_usd": 0.0,
                "latency_s": round(elapsed, 2),
                "error": msg,
            }
        _append(FIRST_JSONL, rec)

    print()
    print("-" * 65)
    print(f"Stage 1 done. {n_ok} OK  |  {n_xx} wrong  |  {n_err} errors")
    print(f"Stage 1 accuracy: {n_ok}/{n_ok+n_xx+n_err} = "
          f"{n_ok/(n_ok+n_xx+n_err)*100:.1f}%")
    print(f"Stage 1 cost: ${total_cost:.4f}")
    print("-" * 65)


def _run_debate(client, tp_prompt: str, tn_prompt: str, clips: List[Dict]) -> None:
    """Stage 2: v7 recovery on first-pass failures."""
    first = _load_jsonl_dict(FIRST_JSONL)
    clip_map = {c["video_id"]: c for c in clips}

    failures = []
    for vid, rec in sorted(first.items()):
        if rec.get("verdict") is None:
            continue
        if rec["verdict"] != rec["gt_verdict"]:
            failures.append((vid, rec))

    print(f"\n=== STAGE 2: v7 debate ===")
    print(f"Failures to debate: {len(failures)}")
    for vid, rec in failures:
        kind = "FN" if rec["gt_verdict"] == "YES" else "FP"
        print(f"  {vid}  GT={rec['gt_verdict']}  s8={rec['verdict']}  [{kind}]")
    print()

    if not failures:
        print("No failures. Stage 2 done.")
        return

    existing = _load_jsonl_dict(DEBATE_JSONL)
    total_cost = 0.0
    n_fixed = n_still = n_err = 0

    for idx, (vid, first_rec) in enumerate(failures, start=1):
        clip = clip_map[vid]
        gt = first_rec["gt_verdict"]

        if vid in existing and existing[vid].get("recovery_verdict") is not None:
            r = existing[vid]
            flipped = "FIXED" if r["recovery_verdict"] == gt else "still-wrong"
            print(f"[{idx:2d}/{len(failures)}] {vid} -- skip (already done, "
                  f"recovery={r['recovery_verdict']} {flipped})")
            total_cost += r.get("cost_usd", 0.0)
            if flipped == "FIXED":
                n_fixed += 1
            else:
                n_still += 1
            continue

        if gt == "YES":
            recovery_name = "PROMPT_G_OPT_v7_TP_RECOVERY"
            prompt = tp_prompt
        else:
            recovery_name = "PROMPT_G_OPT_v7_TN_RECOVERY"
            prompt = tn_prompt

        print(f"[{idx:2d}/{len(failures)}] {vid}  GT={gt}  s8={first_rec['verdict']}  "
              f"-> {recovery_name}", end="  ", flush=True)

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        clip_start = time.time()
        try:
            raw, usage, latency = _call_with_budget(client, prompt, vid, f"{vid}_hires_s8")
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
                "stride": 8,
                "detail": "high",
                "gt_verdict": gt,
                "target": clip["target"],
                "t_seconds": clip["t_seconds"],
                "v6_s8_verdict": first_rec["verdict"],
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
                "video_id": vid, "recovery_prompt": recovery_name,
                "resolution": "native_1280x720", "stride": 8, "detail": "high",
                "gt_verdict": gt, "target": clip["target"],
                "t_seconds": clip["t_seconds"],
                "v6_s8_verdict": first_rec["verdict"],
                "recovery_verdict": None, "recovery_reasoning": None,
                "full_json": {}, "scores": {"composite": 0, "verdict": 0, "alignment": 0, "length": 0, "word_count": 0},
                "raw": "", "usage": {}, "cost_usd": 0.0,
                "latency_s": round(elapsed, 2), "error": msg,
            }
        _append(DEBATE_JSONL, rec)

    print()
    print("-" * 65)
    print(f"Stage 2 done. FIXED={n_fixed}  still-wrong={n_still}  errors={n_err}")
    print(f"Stage 2 cost: ${total_cost:.4f}")
    print("-" * 65)


def main() -> None:
    first_prompt = _load_prompt(PROMPT_FIRST_FILE, PROMPT_FIRST_VAR)
    tp_prompt    = _load_prompt(PROMPT_TP_FILE, "PROMPT_G_OPT_v7_TP_RECOVERY")
    tn_prompt    = _load_prompt(PROMPT_TN_FILE, "PROMPT_G_OPT_v7_TN_RECOVERY")
    print(f"Loaded first-pass prompt ({len(first_prompt)} chars)")
    print(f"Loaded v7 TP_RECOVERY ({len(tp_prompt)} chars)")
    print(f"Loaded v7 TN_RECOVERY ({len(tn_prompt)} chars)")
    print(f"Timeout: {DEFAULT_TIMEOUT}s/attempt, {MAX_RETRIES} retries, "
          f"{CLIP_BUDGET_SECS}s wall-clock budget per clip")

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_v7_stride8"),
        },
    )

    clips_all = _read_gt_excel_with_en(DEFAULT_GT_XLSX)
    clip_map = {c["video_id"]: c for c in clips_all}
    clips = [clip_map[v] for v in CLIPS_TO_RUN if v in clip_map]
    print(f"Clips: {len(clips)}  "
          f"(YES: {sum(c['gt_verdict']=='YES' for c in clips)}, "
          f"NO: {sum(c['gt_verdict']=='NO' for c in clips)})")

    print("\nWarming up BERTScore...")
    warmup_bertscore()
    print("BERTScore ready.")

    _run_first_pass(client, first_prompt, clips)
    _run_debate(client, tp_prompt, tn_prompt, clips)

    print("\nALL DONE.")
    print(f"  First pass : {FIRST_JSONL}")
    print(f"  Debate     : {DEBATE_JSONL}")


if __name__ == "__main__":
    main()

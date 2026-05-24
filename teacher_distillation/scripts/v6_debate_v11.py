"""Debate / second-opinion pass for the 100-clip v11 set.

For each clip where v6@hires got the verdict wrong (or errored):
  - GT=YES, v6=NO  (FN)  -> PROMPT_G_OPT_v6_TP_RECOVERY  (proactive hazard analyst)
  - GT=NO,  v6=YES (FP)  -> PROMPT_G_OPT_v6_TN_RECOVERY  (conservative auditor)
  - v6=None (API error)  -> treated as FN if GT=YES, FP if GT=NO

No BERTScore — no GT reasoning in v11.

Timeout safety (same as v6_hires_v11.py):
  DEFAULT_TIMEOUT   = 90s  per HTTP attempt
  MAX_RETRIES       = 2    (3 total attempts)
  CLIP_BUDGET_SECS  = 300s hard wall-clock budget via ThreadPoolExecutor

Resumable: keyed by video_id in v6_debate_v11.jsonl.
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "teacher_distillation" / "scripts"))

from teacher_bakeoff import (  # noqa: E402
    _build_messages, _calc_cost, _call_model, _load_clip_frames, _parse_response,
)

MODEL_SLUG = "google/gemini-3.1-pro-preview"
PRICE_IN   = 2.00
PRICE_OUT  = 12.00
TEMPERATURE = 0.1

# ── Timeout / retry settings ─────────────────────────────────────────────────
DEFAULT_TIMEOUT   = 90.0   # seconds per HTTP attempt
MAX_RETRIES       = 2      # retries after first failure (3 total attempts)
RETRY_DELAY       = 5.0    # base delay between retries (exponential backoff)
CLIP_BUDGET_SECS  = 300    # hard wall-clock cap per clip (covers all retries)
INTER_CALL_DELAY  = 1.5
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_TOKEN_HARD_CAP = 100_000

V11_XLSX          = REPO_ROOT / "outputs" / "teacher_dataset_v11.xlsx"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"
V6_JSONL          = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips" / "v6_hires_v11.jsonl"
OUT_DIR           = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips"
OUT_JSONL         = OUT_DIR / "v6_debate_v11.jsonl"

PROMPT_TP_FILE    = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_TP_RECOVERY.py"
PROMPT_TN_FILE    = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_TN_RECOVERY.py"


def _load_prompt_by_var(var_name: str, candidate_files: List[Path]) -> str:
    """Search candidate files for the requested prompt variable.
    Handles cases where the variable name doesn't match the filename."""
    for path in candidate_files:
        spec = importlib.util.spec_from_file_location(f"_p_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, var_name):
            if path.stem != var_name:
                print(f"  [INFO] Loaded var '{var_name}' from file '{path.name}' "
                      f"(name mismatch is expected)")
            return getattr(mod, var_name)
    raise AttributeError(
        f"Variable '{var_name}' not found in any of: {[p.name for p in candidate_files]}"
    )


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


def _append(rec: Dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_one(client, prompt: str, vid: str) -> tuple:
    """Make the API call. Returns (raw, usage, latency)."""
    indices = list(range(1, 17))
    b64s = _load_clip_frames(DEFAULT_FRAMES_ROOT, f"{vid}_hires", indices, frame_size=0)
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
            f"Prompt token count {prompt_tok} exceeded hard cap {PROMPT_TOKEN_HARD_CAP}"
        )
    return raw, usage, latency


def main() -> None:
    candidates = [PROMPT_TP_FILE, PROMPT_TN_FILE]
    tp_prompt = _load_prompt_by_var("PROMPT_G_OPT_v6_TP_RECOVERY", candidates)
    tn_prompt = _load_prompt_by_var("PROMPT_G_OPT_v6_TN_RECOVERY", candidates)
    print(f"Loaded TP_RECOVERY ({len(tp_prompt)} chars)")
    print(f"Loaded TN_RECOVERY ({len(tn_prompt)} chars)")
    print(f"Timeout: {DEFAULT_TIMEOUT}s/attempt, {MAX_RETRIES} retries, "
          f"{CLIP_BUDGET_SECS}s wall-clock budget per clip\n")

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_v11_debate"),
        },
    )

    # Load GT from v11.xlsx
    df = pd.read_excel(V11_XLSX)
    gt_map = {f"{int(row['video_id']):05d}": row["gt_verdict"] for _, row in df.iterrows()}

    # Load v6 results
    v6 = _load_jsonl_dict(V6_JSONL)
    print(f"v6@hires records loaded: {len(v6)}")

    # Identify failures: wrong verdict OR None verdict
    failures: List[tuple] = []
    for vid, rec in sorted(v6.items()):
        gt = gt_map.get(vid, rec.get("gt_verdict"))
        v6_verdict = rec.get("verdict")
        if v6_verdict != gt:   # includes None != "YES"/"NO"
            failures.append((vid, gt, v6_verdict))

    print(f"Failures to debate: {len(failures)}")
    for vid, gt, v6v in failures:
        kind = "FN" if gt == "YES" else "FP"
        print(f"  {vid}  GT={gt}  v6={v6v or 'None'}  [{kind}]")
    print()

    if not failures:
        print("No failures. Exiting.")
        return

    existing = _load_jsonl_dict(OUT_JSONL)
    done = sum(1 for r in existing.values() if r.get("recovery_verdict") is not None)
    print(f"Resume: {done}/{len(failures)} already done\n")

    total_cost = 0.0
    n_fixed = n_still_wrong = n_err = n_timeout = 0

    for idx, (vid, gt, v6_verdict) in enumerate(failures, start=1):
        # Already done
        if vid in existing and existing[vid].get("recovery_verdict") is not None:
            r = existing[vid]
            flipped = "FIXED" if r["recovery_verdict"] == gt else "still-wrong"
            print(f"[{idx:2d}/{len(failures)}] {vid} -- skip (already done, "
                  f"recovery={r['recovery_verdict']} {flipped})")
            total_cost += r.get("cost_usd", 0.0)
            if flipped == "FIXED":
                n_fixed += 1
            else:
                n_still_wrong += 1
            continue

        # Choose recovery prompt
        if gt == "YES":
            recovery_name = "PROMPT_G_OPT_v6_TP_RECOVERY"
            prompt = tp_prompt
        else:
            recovery_name = "PROMPT_G_OPT_v6_TN_RECOVERY"
            prompt = tn_prompt

        print(f"[{idx:2d}/{len(failures)}] {vid}  GT={gt}  v6={v6_verdict or 'None'}"
              f"  -> {recovery_name}", end="  ", flush=True)

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        clip_start = time.time()
        rec = None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_one, client, prompt, vid)
                try:
                    raw, usage, latency = future.result(timeout=CLIP_BUDGET_SECS)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    elapsed = time.time() - clip_start
                    raise RuntimeError(
                        f"Wall-clock budget exceeded ({elapsed:.0f}s > {CLIP_BUDGET_SECS}s)"
                    )

            parsed, recovery_verdict = _parse_response(raw)
            cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)
            reasoning = parsed.get("verdict_reasoning") if parsed else None

            flipped = "FIXED" if recovery_verdict == gt else "still-wrong"
            print(
                f"recovery={recovery_verdict or '??':3s}  {flipped}  "
                f"cost=${cost:.4f}  "
                f"in_tok={usage.get('prompt_tokens')}  "
                f"out_tok={usage.get('completion_tokens')}  "
                f"{latency:.1f}s"
            )
            total_cost += cost
            if flipped == "FIXED":
                n_fixed += 1
            else:
                n_still_wrong += 1

            rec = {
                "video_id": vid,
                "recovery_prompt": recovery_name,
                "resolution": "native_1280x720",
                "detail": "high",
                "gt_verdict": gt,
                "v6_verdict": v6_verdict,
                "recovery_verdict": recovery_verdict,
                "recovery_reasoning": reasoning,
                "full_json": parsed or {},
                "raw": raw,
                "usage": usage,
                "cost_usd": cost,
                "latency_s": round(latency, 2),
                "error": None,
            }

        except concurrent.futures.TimeoutError:
            elapsed = time.time() - clip_start
            msg = f"Wall-clock budget exceeded ({elapsed:.0f}s)"
            print(f"[TIMEOUT] {msg}")
            n_timeout += 1
            rec = _err_rec(vid, recovery_name, gt, v6_verdict, clip_start, msg)
        except Exception as exc:
            elapsed = time.time() - clip_start
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[ERR] {msg}")
            n_err += 1
            rec = _err_rec(vid, recovery_name, gt, v6_verdict, clip_start, msg)

        if rec is not None:
            _append(rec)

    print()
    print("=" * 65)
    print(f"Debate done.  FIXED={n_fixed}  still-wrong={n_still_wrong}  "
          f"errors={n_err}  timeouts={n_timeout}")
    total_debated = n_fixed + n_still_wrong + n_err + n_timeout
    print(f"After debate accuracy boost: +{n_fixed} clips flipped to correct")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Output: {OUT_JSONL}")
    print("=" * 65)


def _err_rec(vid, recovery_name, gt, v6_verdict, clip_start, msg) -> Dict:
    return {
        "video_id": vid,
        "recovery_prompt": recovery_name,
        "resolution": "native_1280x720",
        "detail": "high",
        "gt_verdict": gt,
        "v6_verdict": v6_verdict,
        "recovery_verdict": None,
        "recovery_reasoning": None,
        "full_json": {},
        "raw": "",
        "usage": {},
        "cost_usd": 0.0,
        "latency_s": round(time.time() - clip_start, 2),
        "error": msg,
    }


if __name__ == "__main__":
    main()

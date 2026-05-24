"""Run PROMPT_G_OPT_v6_balanced @ 1280x720, detail='high' on all 100 clips
from outputs/teacher_dataset_v11.xlsx.

Special case: clip 01552 is copied from v6_hires_full18.jsonl (already done).

Timeout safety:
  DEFAULT_TIMEOUT    = 90s   per HTTP attempt (OpenAI SDK)
  MAX_RETRIES        = 2     (3 total attempts with exponential backoff)
  CLIP_BUDGET_SECS   = 300s  hard wall-clock budget per clip via ThreadPoolExecutor
  If a clip exceeds CLIP_BUDGET_SECS it is recorded as error and skipped;
  the run continues to the next clip.

Resumable: keyed by video_id in v6_hires_v11.jsonl.
No BERTScore (no GT reasoning in v11).
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

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
PRICE_IN = 2.00
PRICE_OUT = 12.00
TEMPERATURE = 0.1

# ── Timeout / retry settings ────────────────────────────────────────────────
DEFAULT_TIMEOUT    = 90.0    # seconds per HTTP attempt
MAX_RETRIES        = 2       # retries after first failure (3 total attempts)
RETRY_DELAY        = 5.0     # base delay between retries (exponential backoff)
CLIP_BUDGET_SECS   = 300     # hard wall-clock cap per clip (covers all retries)
INTER_CALL_DELAY   = 1.5     # polite gap between clips
# ────────────────────────────────────────────────────────────────────────────

PROMPT_TOKEN_HARD_CAP = 100_000

V11_XLSX           = REPO_ROOT / "outputs" / "teacher_dataset_v11.xlsx"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"
OUT_DIR            = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips"
OUT_JSONL          = OUT_DIR / "v6_hires_v11.jsonl"
REUSE_JSONL        = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v6_hires_full18.jsonl"
PROMPT_FILE        = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_balanced.py"
PROMPT_VAR         = "PROMPT_G_OPT_v6_balanced"

REUSE_VID          = "01552"   # already in v6_hires_full18.jsonl — skip API call


def _load_prompt() -> str:
    spec = importlib.util.spec_from_file_location("_p_v6", PROMPT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, PROMPT_VAR)


def _load_jsonl(path: Path) -> Dict[str, Dict]:
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
            continue
    return out


def _append(rec: Dict) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_one(client, prompt: str, vid: str, gt_verdict: str, t_seconds: float) -> Dict:
    """Make the API call for a single clip. Runs inside a thread for timeout."""
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

    parsed, verdict = _parse_response(raw)
    cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)
    reasoning = parsed.get("verdict_reasoning") if parsed else None

    return {
        "video_id": vid,
        "prompt_name": "PROMPT_G_OPT_v6_balanced_hires",
        "resolution": "native_1280x720",
        "detail": "high",
        "gt_verdict": gt_verdict,
        "t_seconds": t_seconds,
        "verdict": verdict,
        "reasoning": reasoning,
        "full_json": parsed or {},
        "raw": raw,
        "usage": usage,
        "cost_usd": cost,
        "latency_s": round(latency, 2),
        "error": None,
    }


def main() -> None:
    prompt = _load_prompt()
    print(f"Prompt loaded: {PROMPT_VAR} ({len(prompt)} chars)")
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
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_v11_100clips"),
        },
    )

    df = pd.read_excel(V11_XLSX)
    clips = [
        {"video_id": f"{int(row['video_id']):05d}", "gt_verdict": row["gt_verdict"], "t_seconds": row["t_seconds"]}
        for _, row in df.iterrows()
    ]
    print(f"Total clips: {len(clips)}  (YES: {sum(c['gt_verdict']=='YES' for c in clips)}, "
          f"NO: {sum(c['gt_verdict']=='NO' for c in clips)})")

    existing = _load_jsonl(OUT_JSONL)
    done = sum(1 for v in existing.values() if v.get("verdict") is not None)
    print(f"Resume: {done}/{len(clips)} clips already done\n")

    # Load reuse source (01552)
    reuse_source = _load_jsonl(REUSE_JSONL)

    total_cost = 0.0
    n_ok = 0
    n_err = 0
    n_timeout = 0

    for idx, clip in enumerate(clips, start=1):
        vid = clip["video_id"]
        gt = clip["gt_verdict"]

        # Already done
        if vid in existing and existing[vid].get("verdict") is not None:
            rec = existing[vid]
            ok = "[OK]" if rec.get("verdict") == gt else "[XX]"
            print(f"[{idx:3d}/{len(clips)}] {vid} -- skip (already done, verdict={rec['verdict']} {ok})")
            total_cost += rec.get("cost_usd", 0.0)
            n_ok += 1
            continue

        # Special reuse: 01552 from prior run
        if vid == REUSE_VID:
            src = reuse_source.get(vid)
            if src:
                rec = dict(src)  # copy verbatim
                ok = "[OK]" if rec.get("verdict") == gt else "[XX]"
                print(f"[{idx:3d}/{len(clips)}] {vid} -- REUSE from v6_hires_full18 "
                      f"(verdict={rec.get('verdict')} {ok})")
                _append(rec)
                total_cost += rec.get("cost_usd", 0.0)
                n_ok += 1
                continue
            else:
                print(f"[{idx:3d}/{len(clips)}] {vid} -- reuse source not found, will re-run")

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        print(f"[{idx:3d}/{len(clips)}] {vid}  gt={gt}  t={clip['t_seconds']:.2f}s", end="  ", flush=True)

        clip_start = time.time()
        rec = None
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_one, client, prompt, vid, gt, float(clip["t_seconds"]))
                try:
                    rec = future.result(timeout=CLIP_BUDGET_SECS)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    elapsed = time.time() - clip_start
                    raise RuntimeError(
                        f"Wall-clock budget exceeded ({elapsed:.0f}s > {CLIP_BUDGET_SECS}s)"
                    )

            verdict = rec.get("verdict")
            ok = "[OK]" if verdict == gt else "[XX]"
            cost = rec.get("cost_usd", 0.0)
            usage = rec.get("usage") or {}
            print(
                f"verdict={verdict or '??':3s} {ok}  "
                f"cost=${cost:.4f}  "
                f"in_tok={usage.get('prompt_tokens')}  "
                f"out_tok={usage.get('completion_tokens')}  "
                f"{rec.get('latency_s', 0):.1f}s"
            )
            total_cost += cost
            n_ok += 1

        except concurrent.futures.TimeoutError:
            # Already re-raised above, but guard just in case
            elapsed = time.time() - clip_start
            msg = f"Wall-clock budget exceeded ({elapsed:.0f}s)"
            print(f"[TIMEOUT] {msg}")
            n_timeout += 1
            rec = {
                "video_id": vid, "prompt_name": "PROMPT_G_OPT_v6_balanced_hires",
                "resolution": "native_1280x720", "detail": "high",
                "gt_verdict": gt, "t_seconds": clip["t_seconds"],
                "verdict": None, "reasoning": None,
                "full_json": {}, "raw": "", "usage": {}, "cost_usd": 0.0,
                "latency_s": round(elapsed, 2), "error": msg,
            }
        except Exception as exc:
            elapsed = time.time() - clip_start
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[ERR] {msg}")
            n_err += 1
            rec = {
                "video_id": vid, "prompt_name": "PROMPT_G_OPT_v6_balanced_hires",
                "resolution": "native_1280x720", "detail": "high",
                "gt_verdict": gt, "t_seconds": clip["t_seconds"],
                "verdict": None, "reasoning": None,
                "full_json": {}, "raw": "", "usage": {}, "cost_usd": 0.0,
                "latency_s": round(elapsed, 2), "error": msg,
            }

        if rec is not None:
            _append(rec)

    print()
    print("=" * 65)
    print(f"Done. {n_ok} OK  |  {n_err} errors  |  {n_timeout} timeouts")
    print(f"Total cost: ${total_cost:.4f}")
    if n_err + n_timeout > 0:
        print("Re-run this script to retry failed/timed-out clips (resume-safe).")
    print(f"Output: {OUT_JSONL}")
    print("=" * 65)


if __name__ == "__main__":
    main()

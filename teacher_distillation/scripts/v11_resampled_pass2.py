"""v11-resampled Stage 3: v7.1 recovery.

Two batches:
  (A) FP recovery: clips that still predicted YES after pass-1 resample
      -> PROMPT_G_OPT_v7_1_TN_RECOVERY on <vid>_hires_early/ frames
  (B) FN recovery: all 15 v11 FN clips
      -> PROMPT_G_OPT_v7_1_TP_RECOVERY on the ORIGINAL <vid>_hires/ frames

Self-contained: no imports from teacher_bakeoff / teacher_prompt_bakeoff.
Resume-safe.

Output: outputs/prompt_bakeoff/v11_100clips_resampled/v7_1_recovery.jsonl
"""
from __future__ import annotations

import base64
import concurrent.futures
import importlib.util
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from PIL import Image

try:
    from openai import (
        APIConnectionError as _OAIConnectionError,
        APITimeoutError as _OAITimeoutError,
        OpenAI,
        RateLimitError as _OAIRateLimitError,
    )
except ImportError:
    from openai import OpenAI  # type: ignore[no-redef]
    _OAIConnectionError = Exception
    _OAITimeoutError = Exception
    _OAIRateLimitError = Exception

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_SLUG  = "google/gemini-3.1-pro-preview"
PRICE_IN    = 2.00
PRICE_OUT   = 12.00
TEMPERATURE = 0.1

DEFAULT_TIMEOUT       = 90.0
MAX_RETRIES           = 2
RETRY_DELAY           = 5.0
CLIP_BUDGET_SECS      = 300
INTER_CALL_DELAY      = 1.5
PROMPT_TOKEN_HARD_CAP = 100_000

DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"
PROMPT_TN_FILE      = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v7_1_TN_RECOVERY.py"
PROMPT_TP_FILE      = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v7_1_TP_RECOVERY.py"

PASS1_JSONL      = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips_resampled" / "v6_balanced_resampled_pass1.jsonl"
V11_HIRES_JSONL  = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips" / "v6_hires_v11.jsonl"
OUT_JSONL        = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips_resampled" / "v7_1_recovery.jsonl"

# 15 FN video_ids from leaderboard_v6_debate_v11.md
FN_CLIPS = [
    "00065", "00089", "00097", "00401", "00428", "00573", "00590", "00604",
    "00621", "00670", "00741", "00832", "00876", "01013", "01024",
]


# ---- Inlined utilities ----

def _normalize_verdict(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in {"YES", "NO"} else None


def _normalize_confidence(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in {"HIGH", "MEDIUM", "LOW"} else None


def _encode_image(path: Path, frame_size: int) -> str:
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (frame_size or 64, frame_size or 64), color=(0, 0, 0))
    if frame_size and img.size != (frame_size, frame_size):
        img = img.resize((frame_size, frame_size))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(prompt: str, image_b64s: Sequence[str], detail: str = "low") -> List[Dict]:
    content: List[Dict] = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        image_url: Dict = {"url": b64}
        if detail:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})
    return [{"role": "user", "content": content}]


def _extract_json_object(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def _parse_response(raw: str) -> Tuple[Optional[Dict], Optional[str]]:
    parsed = _extract_json_object(raw)
    if parsed is None:
        return None, None
    parsed["collision_verdict"] = _normalize_verdict(parsed.get("collision_verdict"))
    parsed["confidence"] = _normalize_confidence(parsed.get("confidence"))
    return parsed, parsed["collision_verdict"]


def _call_model(client, model, messages, timeout, max_retries, retry_delay,
                temperature=0.1, max_tokens=8192) -> Tuple[str, Dict]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                timeout=timeout, max_tokens=max_tokens,
            )
            text = response.choices[0].message.content if response.choices else ""
            usage = (response.usage.model_dump()
                     if hasattr(response, "usage") and response.usage else {})
            return text or "", usage
        except _OAITimeoutError as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}] timeout -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except _OAIRateLimitError as exc:
            last_exc = exc
            wait = retry_delay * 2 * (2 ** (attempt - 1))
            print(f"  [retry {attempt}] rate-limit -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except _OAIConnectionError as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}] connection -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}] error: {exc!r} -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_exc}") from last_exc


def _calc_cost(usage: Dict, price_in: float, price_out: float) -> float:
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    return in_tok * price_in / 1_000_000 + out_tok * price_out / 1_000_000


def _load_clip_frames(frames_root: Path, video_id: str, indices: List[int], frame_size: int) -> List[str]:
    folder = frames_root / video_id
    return [_encode_image(folder / f"frame_{i:05d}.jpg", frame_size) for i in indices]


# ---- Script-specific ----

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
            # composite key: video_id + recovery_prompt (since one vid could be in both batches in theory)
            key = (rec["video_id"], rec.get("recovery_prompt", ""))
            out[key] = rec
        except Exception:
            pass
    return out


def _append(path: Path, rec: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _call_with_budget(client, prompt: str, vid: str, frames_subdir: str) -> Tuple[str, Dict, float]:
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
            raise RuntimeError(f"Prompt token count {prompt_tok} > hard cap {PROMPT_TOKEN_HARD_CAP}")
        return raw, usage, latency

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_inner)
        try:
            return future.result(timeout=CLIP_BUDGET_SECS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise RuntimeError(f"Wall-clock budget {CLIP_BUDGET_SECS}s exceeded for {vid}")


def _run_one(client, prompt_text: str, prompt_name: str, vid: str, frames_subdir: str,
             gt: str, t_original, t_new, batch_label: str) -> Dict:
    print(f"  {vid}  GT={gt}  [{batch_label}]", end="  ", flush=True)
    clip_start = time.time()
    try:
        raw, usage, latency = _call_with_budget(client, prompt_text, vid, frames_subdir)
        parsed, verdict = _parse_response(raw)
        cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)
        reasoning = parsed.get("verdict_reasoning") if parsed else None
        outcome = "FIXED" if verdict == gt else "still-wrong"
        print(f"recovery={verdict or '??':3s}  {outcome}  cost=${cost:.4f}  "
              f"in_tok={usage.get('prompt_tokens')}  out_tok={usage.get('completion_tokens')}  "
              f"{latency:.1f}s")
        return {
            "video_id": vid,
            "recovery_prompt": prompt_name,
            "batch": batch_label,
            "resolution": "native_1280x720",
            "stride": 4,
            "detail": "high",
            "gt_verdict": gt,
            "t_original": t_original,
            "t_new": t_new,
            "frames_subdir": frames_subdir,
            "recovery_verdict": verdict,
            "recovery_reasoning": reasoning,
            "full_json": parsed or {},
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
        return {
            "video_id": vid, "recovery_prompt": prompt_name, "batch": batch_label,
            "resolution": "native_1280x720", "stride": 4, "detail": "high",
            "gt_verdict": gt, "t_original": t_original, "t_new": t_new,
            "frames_subdir": frames_subdir,
            "recovery_verdict": None, "recovery_reasoning": None, "full_json": {},
            "raw": "", "usage": {}, "cost_usd": 0.0,
            "latency_s": round(elapsed, 2), "error": msg,
        }


def main() -> None:
    tp_prompt = _load_prompt(PROMPT_TP_FILE, "PROMPT_G_OPT_v7_1_TP_RECOVERY")
    tn_prompt = _load_prompt(PROMPT_TN_FILE, "PROMPT_G_OPT_v7_1_TN_RECOVERY")
    print(f"Loaded TP_RECOVERY ({len(tp_prompt)} chars)")
    print(f"Loaded TN_RECOVERY ({len(tn_prompt)} chars)")

    # Identify FP clips that need TN_RECOVERY (still YES after resample)
    pass1 = {}
    for line in PASS1_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pass1[r["video_id"]] = r
    fp_still_yes = [vid for vid, r in pass1.items()
                    if r.get("verdict") == "YES" and r.get("gt_verdict") == "NO"]
    fp_still_yes.sort()
    print(f"\nFP clips still YES after resample (need TN_RECOVERY): {len(fp_still_yes)}")
    print(f"  {fp_still_yes}")
    print(f"\nFN clips (need TP_RECOVERY): {len(FN_CLIPS)}")

    # Load v11 first-pass for FN metadata
    v11 = {}
    for line in V11_HIRES_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        v11[r["video_id"]] = r

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title":      os.environ.get("OPENROUTER_APP_TITLE", "MMLM_v11_resampled_pass2"),
        },
    )

    existing = _load_jsonl_dict(OUT_JSONL)
    total_cost = 0.0
    n_fp_fixed = n_fp_still = 0
    n_fn_fixed = n_fn_still = 0
    n_err = 0

    print(f"\n=== BATCH A: FP recovery (TN_RECOVERY, {len(fp_still_yes)} clips) ===")
    for vid in fp_still_yes:
        key = (vid, "PROMPT_G_OPT_v7_1_TN_RECOVERY")
        gt = "NO"
        pass1_rec = pass1.get(vid, {})
        t_orig = pass1_rec.get("t_original")
        t_new = pass1_rec.get("t_new")
        frames_subdir = f"{vid}_hires_early"

        if key in existing and existing[key].get("recovery_verdict") is not None:
            r = existing[key]
            outcome = "FIXED" if r["recovery_verdict"] == gt else "still-wrong"
            print(f"  {vid} -- skip (done: recovery={r['recovery_verdict']} {outcome})")
            total_cost += r.get("cost_usd", 0.0)
            if outcome == "FIXED":
                n_fp_fixed += 1
            else:
                n_fp_still += 1
            continue

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)
        rec = _run_one(client, tn_prompt, "PROMPT_G_OPT_v7_1_TN_RECOVERY",
                       vid, frames_subdir, gt, t_orig, t_new, "FP_recovery")
        _append(OUT_JSONL, rec)
        if rec["error"]:
            n_err += 1
        elif rec["recovery_verdict"] == gt:
            n_fp_fixed += 1
        else:
            n_fp_still += 1
        total_cost += rec.get("cost_usd", 0.0)

    print(f"\n=== BATCH B: FN recovery (TP_RECOVERY, {len(FN_CLIPS)} clips) ===")
    for vid in FN_CLIPS:
        key = (vid, "PROMPT_G_OPT_v7_1_TP_RECOVERY")
        v11_rec = v11.get(vid, {})
        gt = v11_rec.get("gt_verdict", "YES")
        t_orig = v11_rec.get("t_seconds")
        frames_subdir = f"{vid}_hires"   # ORIGINAL frames

        if key in existing and existing[key].get("recovery_verdict") is not None:
            r = existing[key]
            outcome = "FIXED" if r["recovery_verdict"] == gt else "still-wrong"
            print(f"  {vid} -- skip (done: recovery={r['recovery_verdict']} {outcome})")
            total_cost += r.get("cost_usd", 0.0)
            if outcome == "FIXED":
                n_fn_fixed += 1
            else:
                n_fn_still += 1
            continue

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)
        rec = _run_one(client, tp_prompt, "PROMPT_G_OPT_v7_1_TP_RECOVERY",
                       vid, frames_subdir, gt, t_orig, t_orig, "FN_recovery")
        _append(OUT_JSONL, rec)
        if rec["error"]:
            n_err += 1
        elif rec["recovery_verdict"] == gt:
            n_fn_fixed += 1
        else:
            n_fn_still += 1
        total_cost += rec.get("cost_usd", 0.0)

    print()
    print("-" * 65)
    print(f"FP recovery: {n_fp_fixed}/{len(fp_still_yes)} FIXED, "
          f"{n_fp_still} still-wrong")
    print(f"FN recovery: {n_fn_fixed}/{len(FN_CLIPS)} FIXED, "
          f"{n_fn_still} still-wrong")
    print(f"Errors: {n_err}")
    print(f"Total cost (this run): ${total_cost:.4f}")
    print(f"Output: {OUT_JSONL}")
    print("-" * 65)


if __name__ == "__main__":
    main()

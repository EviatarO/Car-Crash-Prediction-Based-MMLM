"""v11-resampled Stage 2: pass-1 v6_balanced on 18 FP at new (t-4s) timestamps.

Reads frames from dataset/train/<vid>_hires_early/ (created by extract_resampled_frames.py).
Calls Gemini 3.1 Pro with PROMPT_G_OPT_v6_balanced at native 1280x720, detail=high.
Resume-safe (append to JSONL, skip clips already done).

Self-contained: no imports from teacher_bakeoff / teacher_prompt_bakeoff.

Output: outputs/prompt_bakeoff/v11_100clips_resampled/v6_balanced_resampled_pass1.jsonl
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

# ---- Config ----
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
PROMPT_FILE         = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_balanced.py"
RESAMPLE_LOG        = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips_resampled" / "resample_log.json"
V11_HIRES_JSONL     = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips" / "v6_hires_v11.jsonl"

OUT_JSONL = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips_resampled" / "v6_balanced_resampled_pass1.jsonl"

FP_CLIPS = [
    "01045", "01144", "01225", "01261", "01305", "01307", "01400", "01420",
    "01470", "01508", "01539", "01569", "01614", "01655", "01771", "01817",
    "01904", "02064",
]


# ---- Inlined utilities (proven in v7_1_full5_debate.py) ----

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


def _call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    timeout: float,
    max_retries: int,
    retry_delay: float,
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> Tuple[str, Dict]:
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
            print(f"  [retry {attempt}/{max_retries}] timeout -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except _OAIRateLimitError as exc:
            last_exc = exc
            wait = retry_delay * 2 * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{max_retries}] rate-limit -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except _OAIConnectionError as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{max_retries}] connection -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except Exception as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{max_retries}] error: {exc!r} -- waiting {wait:.1f}s", flush=True)
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
            out[rec["video_id"]] = rec
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


# ---- Main ----

def main() -> None:
    prompt = _load_prompt(PROMPT_FILE, "PROMPT_G_OPT_v6_balanced")
    print(f"Loaded v6_balanced ({len(prompt)} chars)")

    # Load resample_log for t_new per clip
    log_recs = json.loads(RESAMPLE_LOG.read_text(encoding="utf-8"))
    t_new_map = {r["video_id"]: r.get("t_new") for r in log_recs if "video_id" in r}

    # Load v11 first-pass for gt_verdict + t_original
    v11 = _load_jsonl_dict(V11_HIRES_JSONL)

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title":      os.environ.get("OPENROUTER_APP_TITLE", "MMLM_v11_resampled_pass1"),
        },
    )

    existing = _load_jsonl_dict(OUT_JSONL)
    total_cost = 0.0
    n_correct = n_wrong = n_err = 0

    print(f"=== v11 RESAMPLED PASS-1 (v6_balanced @ t-4s) ===")
    print(f"Already done: {sorted(existing.keys())}\n")

    for idx, vid in enumerate(FP_CLIPS, start=1):
        v11_rec = v11.get(vid, {})
        gt = v11_rec.get("gt_verdict")  # should be "NO" for all 18 FP
        t_orig = v11_rec.get("t_seconds")
        t_new = t_new_map.get(vid)

        if vid in existing and existing[vid].get("verdict") is not None:
            r = existing[vid]
            outcome = "FIXED" if r["verdict"] == gt else "still-wrong"
            print(f"[{idx}/18] {vid} -- skip (done: verdict={r['verdict']} {outcome})")
            total_cost += r.get("cost_usd", 0.0)
            if outcome == "FIXED":
                n_correct += 1
            else:
                n_wrong += 1
            continue

        frames_subdir = f"{vid}_hires_early"
        print(f"[{idx}/18] {vid}  GT={gt}  t_orig={t_orig}  t_new={t_new}", end="  ", flush=True)

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        clip_start = time.time()
        try:
            raw, usage, latency = _call_with_budget(client, prompt, vid, frames_subdir)
            parsed, verdict = _parse_response(raw)
            cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)
            reasoning = parsed.get("verdict_reasoning") if parsed else None

            outcome = "FIXED" if verdict == gt else "still-wrong"
            print(f"verdict={verdict or '??':3s}  {outcome}  "
                  f"cost=${cost:.4f}  in_tok={usage.get('prompt_tokens')}  "
                  f"out_tok={usage.get('completion_tokens')}  {latency:.1f}s")
            total_cost += cost
            if outcome == "FIXED":
                n_correct += 1
            else:
                n_wrong += 1

            rec = {
                "video_id": vid,
                "prompt_name": "PROMPT_G_OPT_v6_balanced",
                "resolution": "native_1280x720",
                "stride": 4,
                "detail": "high",
                "gt_verdict": gt,
                "t_original": t_orig,
                "t_new": t_new,
                "frames_subdir": frames_subdir,
                "verdict": verdict,
                "reasoning": reasoning,
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
            n_err += 1
            rec = {
                "video_id": vid, "prompt_name": "PROMPT_G_OPT_v6_balanced",
                "resolution": "native_1280x720", "stride": 4, "detail": "high",
                "gt_verdict": gt, "t_original": t_orig, "t_new": t_new,
                "frames_subdir": frames_subdir,
                "verdict": None, "reasoning": None, "full_json": {},
                "raw": "", "usage": {}, "cost_usd": 0.0,
                "latency_s": round(elapsed, 2), "error": msg,
            }
        _append(OUT_JSONL, rec)

    print()
    print("-" * 65)
    print(f"Done. FIXED (verdict=GT)={n_correct}  still-wrong={n_wrong}  errors={n_err}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Output: {OUT_JSONL}")
    print("-" * 65)


if __name__ == "__main__":
    main()

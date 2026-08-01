"""
semsup_v6_control_rerun.py
============================
Runs PROMPT_G_OPT_v6_balanced.py, UNMODIFIED, on the same 18 val_e3a.jsonl
clips against any OpenRouter model - originally built as a same-day control
to test whether v6's historical recorded result (72.2%/83.3% verdict accuracy,
mean reasoning score 6.28/6.78) is still reproducible against the original
model (google/gemini-3.1-pro-preview, the default here). It was NOT
reproducible (see outputs/prompt_bakeoff/semsup_val18/summary.md: rerunning
today scored 50.0%/4.61, most likely model drift on the "preview" alias) -
which is why this became a teacher-model bake-off tool: --model lets the same
v6 prompt be pointed at any OpenRouter-hosted vision model, so different
teacher candidates can be compared on identical ground (same prompt, same 18
clips, same image settings).

Confirmed byte-for-byte identical to the original v6_hires_full18.py run:
image encoding (_encode_image, same PIL logic), and (for the Gemini default)
model slug + temperature. One real difference found and fixed: the original
call capped max_tokens=8192 (teacher_bakeoff.py's _call_model) to reserve room
for reasoning tokens before the JSON output; this repo's newer scripts had
dropped it. Kept here as the default for every model, for consistency -
override with --max-tokens if a specific model needs something different.

Not importing teacher_bakeoff.py or Teacher_dataset_distill_v11.py directly -
both have a pre-existing broken top-level import (prompts/PROMPT_G2.py and
prompts/templates.py respectively no longer exist at those paths, following
a prompts/ reorganization). Unrelated to this script; left as-is. The needed
functions are copied below instead (same pattern as
semsup_caption_promptbakeoff.py).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

try:
    from openai import (
        APIConnectionError as _OAIConnectionError,
        APITimeoutError as _OAITimeoutError,
        RateLimitError as _OAIRateLimitError,
    )
except ImportError:
    _OAIConnectionError = Exception  # type: ignore[assignment,misc]
    _OAITimeoutError = Exception      # type: ignore[assignment,misc]
    _OAIRateLimitError = Exception    # type: ignore[assignment,misc]

from prompts.PROMPT_G_OPT_v6_balanced import PROMPT_G_OPT_v6_balanced  # noqa: E402

DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
VAL_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "val_e3a.jsonl"
FRAMES_ROOT = PROJECT_ROOT / "dataset" / "train"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "prompt_bakeoff" / "semsup_val18" / "raw_v6_control_rerun.jsonl"


def _encode_image(path: Path, frame_size: int) -> str:
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (frame_size, frame_size), color=(0, 0, 0))
    if frame_size and img.size != (frame_size, frame_size):
        img = img.resize((frame_size, frame_size))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(prompt: str, image_b64s: Sequence[str], detail: Optional[str]) -> List[Dict]:
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
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def _normalize_verdict(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().upper()
    return s if s in ("YES", "NO") else None


def _call_model(client: OpenAI, model: str, messages: List[Dict], timeout: float,
                 max_retries: int, retry_delay: float, temperature: float,
                 max_tokens: int) -> Tuple[str, Dict]:
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
        except (_OAITimeoutError, _OAIRateLimitError, _OAIConnectionError, Exception) as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{max_retries}] {exc!r} -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_exc}") from last_exc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                     help="OpenRouter model slug, e.g. qwen/qwen3.7-flash, openai/gpt-5.6-luna-pro")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--frame-size", type=int, default=0, help="0 = native resolution")
    ap.add_argument("--detail", default="high", choices=["low", "high", "auto"],
                     help="OpenAI-specific vision param; some providers (e.g. Qwen) may ignore it")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-delay", type=float, default=3.0)
    ap.add_argument("--resume", action="store_true",
                     help="skip video_ids already present in --out (append mode) instead of "
                          "overwriting - use to retry just the clips that failed last time")
    args = ap.parse_args()
    out_path = Path(args.out)

    load_dotenv()
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key,
        default_headers={"HTTP-Referer": "http://localhost", "X-Title": "MMLM_V6_Teacher_Bakeoff"},
    )

    val = [json.loads(l) for l in open(VAL_MANIFEST, encoding="utf-8") if l.strip()]

    done_ids = set()
    if args.resume and out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            l = l.strip()
            if l:
                done_ids.add(json.loads(l)["video_id"])
        val = [r for r in val if r["video_id"] not in done_ids]
        print(f"--resume: {len(done_ids)} already done, {len(val)} pending")

    print(f"Running PROMPT_G_OPT_v6_balanced on {len(val)} clips "
          f"(model={args.model}, temp={args.temperature}, frame_size={args.frame_size}, "
          f"detail={args.detail}, max_tokens={args.max_tokens})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    t0 = time.time()
    mode = "a" if args.resume else "w"
    with open(out_path, mode, encoding="utf-8") as out_f:
        for idx, row in enumerate(val, start=1):
            vid = row["video_id"]
            frame_dir = FRAMES_ROOT / row["frames_dir"]
            frame_paths = [frame_dir / f"frame_{i:05d}.jpg" for i in range(1, 17)]
            image_b64s = [_encode_image(p, frame_size=args.frame_size) for p in frame_paths]
            messages = _build_messages(PROMPT_G_OPT_v6_balanced, image_b64s, detail=args.detail)

            try:
                raw_text, usage = _call_model(
                    client, args.model, messages, timeout=args.timeout, max_retries=args.max_retries,
                    retry_delay=args.retry_delay, temperature=args.temperature, max_tokens=args.max_tokens,
                )
            except RuntimeError as e:
                print(f"  [{idx:2d}/{len(val)}] [FAIL] {vid}: {e}")
                n_failed += 1
                continue

            parsed = _extract_json_object(raw_text)
            if parsed is None:
                print(f"  [{idx:2d}/{len(val)}] [BAD-JSON] {vid}: {raw_text[:200]!r}")
                n_failed += 1
                continue
            verdict = _normalize_verdict(parsed.get("collision_verdict"))
            if verdict is None:
                print(f"  [{idx:2d}/{len(val)}] [BAD-VERDICT] {vid}: {parsed.get('collision_verdict')!r}")
                n_failed += 1
                continue

            out_row = {
                "video_id": vid,
                "gt_verdict": row["gt_verdict"],
                "verdict": verdict,
                "verdict_reasoning": parsed.get("verdict_reasoning", ""),
                "scene_context": parsed.get("scene_context", ""),
                "dynamic_objects": parsed.get("dynamic_objects", []),
                "temporal_analysis": parsed.get("temporal_analysis", ""),
            }
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_f.flush()
            n_ok += 1
            print(f"  [{idx:2d}/{len(val)}] {vid}: gt={row['gt_verdict']:3s} verdict={verdict:3s} "
                  f"{'OK' if verdict == row['gt_verdict'] else 'WRONG'}")

    print()
    print("=" * 70)
    print(f"DONE. ok={n_ok} failed={n_failed} wall={time.time()-t0:.0f}s")
    print(f"Output: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

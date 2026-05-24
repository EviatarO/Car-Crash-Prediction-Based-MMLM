"""Smoke test: run v6_hires_v11 pipeline on exactly ONE clip to verify
frame loading, API call, JSON parsing, and JSONL write all work before
committing to the full 100-clip run.

Picks the first clip from v11.xlsx that is NOT already in
outputs/prompt_bakeoff/v11_100clips/v6_hires_v11.jsonl.
Writes the result there so the full run can resume from it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

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
DEFAULT_TIMEOUT = 240.0
MAX_RETRIES = 3
RETRY_DELAY = 3.0

V11_XLSX = REPO_ROOT / "outputs" / "teacher_dataset_v11.xlsx"
OUT_DIR = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v11_100clips"
OUT_JSONL = OUT_DIR / "v6_hires_v11.jsonl"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"
PROMPT_FILE = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_balanced.py"
PROMPT_VAR = "PROMPT_G_OPT_v6_balanced"
PROMPT_TOKEN_HARD_CAP = 100_000


def _load_prompt() -> str:
    spec = importlib.util.spec_from_file_location("_p_v6", PROMPT_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, PROMPT_VAR)


def _load_existing() -> Dict[str, Dict]:
    if not OUT_JSONL.exists():
        return {}
    out: Dict[str, Dict] = {}
    for line in OUT_JSONL.read_text(encoding="utf-8").splitlines():
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


def main() -> None:
    prompt = _load_prompt()
    print(f"Prompt loaded: {PROMPT_VAR} ({len(prompt)} chars)")

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_v11_smoke"),
        },
    )

    df = pd.read_excel(V11_XLSX)
    existing = _load_existing()
    print(f"Already done in JSONL: {len(existing)} clips\n")

    # Pick first clip not yet in JSONL
    test_clip = None
    for _, row in df.iterrows():
        vid = f"{int(row['video_id']):05d}"
        if vid not in existing:
            test_clip = {"video_id": vid, "gt_verdict": row["gt_verdict"], "t_seconds": row["t_seconds"]}
            break

    if test_clip is None:
        print("All clips already done — nothing to test.")
        return

    vid = test_clip["video_id"]
    gt = test_clip["gt_verdict"]
    print(f"Smoke test clip: {vid}  GT={gt}  t={test_clip['t_seconds']:.2f}s")

    # Verify frames exist
    frames_dir = DEFAULT_FRAMES_ROOT / f"{vid}_hires"
    frame_files = sorted(frames_dir.glob("frame_*.jpg")) if frames_dir.exists() else []
    print(f"Frames dir: {frames_dir}  ({len(frame_files)} jpgs found)")
    if len(frame_files) < 16:
        raise SystemExit(f"Frame dir missing or incomplete: {frames_dir}")

    indices = list(range(1, 17))
    b64s = _load_clip_frames(DEFAULT_FRAMES_ROOT, f"{vid}_hires", indices, frame_size=0)
    print(f"Loaded {len(b64s)} base64 frames OK")

    messages = _build_messages(prompt, b64s, detail="high")
    print(f"Built messages: {len(messages)} message(s), "
          f"image parts in user turn: {sum(1 for c in messages[-1]['content'] if isinstance(c, dict) and c.get('type') == 'image_url')}")

    print("Calling Gemini...")
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

    ok_str = "[OK]" if verdict == gt else "[XX]"
    print(f"\n  verdict={verdict}  {ok_str}")
    print(f"  reasoning: {(reasoning or '')[:200]}...")
    print(f"  in_tok={usage.get('prompt_tokens')}  out_tok={usage.get('completion_tokens')}")
    print(f"  cost=${cost:.4f}  latency={latency:.1f}s")

    rec = {
        "video_id": vid,
        "prompt_name": "PROMPT_G_OPT_v6_balanced_hires",
        "resolution": "native_1280x720",
        "detail": "high",
        "gt_verdict": gt,
        "t_seconds": test_clip["t_seconds"],
        "verdict": verdict,
        "reasoning": reasoning,
        "full_json": parsed or {},
        "raw": raw,
        "usage": usage,
        "cost_usd": cost,
        "latency_s": round(latency, 2),
        "error": None,
    }
    _append(rec)
    print(f"\nAppended to: {OUT_JSONL}")
    print("\n=== SMOKE TEST PASSED ===")


if __name__ == "__main__":
    main()

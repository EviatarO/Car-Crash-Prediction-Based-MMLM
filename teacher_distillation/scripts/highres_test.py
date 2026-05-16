"""High-resolution diagnostic: PROMPT_G_OPT_v6_balanced on 6 problem clips
at NATIVE resolution (1280x720) with Gemini detail='high'.

Tests whether the 4 universal failures (00529, 01153, 01281, 01504) and 2
v6 hallucinations (00372, 01737) flip to correct when given higher visual detail.

Resumable via outputs/prompt_bakeoff/highres_test.jsonl key=(video_id).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

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
PRICE_IN = 2.00
PRICE_OUT = 12.00
TEMPERATURE = 0.1
DEFAULT_TIMEOUT = 240.0  # high-detail calls may be slower
MAX_RETRIES = 3
RETRY_DELAY = 3.0
INTER_CALL_DELAY = 1.5

CLIPS_TO_RUN = ["00529", "01153", "01281", "01504", "00372", "01737"]

DEFAULT_GT_XLSX = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"
OUT_JSONL = REPO_ROOT / "outputs" / "prompt_bakeoff" / "highres_test.jsonl"

PROMPT_FILE = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_balanced.py"
PROMPT_VAR = "PROMPT_G_OPT_v6_balanced"


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
    print(f"Loaded prompt: {PROMPT_VAR} ({len(prompt)} chars)")

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_Highres"),
        },
    )

    clips_all = _read_gt_excel_with_en(DEFAULT_GT_XLSX)
    clip_map = {c["video_id"]: c for c in clips_all}
    clips = [clip_map[v] for v in CLIPS_TO_RUN if v in clip_map]
    print(f"Clips to run: {[c['video_id'] for c in clips]}")

    print("Warming up BERTScore...")
    warmup_bertscore()
    print("BERTScore ready.\n")

    existing = _load_existing()
    done = sum(1 for v in existing.values() if v.get("verdict") is not None)
    print(f"Resume: {done}/{len(clips)} clips already done\n")

    total_cost = 0.0
    for idx, clip in enumerate(clips, start=1):
        vid = clip["video_id"]
        rec_existing = existing.get(vid)
        if rec_existing and rec_existing.get("verdict") is not None:
            print(f"[{idx}/{len(clips)}] {vid} -- already done (verdict={rec_existing['verdict']})")
            total_cost += rec_existing.get("cost_usd", 0.0)
            continue

        print(f"[{idx}/{len(clips)}] video={vid} target={clip['target']} gt={clip['gt_verdict']}")
        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        # Load NATIVE resolution frames (frame_size=0 → no resize in _encode_image)
        indices = list(range(1, 17))
        b64s = _load_clip_frames(DEFAULT_FRAMES_ROOT, f"{vid}_hires", indices, frame_size=0)

        # Build messages with detail="high" (NOT "low")
        messages = _build_messages(prompt, b64s, detail="high")

        t0 = time.time()
        try:
            raw, usage = _call_model(
                client, MODEL_SLUG, messages,
                timeout=DEFAULT_TIMEOUT, max_retries=MAX_RETRIES,
                retry_delay=RETRY_DELAY, temperature=TEMPERATURE,
            )
            latency = time.time() - t0
            parsed, verdict = _parse_response(raw)
            cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)
            reasoning = parsed.get("verdict_reasoning") if parsed else None
            sb = score_one(verdict, reasoning, clip["gt_verdict"], clip["gt_reasoning_en"])
            rec = {
                "video_id": vid,
                "prompt_name": "PROMPT_G_OPT_v6_balanced_hires",
                "resolution": "native_1280x720",
                "detail": "high",
                "gt_verdict": clip["gt_verdict"],
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
            ok = "[OK]" if verdict == clip["gt_verdict"] else "[XX]"
            print(
                f"    verdict={verdict or '??':3s} {ok}  BERT={sb.alignment:.3f}  "
                f"words={sb.word_count:3d}  cost=${cost:.4f}  in_tok={usage.get('prompt_tokens')}  "
                f"out_tok={usage.get('completion_tokens')}  {latency:.1f}s"
            )
            total_cost += cost
        except Exception as exc:
            latency = time.time() - t0
            print(f"    ERROR: {exc}")
            rec = {
                "video_id": vid,
                "prompt_name": "PROMPT_G_OPT_v6_balanced_hires",
                "resolution": "native_1280x720",
                "detail": "high",
                "gt_verdict": clip["gt_verdict"],
                "target": clip["target"],
                "t_seconds": clip["t_seconds"],
                "verdict": None, "reasoning": None,
                "full_json": {}, "scores": {"composite": 0.0, "verdict": 0.0,
                                            "alignment": 0.0, "length": 0.0, "word_count": 0},
                "raw": "", "usage": {}, "cost_usd": 0.0,
                "latency_s": round(latency, 2),
                "error": str(exc),
            }
        _append(rec)

    print()
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Wrote: {OUT_JSONL}")


if __name__ == "__main__":
    main()

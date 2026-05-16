"""Debate / second-opinion pass for clips where v6@hires got the verdict wrong.

For each failing clip in the merged v6@hires set (highres_test.jsonl +
v6_hires_full18.jsonl):
- if GT=YES, v6=NO (FN) -> run PROMPT_G_OPT_v6_TP_RECOVERY  (proactive hazard analyst)
- if GT=NO,  v6=YES (FP) -> run PROMPT_G_OPT_v6_TN_RECOVERY  (conservative auditor)

Same hi-res frames (<vid>_hires/), same detail='high', same 16-frame window.

Resumable via outputs/prompt_bakeoff/v6_debate.jsonl key=video_id.
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
DEFAULT_TIMEOUT = 240.0
MAX_RETRIES = 3
RETRY_DELAY = 3.0
INTER_CALL_DELAY = 1.5

DEFAULT_GT_XLSX = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"
DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"

HIRES_JSONL_1 = REPO_ROOT / "outputs" / "prompt_bakeoff" / "highres_test.jsonl"
HIRES_JSONL_2 = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v6_hires_full18.jsonl"

OUT_JSONL = REPO_ROOT / "outputs" / "prompt_bakeoff" / "v6_debate.jsonl"

PROMPT_TP_FILE = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_TP_RECOVERY.py"
PROMPT_TN_FILE = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_TN_RECOVERY.py"

PROMPT_TOKEN_HARD_CAP = 100_000


def _load_prompt_by_var(var_name: str, candidate_files: List[Path]) -> str:
    """Search the given files for the requested PROMPT variable.

    The two recovery prompt files currently have their variable names swapped
    relative to filenames, so we search both files and return the body whose
    variable name matches what's asked for."""
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
        f"Variable '{var_name}' not found in any of: "
        f"{[p.name for p in candidate_files]}"
    )


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _merge_v6_hires() -> Dict[str, Dict]:
    """Return {video_id -> v6 hires record} merged from both JSONLs."""
    out: Dict[str, Dict] = {}
    for path in (HIRES_JSONL_1, HIRES_JSONL_2):
        for rec in _load_jsonl(path):
            vid = rec.get("video_id")
            if vid is None or rec.get("verdict") is None:
                continue
            out[vid] = rec
    return out


def _load_existing() -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for rec in _load_jsonl(OUT_JSONL):
        out[rec["video_id"]] = rec
    return out


def _append(rec: Dict) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    candidates = [PROMPT_TP_FILE, PROMPT_TN_FILE]
    tp_prompt = _load_prompt_by_var("PROMPT_G_OPT_v6_TP_RECOVERY", candidates)
    tn_prompt = _load_prompt_by_var("PROMPT_G_OPT_v6_TN_RECOVERY", candidates)
    print(f"Loaded TP_RECOVERY ({len(tp_prompt)} chars)")
    print(f"Loaded TN_RECOVERY ({len(tn_prompt)} chars)\n")

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_Debate"),
        },
    )

    # Identify failing clips
    v6 = _merge_v6_hires()
    failures = []
    for vid, rec in v6.items():
        if rec["verdict"] != rec["gt_verdict"]:
            failures.append((vid, rec))
    failures.sort(key=lambda x: x[0])

    print(f"v6@hires records loaded: {len(v6)}")
    print(f"Failing clips (verdict != gt): {len(failures)}")
    for vid, rec in failures:
        kind = "FN" if rec["gt_verdict"] == "YES" else "FP"
        print(f"  {vid}  GT={rec['gt_verdict']}  v6={rec['verdict']}  [{kind}]")
    print()

    if not failures:
        print("No failures to debate. Exiting.")
        return

    clips_all = _read_gt_excel_with_en(DEFAULT_GT_XLSX)
    clip_map = {c["video_id"]: c for c in clips_all}

    print("Warming up BERTScore...")
    warmup_bertscore()
    print("BERTScore ready.\n")

    existing = _load_existing()
    total_cost = 0.0

    for idx, (vid, v6_rec) in enumerate(failures, start=1):
        if vid in existing and existing[vid].get("recovery_verdict") is not None:
            print(f"[{idx}/{len(failures)}] {vid} -- already done "
                  f"(recovery={existing[vid]['recovery_verdict']})")
            total_cost += existing[vid].get("cost_usd", 0.0)
            continue

        clip = clip_map[vid]
        gt = v6_rec["gt_verdict"]
        if gt == "YES":
            recovery_name = "PROMPT_G_OPT_v6_TP_RECOVERY"
            prompt = tp_prompt
        else:
            recovery_name = "PROMPT_G_OPT_v6_TN_RECOVERY"
            prompt = tn_prompt

        print(f"[{idx}/{len(failures)}] {vid}  GT={gt}  v6={v6_rec['verdict']}  "
              f"-> {recovery_name}")
        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        indices = list(range(1, 17))
        b64s = _load_clip_frames(DEFAULT_FRAMES_ROOT, f"{vid}_hires", indices, frame_size=0)
        messages = _build_messages(prompt, b64s, detail="high")

        t0 = time.time()
        try:
            raw, usage = _call_model(
                client, MODEL_SLUG, messages,
                timeout=DEFAULT_TIMEOUT, max_retries=MAX_RETRIES,
                retry_delay=RETRY_DELAY, temperature=TEMPERATURE,
            )
            latency = time.time() - t0
            prompt_tok = usage.get("prompt_tokens", 0) if usage else 0
            if prompt_tok and prompt_tok > PROMPT_TOKEN_HARD_CAP:
                raise RuntimeError(
                    f"Prompt token count {prompt_tok} exceeded hard cap "
                    f"{PROMPT_TOKEN_HARD_CAP} — aborting"
                )
            parsed, verdict = _parse_response(raw)
            cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)
            reasoning = parsed.get("verdict_reasoning") if parsed else None
            sb = score_one(verdict, reasoning, gt, clip["gt_reasoning_en"])
            rec = {
                "video_id": vid,
                "recovery_prompt": recovery_name,
                "resolution": "native_1280x720",
                "detail": "high",
                "gt_verdict": gt,
                "target": clip["target"],
                "t_seconds": clip["t_seconds"],
                "v6_verdict": v6_rec["verdict"],
                "recovery_verdict": verdict,
                "recovery_reasoning": reasoning,
                "full_json": parsed or {},
                "scores": sb.to_dict(),
                "raw": raw,
                "usage": usage,
                "cost_usd": cost,
                "latency_s": round(latency, 2),
                "error": None,
            }
            flipped = "FIXED" if verdict == gt else ("BROKE" if verdict != gt and v6_rec["verdict"] == gt else "still-wrong")
            print(
                f"    recovery={verdict or '??':3s}  flipped={flipped}  BERT={sb.alignment:.3f}  "
                f"cost=${cost:.4f}  in_tok={usage.get('prompt_tokens')}  "
                f"out_tok={usage.get('completion_tokens')}  {latency:.1f}s"
            )
            total_cost += cost
        except Exception as exc:
            latency = time.time() - t0
            print(f"    ERROR: {exc}")
            rec = {
                "video_id": vid,
                "recovery_prompt": recovery_name,
                "resolution": "native_1280x720",
                "detail": "high",
                "gt_verdict": gt,
                "target": clip["target"],
                "t_seconds": clip["t_seconds"],
                "v6_verdict": v6_rec["verdict"],
                "recovery_verdict": None,
                "recovery_reasoning": None,
                "full_json": {},
                "scores": {"composite": 0.0, "verdict": 0.0,
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

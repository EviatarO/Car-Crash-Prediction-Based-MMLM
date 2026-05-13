"""APO Proposer: ProTeGi-style failure-driven instruction generator.

Takes the current best instruction + N worst-scoring training clips (with their
GT and actual model outputs), asks Claude Opus to:
  1. Diagnose the systematic failure pattern
  2. Propose K alternative instructions targeting that pattern

Returns a list of (diagnosis, instruction) pairs.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


PROPOSER_MODEL = "anthropic/claude-opus-4.7"
PROPOSER_PRICE_IN = 5.00   # USD per 1M tokens
PROPOSER_PRICE_OUT = 25.00


def build_failure_brief(
    beam: List[Dict],
    worst_clips: List[Dict],
    n_candidates: int = 5,
    score_history: Optional[List[Dict]] = None,
) -> str:
    """Construct the meta-prompt sent to Claude — beam-aware.

    Args:
        beam: list of current top instructions, each as dict with:
            - instruction (str)
            - mean_composite (float)
        worst_clips: list of failure dicts pooled from across beam members. Each:
            - video_id, gt_verdict, gt_reasoning_en
            - pred_verdict, pred_reasoning, pred_temporal, pred_spatial
            - composite, verdict_score, alignment_score, length_score, word_count
            - failure_type ("FN" / "FP" / "verdict_correct_low_alignment" / "length_overflow")
            - source_label (str): e.g. "from beam member 2"
        n_candidates: how many candidate instructions Claude should propose
        score_history: optional list of {"instruction": str, "score": float}

    Returns:
        A formatted string prompt for Claude.
    """
    lines = [
        "You are optimizing a prompt instruction for a vision-language model that predicts",
        "ego-vehicle collisions from 16 dashcam frames.",
        "",
        "The model's output schema is FIXED (cannot be changed). Each prediction includes:",
        "scene_context, ego_state, dynamic_objects, temporal_analysis,",
        "spatiotemporal_attention, time_to_contact, collision_verdict (YES/NO),",
        "verdict_reasoning (max 150 words).",
        "",
        f"CURRENT BEAM (top-{len(beam)} instructions, ranked by mean composite score):",
    ]
    for i, m in enumerate(beam, start=1):
        lines += [
            f'  [Beam #{i}] mean_composite={m["mean_composite"]:.3f}',
            f'    instruction: "{m["instruction"]}"',
        ]
    lines.append("")

    if score_history and len(score_history) > 1:
        lines.append("RECENT BEST-IN-BEAM HISTORY (most recent last):")
        for h in score_history[-4:]:
            lines.append(f"  iter={h.get('iteration', '?')}  best_in_beam={h['mean_composite']:.3f}")
        lines.append("")

    lines += [
        f"BELOW ARE THE LOWEST-SCORING (clip, beam-member) PAIRS POOLED ACROSS THE WHOLE BEAM.",
        "Each shows which beam member produced this failure, the ground truth, the model's actual",
        "output, and per-component scores. Use these to diagnose patterns shared across the beam.",
        "",
    ]

    for i, clip in enumerate(worst_clips, start=1):
        lines += [
            f"--- Failure {i}: {clip.get('source_label', '?')}  video_id={clip['video_id']}  gt_verdict={clip['gt_verdict']}  composite={clip['composite']:.3f} ---",
            f"  GT reasoning: {clip['gt_reasoning_en'][:600]}",
            f"  Model verdict:        {clip['pred_verdict']}  ({'✅' if clip['verdict_score'] == 1.0 else '❌'})",
            f"  Model reasoning:      {(clip['pred_reasoning'] or '')[:400]}",
            f"  Model spatiotemporal: {(clip.get('pred_spatial') or '')[:200]}",
            f"  Model temporal:       {(clip.get('pred_temporal') or '')[:200]}",
            f"  Per-component: verdict={clip['verdict_score']:.2f}  alignment={clip['alignment_score']:.2f}  length={clip['length_score']:.2f}  words={clip['word_count']}",
            f"  Failure type: {clip['failure_type']}",
            "",
        ]

    lines += [
        "TASK:",
        "1. DIAGNOSE: identify the systematic weakness shared across the beam that produces these failures.",
        "   Compare what's common between the beam members and where they all fail similarly.",
        "   Be specific (e.g. 'all beam members lack explicit guidance for ruling OUT collision when ego decelerates').",
        "",
        f"2. PROPOSE: write {n_candidates} DIVERSE alternative instructions targeting the diagnosed weakness.",
        "   Constraints for each:",
        "   - 1 to 3 sentences, ≤80 words total",
        "   - Address the diagnosed failure pattern directly",
        "   - Take DIFFERENT angles across the candidates (don't propose 5 near-duplicates)",
        "   - Do NOT redefine the output schema (the 8 fields are already enforced)",
        "   - Do NOT prescribe an exhaustive analysis protocol — be concise; let the model reason",
        "",
        "Return ONLY a JSON array (no surrounding text):",
        '[',
        '  {"diagnosis": "...", "instruction": "..."},',
        '  ... (' + str(n_candidates) + ' total)',
        ']',
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

def _extract_json_array(raw: str) -> Optional[List[Dict]]:
    """Robust JSON-array extraction: try whole, then fenced block, then [..] substring."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return None


def propose_candidates(
    client: OpenAI,
    failure_brief: str,
    n: int = 3,
    timeout: float = 120.0,
    max_retries: int = 3,
    retry_delay: float = 3.0,
) -> Tuple[List[Dict], Dict]:
    """Call Claude with the failure brief, get N candidate instructions.

    Returns:
        (candidates, usage_info)
        candidates: list of dicts with keys {"diagnosis", "instruction"}
        usage_info: dict with prompt_tokens, completion_tokens, cost_usd, latency_s
    """
    messages = [
        {"role": "user", "content": failure_brief},
    ]

    last_exc = None
    raw = ""
    usage = {}
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=PROPOSER_MODEL,
                messages=messages,
                temperature=0.9,    # high diversity for proposals
                timeout=timeout,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            break
        except Exception as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [proposer retry {attempt}/{max_retries}] {exc!r} -- waiting {wait:.1f}s", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
    latency_s = time.time() - t0

    if not raw:
        raise RuntimeError(f"Proposer call failed: {last_exc}") from last_exc

    candidates = _extract_json_array(raw)
    if not candidates:
        raise RuntimeError(f"Proposer returned non-JSON: {raw[:300]}...")

    # Take first n candidates
    candidates = candidates[:n]

    # Validate each candidate has required fields
    cleaned = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        instruction = c.get("instruction", "").strip()
        diagnosis = c.get("diagnosis", "").strip()
        if instruction:
            cleaned.append({"diagnosis": diagnosis, "instruction": instruction})

    if not cleaned:
        raise RuntimeError(f"Proposer returned no valid candidates: {raw[:300]}...")

    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    cost = in_tok * PROPOSER_PRICE_IN / 1_000_000 + out_tok * PROPOSER_PRICE_OUT / 1_000_000

    return cleaned, {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "cost_usd": cost,
        "latency_s": round(latency_s, 2),
        "raw": raw,
    }

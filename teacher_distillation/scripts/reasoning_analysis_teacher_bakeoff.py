"""reasoning_analysis_teacher_bakeoff.py -- compares two candidate teacher
models (Qwen3.7 Flash, GPT-5.6 Luna Pro) against the current teacher's
same-day baseline (Gemini 3.1 Pro Preview, re-scored from
reasoning_analysis_semsup_val18.py's v6-rerun-today arm), all using the
UNMODIFIED PROMPT_G_OPT_v6_balanced prompt on the 18-clip GT validation set.

Same rubric and structure as reasoning_analysis_semsup_val18.py: 0-10
qualitative score + rationale per clip, hardcoded after reading each
reasoning against gt_reasoning_en, plus BERTScore-F1 (apo_metric) as a
secondary metric.

Reads:
- dataset/manifests/val_e3a.jsonl
- outputs/prompt_bakeoff/semsup_val18/raw_v6_control_rerun.jsonl (Gemini today)
- outputs/prompt_bakeoff/semsup_val18/raw_v6_qwen37flash.jsonl
- outputs/prompt_bakeoff/semsup_val18/raw_v6_gpt56lunapro.jsonl

Writes:
- outputs/prompt_bakeoff/semsup_val18/reasoning_analysis_teacher_bakeoff.xlsx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from apo_metric import score_one, warmup_bertscore  # noqa: E402

VAL_MANIFEST = REPO_ROOT / "dataset" / "manifests" / "val_e3a.jsonl"
OUT_DIR = REPO_ROOT / "outputs" / "prompt_bakeoff" / "semsup_val18"
GEMINI_JSONL = OUT_DIR / "raw_v6_control_rerun.jsonl"
QWEN_JSONL = OUT_DIR / "raw_v6_qwen37flash.jsonl"
GPT_JSONL = OUT_DIR / "raw_v6_gpt56lunapro.jsonl"
OUT_XLSX = OUT_DIR / "reasoning_analysis_teacher_bakeoff.xlsx"

# Gemini-today scores are reused verbatim from
# reasoning_analysis_semsup_val18.py's V6_RERUN_SCORES (same run, same
# rationale) - not re-scored here to avoid re-litigating an already-published
# judgment against a different scoring session.
GEMINI_TODAY_SCORES = {
    "00077": (8, "Braking lead vehicle, critical closing speed, rear-end imminent - matches GT; doesn't state the merge-in event"),
    "00147": (7, "Correct verdict; blames the other vehicle 'cutting across' rather than EGO's own deviation - same attribution error as the original v6 run"),
    "00283": (7, "Pickup+trailer from the right shoulder matches GT's agent and origin; misses the stationary-then-turns/blocking nuance"),
    "00319": (9, "Close paraphrase of GT: crossing vehicle from the right, rapid distance decrease, insufficient space"),
    "00372": (2, "Misses the stopped-sedan/pedestrian-crosswalk hazard entirely; wrong verdict"),
    "00474": (1, "Same miss as V2/V3: 'no sudden maneuvers' when GT's entire event is the van's left turn"),
    "00493": (8, "Braking lead vehicle, EGO fails to maintain distance, critically short gap - matches GT's core mechanism and verdict"),
    "00529": (1, "Misses the forced-merge-due-to-obstruction event the ORIGINAL v6 run uniquely caught (score 10)"),
    "00687": (8, "Grey SUV merging into ego lane, converging trajectory, rapid closure - matches GT and verdict"),
    "01153": (1, "Reproduces the near-identical hallucinated left-turning white sedan as the original v6 run, V2, and V3"),
    "01281": (1, "Fabricates an abrupt black-SUV lane change not in GT at all"),
    "01504": (1, "Same red/dark SUV colour confusion as V2/V3, but drives a wrong verdict where V2/V3 both got this one right"),
    "01550": (8, "Matches GT's calm, controlled-closing framing; correct verdict"),
    "01552": (1, "Fabricates a rear-end hazard from a box truck GT does not support"),
    "01643": (9, "Clean empty-road match to GT, no extraneous details"),
    "01737": (9, "Matches GT's empty curving night road closely"),
    "02104": (1, "Hallucinates a 'flatbed truck' rear-end hazard, same fabricated object V3 also produced independently"),
    "02117": (1, "Reproduces the same black-SUV-merge hallucination as the original v6 run, V2, and V3 - 4-way agreement on this exact wrong reading"),
}

QWEN_SCORES = {
    "00077": (8, "Following distance, brake lights, critically short gap - matches GT's core mechanism and verdict; doesn't state the merge-in"),
    "00147": (2, "Frames the situation as safe parallel motion; misses EGO's own deviation into the other vehicle's lane entirely"),
    "00283": (1, "Describes the pickup diverging AWAY into an exit lane - the opposite direction of GT's blocking left-turn; wrong verdict"),
    "00319": (1, "Describes an open road with no crossing agent; misses GT's crossing vehicle entirely"),
    "00372": (2, "Describes controlled, safe following; misses the stopping-sedan/pedestrian hazard"),
    "00474": (1, "States the van moves 'parallel... without aggressive merging' - directly contradicts GT's left-turn event"),
    "00493": (3, "Assumes EGO will react safely to the braking sedan, opposite of GT's claim EGO does not slow"),
    "00529": (1, "States the silver SUV moves parallel 'without encroaching' - opposite of GT's drift-into-lane claim"),
    "00687": (9, "'Aggressively merging left... crossed lane divider... converging with insufficient gap' - excellent match to GT's mechanism and verdict"),
    "01153": (6, "Correctly identifies a crossing vehicle but recognizes it is CLEARING the intersection rather than escalating - avoids the hallucination trap every Gemini run fell into on this clip"),
    "01281": (7, "Captures the scene without fabricating a merge event (unlike every Gemini run on this clip); correct verdict"),
    "01504": (6, "Correct controlled-braking outcome; 'red minivan' doesn't match GT's 'dark SUV' naming"),
    "01550": (5, "Places the braking vehicle in an adjacent lane rather than directly ahead as GT states; correct verdict via a different scene read"),
    "01552": (6, "Reasonably captures the gas-station-adjacent scene (minivan from left, truck ahead); correct verdict"),
    "01643": (6, "Correct empty-road verdict; introduces adjacent-lane traffic not mentioned in GT"),
    "01737": (9, "Explicitly names the overpass, matching GT's 'bridge with lighting' closely"),
    "02104": (5, "Misses the merging white sedan event but reaches the correct verdict via a different, still-plausible safe reading"),
    "02117": (7, "Perceives the black SUV/van but correctly judges it a non-threatening merge rather than escalating to collision - the FIRST run in this whole investigation to get this clip right"),
}

GPT56LUNA_SCORES = {
    "00077": (8, "Braking lead vehicle, rapidly shrinking gap - matches GT and verdict; invents a 'Honda' brand detail not in GT"),
    "00147": (2, "Frames the right-side vehicle as parallel rather than considering EGO's own deviation; misses GT's causal mechanism"),
    "00283": (6, "Captures a vehicle crossing INTO the ego lane (correct direction and verdict) but misidentifies it as an SUV rather than the pickup+trailer"),
    "00319": (1, "No clear lateral movement noted; misses GT's crossing vehicle from the right"),
    "00372": (2, "Misses the stopping-sedan/pedestrian hazard; hedges toward safety"),
    "00474": (1, "Same miss as every other run: van framed as parallel, not turning"),
    "00493": (3, "Notes the braking sedan but assumes safe separation, opposite of GT's claim"),
    "00529": (1, "Describes stable, lane-aligned traffic; misses the forced-merge event"),
    "00687": (1, "States the grey SUV 'shifts toward the right edge, consistent with... diverging motion' - the opposite direction from GT's drift-into-lane claim"),
    "01153": (6, "Correctly notes the crossing sedan is 'already moving away' rather than escalating - avoids the hallucination trap"),
    "01281": (7, "Correct verdict, no fabricated merge; invents a 'Lexus' brand detail"),
    "01504": (6, "Correct controlled-braking outcome; 'minivan' doesn't match GT's 'dark SUV' naming"),
    "01550": (8, "Correctly places the lead vehicle AHEAD (matching GT) with stable lateral separation - closest match to GT's framing among all four teacher runs on this clip"),
    "01552": (7, "Reasonably captures the gas-station scene; 'commercial property' nicely matches GT's context"),
    "01643": (6, "Correct empty-road verdict; introduces parked vehicles not in GT"),
    "01737": (8, "Good match to GT's clear curving-ramp scene, doesn't explicitly name the bridge"),
    "02104": (7, "Explicitly considers whether the sedan merges into the lane and correctly judges it does not clearly do so - closest reasoning to GT's 'reasonable distances maintained' framing"),
    "02117": (7, "Correctly judges the black SUV motion as non-threatening/diverging rather than escalating - the SECOND run (with Qwen) to get this clip right after every Gemini run failed it"),
}


def _load_jsonl(p: Path) -> dict:
    return {json.loads(l)["video_id"]: json.loads(l) for l in open(p, encoding="utf-8") if l.strip()}


def main():
    warmup_bertscore()

    val = _load_jsonl(VAL_MANIFEST)
    gemini = _load_jsonl(GEMINI_JSONL)
    qwen = _load_jsonl(QWEN_JSONL)
    gpt = _load_jsonl(GPT_JSONL)

    rows = []
    for vid in sorted(val):
        gt_v = val[vid]["gt_verdict"]
        gt_reason = val[vid]["gt_reasoning_en"]

        gem = gemini[vid]
        gem_correct = gem["verdict"] == gt_v
        gem_bert = round(score_one(gem["verdict"], gem["verdict_reasoning"], gt_v, gt_reason).alignment, 3)
        gem_score, gem_rat = GEMINI_TODAY_SCORES.get(vid, (None, ""))

        q = qwen[vid]
        q_correct = q["verdict"] == gt_v
        q_bert = round(score_one(q["verdict"], q["verdict_reasoning"], gt_v, gt_reason).alignment, 3)
        q_score, q_rat = QWEN_SCORES.get(vid, (None, ""))

        g = gpt[vid]
        g_correct = g["verdict"] == gt_v
        g_bert = round(score_one(g["verdict"], g["verdict_reasoning"], gt_v, gt_reason).alignment, 3)
        g_score, g_rat = GPT56LUNA_SCORES.get(vid, (None, ""))

        rows.append({
            "video_id": vid, "gt_verdict": gt_v, "gt_reasoning_en": gt_reason,
            "gemini_today__verdict": gem["verdict"], "gemini_today__correct": gem_correct,
            "gemini_today__reasoning": gem["verdict_reasoning"], "gemini_today__bert": gem_bert,
            "gemini_today__score": gem_score, "gemini_today__rationale": gem_rat,
            "qwen__verdict": q["verdict"], "qwen__correct": q_correct,
            "qwen__reasoning": q["verdict_reasoning"], "qwen__bert": q_bert,
            "qwen__score": q_score, "qwen__rationale": q_rat,
            "gpt56luna__verdict": g["verdict"], "gpt56luna__correct": g_correct,
            "gpt56luna__reasoning": g["verdict_reasoning"], "gpt56luna__bert": g_bert,
            "gpt56luna__score": g_score, "gpt56luna__rationale": g_rat,
        })

    df = pd.DataFrame(rows)

    def stats(name, ss, corrects):
        ss = [s for s in ss if s is not None]
        return {
            "teacher": name, "n": len(ss),
            "mean": round(sum(ss) / len(ss), 2) if ss else 0,
            "median": sorted(ss)[len(ss) // 2] if ss else 0,
            "n_ge8": sum(1 for s in ss if s >= 8),
            "n_le2": sum(1 for s in ss if s <= 2),
            "verdict_acc": f"{sum(corrects)}/{len(corrects)} ({sum(corrects)/len(corrects):.1%})",
        }

    df_summary = pd.DataFrame([
        {**stats("Gemini 3.1 Pro Preview (today, current teacher)", df["gemini_today__score"].tolist(), df["gemini_today__correct"].tolist()),
         "mean_bert": round(df["gemini_today__bert"].mean(), 3)},
        {**stats("Qwen3.7 Flash", df["qwen__score"].tolist(), df["qwen__correct"].tolist()),
         "mean_bert": round(df["qwen__bert"].mean(), 3)},
        {**stats("GPT-5.6 Luna Pro", df["gpt56luna__score"].tolist(), df["gpt56luna__correct"].tolist()),
         "mean_bert": round(df["gpt56luna__bert"].mean(), 3)},
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="per_clip", index=False)
        df_summary.to_excel(w, sheet_name="summary", index=False)

    print(f"Wrote {OUT_XLSX}: {len(df)} rows, {len(df.columns)} cols")
    print()
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()

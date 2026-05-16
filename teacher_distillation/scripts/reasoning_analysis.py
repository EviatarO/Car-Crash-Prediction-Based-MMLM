"""Generate reasoning_analysis.xlsx with per-clip qualitative scores.

Reads:
- outputs/prompt_bakeoff/results_balanced.jsonl (54 records)
- dataset/teacher_dataset_GT_self_imply.xlsx col G (verdict_reasoning_en)

Writes:
- outputs/prompt_bakeoff/reasoning_analysis.xlsx (per_clip + summary sheets)

Scores below were assigned qualitatively by Claude per the rubric in
reasoning_analysis.md.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
JL = REPO_ROOT / "outputs" / "prompt_bakeoff" / "results_balanced.jsonl"
GT = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"
OUT = REPO_ROOT / "outputs" / "prompt_bakeoff" / "reasoning_analysis.xlsx"

PROMPTS = [
    "PROMPT_G_OPT",
    "PROMPT_G_OPT_v5_balanced",
    "PROMPT_G_OPT_v6_balanced",
]

# (score, rationale) per (clip, prompt) — assigned qualitatively
SCORES = {
    "00077": {
        "PROMPT_G_OPT": (8, "Correct verdict/agent/mechanism; missed merge context"),
        "PROMPT_G_OPT_v5_balanced": (8, "Correct + gates referenced"),
        "PROMPT_G_OPT_v6_balanced": (9, "Notices brake lights like GT"),
    },
    "00147": {
        "PROMPT_G_OPT": (6, "Correct verdict; calls crossing not deviation"),
        "PROMPT_G_OPT_v5_balanced": (7, "Blame-flip; correct agent"),
        "PROMPT_G_OPT_v6_balanced": (7, "Blame-flip; correct agent + lateral conflict"),
    },
    "00283": {
        "PROMPT_G_OPT": (8, "Correct verdict/agent; says merge not left-turn"),
        "PROMPT_G_OPT_v5_balanced": (9, "Correct + final-frames timing"),
        "PROMPT_G_OPT_v6_balanced": (9, "Correct mechanism from right-turn lane"),
    },
    "00319": {
        "PROMPT_G_OPT": (7, "Correct + occlusion detail not in GT"),
        "PROMPT_G_OPT_v5_balanced": (7, "Correct + Gate B; mentions occlusion"),
        "PROMPT_G_OPT_v6_balanced": (8, "Cleanest mechanism matching GT"),
    },
    "00372": {
        "PROMPT_G_OPT": (7, "Correct + rear-end; color wrong (dark vs silver)"),
        "PROMPT_G_OPT_v5_balanced": (7, "Correct + Gate C; agent vague"),
        "PROMPT_G_OPT_v6_balanced": (3, "WRONG mechanism: hallucinated oncoming left-turn"),
    },
    "00474": {
        "PROMPT_G_OPT": (4, "Correct verdict but wrong cause (yellow taxi)"),
        "PROMPT_G_OPT_v5_balanced": (3, "Wrong verdict NO + wrong agent"),
        "PROMPT_G_OPT_v6_balanced": (9, "Correct verdict/agent/mechanism (white van)"),
    },
    "00493": {
        "PROMPT_G_OPT": (7, "Correct verdict; color wrong (white vs silver)"),
        "PROMPT_G_OPT_v5_balanced": (9, "Correct color silver + brake lights"),
        "PROMPT_G_OPT_v6_balanced": (7, "Correct verdict; color wrong"),
    },
    "00529": {
        "PROMPT_G_OPT": (2, "Wrong verdict; missed lateral drift"),
        "PROMPT_G_OPT_v5_balanced": (2, "Wrong verdict; missed lateral drift"),
        "PROMPT_G_OPT_v6_balanced": (2, "Wrong verdict; missed lateral drift"),
    },
    "00687": {
        "PROMPT_G_OPT": (7, "Correct verdict; agent partial (dark vs gray)"),
        "PROMPT_G_OPT_v5_balanced": (8, "Correct + correct color grey"),
        "PROMPT_G_OPT_v6_balanced": (8, "Correct + correct color grey"),
    },
    "01153": {
        "PROMPT_G_OPT": (1, "Hallucinated unprotected left turn"),
        "PROMPT_G_OPT_v5_balanced": (1, "Hallucinated crossing"),
        "PROMPT_G_OPT_v6_balanced": (1, "Hallucinated left turn"),
    },
    "01281": {
        "PROMPT_G_OPT": (1, "Hallucinated black SUV merge"),
        "PROMPT_G_OPT_v5_balanced": (1, "Hallucinated merge + Gate A/B"),
        "PROMPT_G_OPT_v6_balanced": (1, "Hallucinated SUV merge"),
    },
    "01504": {
        "PROMPT_G_OPT": (4, "Wrong verdict; missed ego braking"),
        "PROMPT_G_OPT_v5_balanced": (4, "Wrong verdict; right scene wrong outcome"),
        "PROMPT_G_OPT_v6_balanced": (4, "Wrong verdict; right scene wrong outcome"),
    },
    "01550": {
        "PROMPT_G_OPT": (8, "Correct verdict; gradual closing rate"),
        "PROMPT_G_OPT_v5_balanced": (9, "Steady/controlled paraphrase of GT"),
        "PROMPT_G_OPT_v6_balanced": (8, "Correct + safe distance"),
    },
    "01552": {
        "PROMPT_G_OPT": (6, "Correct; mentions truck only, misses SUV exit"),
        "PROMPT_G_OPT_v5_balanced": (8, "Mentions BOTH agents like GT"),
        "PROMPT_G_OPT_v6_balanced": (8, "Mentions both agents + clearing path"),
    },
    "01643": {
        "PROMPT_G_OPT": (9, "Correct; clean lane description"),
        "PROMPT_G_OPT_v5_balanced": (9, "Correct; all gates evaluated"),
        "PROMPT_G_OPT_v6_balanced": (8, "Correct; extra parked-cars detail"),
    },
    "01737": {
        "PROMPT_G_OPT": (9, "Correct; clean curved-lane match"),
        "PROMPT_G_OPT_v5_balanced": (0, "CATASTROPHIC: hallucinated pedestrian"),
        "PROMPT_G_OPT_v6_balanced": (0, "CATASTROPHIC: hallucinated pedestrian"),
    },
    "02104": {
        "PROMPT_G_OPT": (3, "Wrong verdict; missed controlled approach"),
        "PROMPT_G_OPT_v5_balanced": (8, "Only one to notice ego braking"),
        "PROMPT_G_OPT_v6_balanced": (3, "Wrong verdict; missed controlled approach"),
    },
    "02117": {
        "PROMPT_G_OPT": (1, "Hallucinated dark SUV merge"),
        "PROMPT_G_OPT_v5_balanced": (1, "Hallucinated red SUV cut-in + Gate B"),
        "PROMPT_G_OPT_v6_balanced": (8, "Only one to correctly see stationary right vehicle"),
    },
}


def _norm_vid(v) -> str:
    s = str(v).strip()
    return f"{int(s):05d}" if s.isdigit() else s


def main() -> None:
    recs = [json.loads(l) for l in JL.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_clip: dict = defaultdict(dict)
    for r in recs:
        by_clip[r["video_id"]][r["prompt_name"]] = r

    gt_df = pd.read_excel(GT)
    gt_map = {
        _norm_vid(row[gt_df.columns[0]]): {
            "target": row[gt_df.columns[1]],
            "reasoning": str(row[gt_df.columns[6]]).strip(),
        }
        for _, row in gt_df.iterrows()
    }

    rows = []
    for vid in sorted(by_clip):
        gt = gt_map.get(vid, {"target": None, "reasoning": ""})
        gt_v = "YES" if gt["target"] == 1 else "NO" if gt["target"] == 0 else "?"
        row = {
            "video_id": vid,
            "gt_verdict": gt_v,
            "gt_reasoning_en": gt["reasoning"],
        }
        per = SCORES.get(vid, {})
        for p in PROMPTS:
            r = by_clip[vid].get(p, {})
            score, why = per.get(p, (None, ""))
            row[f"{p}__verdict"] = r.get("verdict")
            row[f"{p}__correct"] = r.get("verdict") == gt_v
            row[f"{p}__bert"] = round(r.get("scores", {}).get("alignment", 0.0) or 0.0, 3)
            row[f"{p}__score"] = score
            row[f"{p}__rationale"] = why
            row[f"{p}__reasoning"] = (r.get("reasoning") or "").strip()

        s_opt = per.get("PROMPT_G_OPT", (0,))[0] or 0
        s_v5 = per.get("PROMPT_G_OPT_v5_balanced", (0,))[0] or 0
        s_v6 = per.get("PROMPT_G_OPT_v6_balanced", (0,))[0] or 0
        best = max(s_opt, s_v5, s_v6)
        winners = []
        if s_opt == best: winners.append("PROMPT_G_OPT")
        if s_v5 == best: winners.append("v5")
        if s_v6 == best: winners.append("v6")
        row["winner"] = "+".join(winners)
        row["best_score"] = best
        rows.append(row)

    df_out = pd.DataFrame(rows)

    summary_rows = []
    for p in PROMPTS:
        ss = [SCORES[v][p][0] for v in SCORES]
        sole_wins = 0
        for v in SCORES:
            mine = SCORES[v][p][0]
            others = [SCORES[v][q][0] for q in PROMPTS if q != p]
            if mine > max(others):
                sole_wins += 1
        summary_rows.append({
            "prompt": p,
            "mean_score": round(sum(ss) / len(ss), 2),
            "median_score": sorted(ss)[len(ss) // 2],
            "n_ge8": sum(1 for s in ss if s >= 8),
            "n_le2": sum(1 for s in ss if s <= 2),
            "sole_wins": sole_wins,
        })
    df_summary = pd.DataFrame(summary_rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        df_out.to_excel(w, sheet_name="per_clip", index=False)
        df_summary.to_excel(w, sheet_name="summary", index=False)

    print(f"Wrote {OUT}")
    print(f"  per_clip rows: {len(df_out)}, cols: {len(df_out.columns)}")
    print(f"  summary rows: {len(df_summary)}")
    print()
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()

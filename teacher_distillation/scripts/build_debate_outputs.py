"""Build results_v6_debate.xlsx and leaderboard_v6_debate.md from:
- outputs/prompt_bakeoff/highres_test.jsonl (6 prior hi-res clips)
- outputs/prompt_bakeoff/v6_hires_full18.jsonl (12 new hi-res clips)
- outputs/prompt_bakeoff/v6_debate.jsonl (recovery records for failing clips)

Does NOT delete or overwrite any existing output file. All deliverables use new
filenames suffixed with `_v6_debate`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "prompt_bakeoff"
HIRES_1 = OUT_DIR / "highres_test.jsonl"
HIRES_2 = OUT_DIR / "v6_hires_full18.jsonl"
DEBATE = OUT_DIR / "v6_debate.jsonl"

OUT_XLSX = OUT_DIR / "results_v6_debate.xlsx"
OUT_MD = OUT_DIR / "leaderboard_v6_debate.md"

GT_XLSX = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"


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


def _norm_vid(v) -> str:
    s = str(v).strip()
    return f"{int(s):05d}" if s.isdigit() else s


def main() -> None:
    # Merge the two hi-res JSONLs (key=video_id; v6_hires_full18 wins on conflict)
    hires: Dict[str, Dict] = {}
    for path in (HIRES_1, HIRES_2):
        for rec in _load_jsonl(path):
            vid = rec.get("video_id")
            if vid is None or rec.get("verdict") is None:
                continue
            hires[vid] = rec

    debate: Dict[str, Dict] = {r["video_id"]: r for r in _load_jsonl(DEBATE)}

    # Pull GT reasoning text
    gt_df = pd.read_excel(GT_XLSX)
    gt_map = {
        _norm_vid(row[gt_df.columns[0]]): {
            "target": row[gt_df.columns[1]],
            "reasoning": str(row[gt_df.columns[6]]).strip(),
        }
        for _, row in gt_df.iterrows()
    }

    rows = []
    for vid in sorted(hires):
        h = hires[vid]
        gt_v = h["gt_verdict"]
        gt_reason = gt_map.get(vid, {}).get("reasoning", "")
        v6_verdict = h["verdict"]
        v6_correct = (v6_verdict == gt_v)
        v6_bert = round(h.get("scores", {}).get("alignment", 0.0) or 0.0, 3)
        v6_reason = (h.get("reasoning") or "").strip()

        d = debate.get(vid)
        if d is None:
            recovery_prompt = ""
            recovery_verdict = ""
            recovery_correct = ""
            recovery_bert = ""
            recovery_reason = ""
            final = v6_verdict
            flipped = ""
        else:
            recovery_prompt = d.get("recovery_prompt", "")
            recovery_verdict = d.get("recovery_verdict") or ""
            recovery_correct = (recovery_verdict == gt_v)
            recovery_bert = round(d.get("scores", {}).get("alignment", 0.0) or 0.0, 3)
            recovery_reason = (d.get("recovery_reasoning") or "").strip()
            final = recovery_verdict or v6_verdict
            if not v6_correct and recovery_verdict == gt_v:
                flipped = "FIXED"
            elif v6_correct and recovery_verdict != gt_v:
                flipped = "BROKE"
            else:
                flipped = "still-wrong"

        rows.append({
            "video_id": vid,
            "gt_verdict": gt_v,
            "gt_reasoning_en": gt_reason,
            "v6_hires__verdict": v6_verdict,
            "v6_hires__correct": v6_correct,
            "v6_hires__bert": v6_bert,
            "v6_hires__reasoning": v6_reason,
            "recovery_prompt": recovery_prompt,
            "recovery__verdict": recovery_verdict,
            "recovery__correct": recovery_correct,
            "recovery__bert": recovery_bert,
            "recovery__reasoning": recovery_reason,
            "final_after_debate": final,
            "flipped_by_debate": flipped,
        })

    df = pd.DataFrame(rows)

    # Build summary metrics
    n = len(df)
    v6_acc = df["v6_hires__correct"].sum() / n if n else 0.0

    # After-debate accuracy: final_after_debate vs gt_verdict
    final_correct = (df["final_after_debate"] == df["gt_verdict"]).sum()
    final_acc = final_correct / n if n else 0.0

    # Confusion matrix - v6-only
    def confusion(pred_col: str) -> Dict[str, int]:
        tp = ((df[pred_col] == "YES") & (df["gt_verdict"] == "YES")).sum()
        fp = ((df[pred_col] == "YES") & (df["gt_verdict"] == "NO")).sum()
        tn = ((df[pred_col] == "NO") & (df["gt_verdict"] == "NO")).sum()
        fn = ((df[pred_col] == "NO") & (df["gt_verdict"] == "YES")).sum()
        return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}

    cm_v6 = confusion("v6_hires__verdict")
    cm_final = confusion("final_after_debate")

    n_debated = (df["recovery_prompt"] != "").sum()
    n_fixed = (df["flipped_by_debate"] == "FIXED").sum()
    n_broke = (df["flipped_by_debate"] == "BROKE").sum()
    n_still_wrong = (df["flipped_by_debate"] == "still-wrong").sum()

    fn_recovered = ((df["flipped_by_debate"] == "FIXED")
                    & (df["recovery_prompt"] == "PROMPT_G_OPT_v6_TP_RECOVERY")).sum()
    fp_recovered = ((df["flipped_by_debate"] == "FIXED")
                    & (df["recovery_prompt"] == "PROMPT_G_OPT_v6_TN_RECOVERY")).sum()

    summary_rows = [
        {"metric": "n_clips", "value": n},
        {"metric": "v6@hires accuracy", "value": f"{v6_acc:.1%}  ({df['v6_hires__correct'].sum()}/{n})"},
        {"metric": "after-debate accuracy", "value": f"{final_acc:.1%}  ({final_correct}/{n})"},
        {"metric": "v6@hires confusion (TP/FP/TN/FN)",
         "value": f"{cm_v6['TP']}/{cm_v6['FP']}/{cm_v6['TN']}/{cm_v6['FN']}"},
        {"metric": "after-debate confusion (TP/FP/TN/FN)",
         "value": f"{cm_final['TP']}/{cm_final['FP']}/{cm_final['TN']}/{cm_final['FN']}"},
        {"metric": "clips debated", "value": int(n_debated)},
        {"metric": "debate FIXED (wrong->right)", "value": int(n_fixed)},
        {"metric": "debate BROKE (right->wrong)", "value": int(n_broke)},
        {"metric": "debate still-wrong", "value": int(n_still_wrong)},
        {"metric": "FN recovered by TP_RECOVERY", "value": int(fn_recovered)},
        {"metric": "FP recovered by TN_RECOVERY", "value": int(fp_recovered)},
    ]
    df_summary = pd.DataFrame(summary_rows)

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="per_clip", index=False)
        df_summary.to_excel(w, sheet_name="summary", index=False)

    print(f"Wrote {OUT_XLSX.name}: {len(df)} rows, {len(df.columns)} cols")
    print()
    print(df_summary.to_string(index=False))

    # Build leaderboard.md
    md_lines = []
    md_lines.append("# Leaderboard — PROMPT_G_OPT_v6_balanced @ Hi-Res + Debate Recovery\n")
    md_lines.append("**Setup:** 18 GT clips, frames @ NATIVE 1280×720, Gemini `detail=\"high\"`,")
    md_lines.append("`temperature=0.1`, 16-frame window, stride=4.\n")
    md_lines.append("**Debate:** Every clip where v6 disagreed with GT was retested with a")
    md_lines.append("targeted recovery prompt (TP_RECOVERY for FN cases, TN_RECOVERY for FP cases).\n")
    md_lines.append("")

    md_lines.append("## Headline metrics\n")
    md_lines.append(f"| Stage | Accuracy | TP | FP | TN | FN |")
    md_lines.append(f"|-------|----------|----|----|----|----|")
    md_lines.append(f"| v6@hires (single-pass) | **{v6_acc:.1%}** ({df['v6_hires__correct'].sum()}/{n}) "
                    f"| {cm_v6['TP']} | {cm_v6['FP']} | {cm_v6['TN']} | {cm_v6['FN']} |")
    md_lines.append(f"| after debate | **{final_acc:.1%}** ({final_correct}/{n}) "
                    f"| {cm_final['TP']} | {cm_final['FP']} | {cm_final['TN']} | {cm_final['FN']} |")
    md_lines.append("")

    md_lines.append("## Debate yield\n")
    md_lines.append(f"- Clips debated: **{int(n_debated)}**")
    md_lines.append(f"- FIXED (wrong→right by recovery prompt): **{int(n_fixed)}**")
    md_lines.append(f"  - FN recovered by `PROMPT_G_OPT_v6_TP_RECOVERY`: {int(fn_recovered)}")
    md_lines.append(f"  - FP recovered by `PROMPT_G_OPT_v6_TN_RECOVERY`: {int(fp_recovered)}")
    md_lines.append(f"- BROKE (right→wrong by recovery prompt): **{int(n_broke)}**  ← debate regression risk")
    md_lines.append(f"- Still-wrong after debate: **{int(n_still_wrong)}**")
    md_lines.append("")

    md_lines.append("## Per-clip table\n")
    md_lines.append("| Clip | GT | v6@hires | recovery | final | flip |")
    md_lines.append("|------|----|----------|----------|-------|------|")
    for _, row in df.iterrows():
        rv = row["recovery__verdict"] or "—"
        rp = row["recovery_prompt"] or ""
        rp_short = (rp.replace("PROMPT_G_OPT_v6_", "").replace("_RECOVERY", "")
                    if rp else "")
        rec_cell = f"{rv} ({rp_short})" if rp else "—"
        flip = row["flipped_by_debate"] or "—"
        v6_mark = "✓" if row["v6_hires__correct"] else "✗"
        final_mark = "✓" if row["final_after_debate"] == row["gt_verdict"] else "✗"
        md_lines.append(
            f"| {row['video_id']} | {row['gt_verdict']} "
            f"| {row['v6_hires__verdict']} {v6_mark} "
            f"| {rec_cell} "
            f"| {row['final_after_debate']} {final_mark} "
            f"| {flip} |"
        )
    md_lines.append("")

    # Per-prompt summary
    md_lines.append("## Files\n")
    md_lines.append(f"- Per-clip data: `outputs/prompt_bakeoff/{OUT_XLSX.name}`")
    md_lines.append(f"- This leaderboard: `outputs/prompt_bakeoff/{OUT_MD.name}`")
    md_lines.append(f"- Source records: `highres_test.jsonl` + `v6_hires_full18.jsonl` + `v6_debate.jsonl`")
    md_lines.append("")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print()
    print(f"Wrote {OUT_MD.name} ({len(md_lines)} lines)")


if __name__ == "__main__":
    main()

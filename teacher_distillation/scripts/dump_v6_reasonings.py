"""Print every v6@hires reasoning and every recovery reasoning alongside GT,
so they can be qualitatively scored."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "outputs" / "prompt_bakeoff"
GT_XLSX = REPO_ROOT / "dataset" / "teacher_dataset_GT_self_imply.xlsx"


def _load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _norm_vid(v):
    s = str(v).strip()
    return f"{int(s):05d}" if s.isdigit() else s


def main():
    hires = {}
    for path in (OUT / "highres_test.jsonl", OUT / "v6_hires_full18.jsonl"):
        for rec in _load(path):
            if rec.get("verdict") is not None:
                hires[rec["video_id"]] = rec
    debate = {r["video_id"]: r for r in _load(OUT / "v6_debate.jsonl")}

    gt_df = pd.read_excel(GT_XLSX)
    gt_map = {
        _norm_vid(row[gt_df.columns[0]]): str(row[gt_df.columns[6]]).strip()
        for _, row in gt_df.iterrows()
    }

    out_lines = []
    for vid in sorted(hires):
        h = hires[vid]
        out_lines.append("=" * 80)
        out_lines.append(f"CLIP {vid}  GT={h['gt_verdict']}  v6={h['verdict']}  "
                         f"{'OK' if h['verdict']==h['gt_verdict'] else 'WRONG'}")
        out_lines.append("-" * 80)
        out_lines.append(f"GT reasoning:\n{gt_map.get(vid,'')}\n")
        out_lines.append(f"v6@hires reasoning:\n{(h.get('reasoning') or '').strip()}\n")
        d = debate.get(vid)
        if d:
            out_lines.append(f"recovery ({d['recovery_prompt']}) verdict={d['recovery_verdict']}:")
            out_lines.append(f"{(d.get('recovery_reasoning') or '').strip()}\n")
    out_path = OUT / "_v6_debate_reasoning_dump.txt"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()

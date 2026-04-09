"""Quick summary of all experiment runs for review."""
import json
import pandas as pd

versions = [
    ("v2 (PROMPT_G)", "outputs/teacher_dataset_v2.jsonl"),
    ("v3 (PROMPT_H)", "outputs/teacher_dataset_v3.jsonl"),
    ("v3.1 (H patched)", "outputs/teacher_dataset_v3_1.jsonl"),
]

dfs = {}
for label, path in versions:
    try:
        df = pd.read_json(path, lines=True)
        df["video_id"] = df["video_id"].astype(str).str.zfill(5)
        dfs[label] = df
        tp = ((df.target == 1) & (df.collision_verdict == "YES")).sum()
        fn = ((df.target == 1) & (df.collision_verdict != "YES")).sum()
        tn = ((df.target == 0) & (df.collision_verdict == "NO")).sum()
        fp = ((df.target == 0) & (df.collision_verdict != "NO")).sum()
        acc = (tp + tn) / len(df)
        print(f"{label:30s}  TP={tp} FP={fp} TN={tn} FN={fn}  Acc={acc:.1%}")
    except Exception as e:
        print(f"{label}: ERROR {e}")

print()
print("=" * 80)
print("CLIP-LEVEL DETAIL")
print("=" * 80)

gt_map = {}
pred_maps = {}
for label, df in dfs.items():
    pred_maps[label] = {}
    for _, r in df.iterrows():
        gt_map[r.video_id] = "YES" if r.target == 1 else "NO"
        pred_maps[label][r.video_id] = str(r.collision_verdict) if r.collision_verdict else "null"

all_ids = sorted(gt_map.keys())
header = f"{'Video':>7} {'GT':>4}"
for label in dfs:
    short = label.split("(")[0].strip()
    header += f" {short:>8}"
header += "  Notes"
print(header)
print("-" * 80)

for vid in all_ids:
    gt = gt_map[vid]
    line = f"{vid:>7} {gt:>4}"
    results = []
    for label in dfs:
        pred = pred_maps[label].get(vid, "-")
        ok = "ok" if pred == gt else "X"
        line += f" {pred:>5}({ok})"
        results.append(pred == gt)

    if all(results):
        note = "all correct"
    elif not any(results):
        note = "ALL WRONG"
    else:
        labels = list(dfs.keys())
        fixed = [labels[i].split("(")[0].strip() for i in range(len(results)) if results[i] and (i == 0 or not results[i - 1])]
        regressed = [labels[i].split("(")[0].strip() for i in range(len(results)) if not results[i] and (i > 0 and results[i - 1])]
        parts = []
        if fixed:
            parts.append("correct in " + ",".join(fixed))
        if regressed:
            parts.append("regressed in " + ",".join(regressed))
        note = "; ".join(parts) if parts else "mixed"

    line += f"  {note}"
    print(line)

print()
print("PROMPT CHANGES ACROSS VERSIONS:")
print("-" * 80)
print("v2: PROMPT_G - 6-step CoT (no lane discipline, no ego trajectory)")
print("v3: PROMPT_H - 8-step CoT (added ego trajectory, lane discipline, three-phase temporal, collision target)")
print("v3.1: PROMPT_H patched - same 8 steps + embedded decision rules in verdict,")
print("       lane discipline NOTE about rear-end irrelevance, RAPID closing threshold in Step 4")
print()
print("KEY INSIGHT: Adding lane discipline (v3) crushed TP recall.")
print("Adding closing-speed priority rules recovered TP but created massive FP.")
print("Current v3.1 attempts to balance with 'RAPID closing' threshold.")

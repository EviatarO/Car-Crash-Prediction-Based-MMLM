"""
build_pilot_data.py -> pilot_data.js   (Experiments page, tab "Boxed-teacher pilot")

One row per pilot window of set A (all three runs R1 / R2 / R3 are run on the same 51 windows so they can be read side by side):
  grid / grid_large : 4x4 contact sheet of the 16 encoder frames with the red box (review/<frames_dir>.jpg made by
                      r1_attention_boxes.py; grid_large is rebuilt here at 1920 px from boxed/<frames_dir>/ for the enlarge view)
  video_id, tte (0.5 / 1.0 / 1.5; for normal windows = seconds before the fake event), label (1 crash / 0 normal)
  runs.R1|R2|R3 : verdict (yes/no), explanation, tags (class / position / motion / gap), box_ok, problems
plus a per-run summary (windows, failures, wall time, latency, cost, verdict accuracy).

    python build_pilot_data.py            (re-run after the runs change; never hand-edit pilot_data.js)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = ROOT / "outputs" / "teacher_pilot_2026-10"
REL = "../outputs/teacher_pilot_2026-10"                       # relative to website/experiments.html (served root = Thesis/)
RUNS = {"R1": "Standard · explanation → verdict", "R2": "Flex · explanation → verdict", "R3": "Standard · verdict → explanation"}
SET = "A"


def load_run(name):
    p = PILOT / "runs" / f"{name}.jsonl"
    return {r["frames_dir"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if p.exists() else {}


def large_grid(frames_dir, label):
    out = PILOT / "grid_large" / f"{frames_dir}.jpg"
    if out.exists():
        return True
    src = sorted((PILOT / "boxed" / frames_dir).glob("frame_*.jpg"))
    if len(src) != 16:
        return False
    out.parent.mkdir(exist_ok=True)
    ims = [cv2.resize(cv2.imread(str(p)), (480, 270)) for p in src]
    for i, im in enumerate(ims):
        cv2.putText(im, str(i + 1), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    rows = [np.hstack(ims[i:i + 4]) for i in range(0, 16, 4)]
    head = np.zeros((40, 1920, 3), np.uint8)
    cv2.putText(head, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out), np.vstack([head] + rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def tags(p):
    return " · ".join(str(p.get(k)) for k in ("agent_class", "position", "object_motion", "gap") if p.get(k))


def main():
    wins = [json.loads(l) for l in open(PILOT / "windows.jsonl", encoding="utf-8")]
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(PILOT / "boxes.jsonl", encoding="utf-8")} \
        if (PILOT / "boxes.jsonl").exists() else {}
    runs = {n: load_run(n) for n in RUNS}
    rows = []
    for w in sorted([w for w in wins if w["set"] == SET], key=lambda w: (w["label"] == 0, w["video_id"], -w["horizon"])):
        f = w["frames_dir"]
        b = boxes.get(f)
        lab = f"{f}  label {w['label']}  {'TTE' if w['label'] else 'mid -'}{w['horizon']}s"
        row = {"key": f, "video_id": w["video_id"], "tte": w["horizon"], "label": w["label"],
               "grid": f"{REL}/review/{f}.jpg" if (PILOT / "review" / f"{f}.jpg").exists() else None,
               "grid_large": f"{REL}/grid_large/{f}.jpg" if b and large_grid(f, lab) else None,
               "box_flag": b["flag"] if b else None, "a1_p": b["a1_p_collision"] if b else None, "runs": {}}
        for n, r in runs.items():
            x = r.get(f)
            p = x["parsed"] if x and x["parsed"] else None
            row["runs"][n] = None if not p else {
                "verdict": p.get("collision"), "explanation": p.get("explanation"), "tags": tags(p),
                "box_ok": p.get("box_ok"), "problems": x["problems"], "wall_s": x["wall_s"]}
        rows.append(row)
    summary = {}
    for n, r in runs.items():
        sp = PILOT / "runs" / f"{n}.summary.json"
        s = json.load(open(sp)) if sp.exists() else None
        got = [x for x in r.values() if x["parsed"]]
        acc = lambda sel: (sum((x["parsed"]["collision"] == "yes") == bool(x["label"]) for x in got if sel(x)),  # noqa: E731
                           sum(1 for x in got if sel(x)))
        summary[n] = {"setting": RUNS[n], "n": len(r), "parsed": len(got),
                      "acc_all": acc(lambda x: True), "acc_crash": acc(lambda x: x["label"] == 1), "acc_normal": acc(lambda x: x["label"] == 0),
                      "run": s}
    data = {"set": SET, "dataset_key": "pilot_boxed_2026-10", "runs": RUNS, "rows": rows, "summary": summary,
            "note": "Gemini 3.8 Flash via OpenRouter, native-resolution frames with the attention-chosen object boxed; blind to the label."}
    n_with = {n: sum(1 for r in rows if r["runs"][n]) for n in RUNS}
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys)), "row keys must be unique (the notes store is keyed by them)"
    (HERE / "pilot_data.js").write_text("window.PILOT_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"pilot_data.js: {len(rows)} rows, rows with a result per run: {n_with}, grids: {sum(bool(r['grid']) for r in rows)}")


if __name__ == "__main__":
    main()

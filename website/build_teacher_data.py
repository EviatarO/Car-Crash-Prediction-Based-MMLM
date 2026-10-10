"""
build_teacher_data.py -> teacher_data.js   (Dataset page, tab "Teacher reasoning dataset")

One row per window of the full teacher dataset (outputs/teacher_dataset_2026-10, 4,203 windows = 1,401 Nexar training videos x 3), same
structure as the pilot table of experiments.html (assets/pilot.js renders both):
  grid / grid_large : 4x4 contact sheet of the 16 frames with the red box (review/<fd>.jpg), 1920-px version for the enlarge view
  label (Nexar event label), pre_alert, label_vis (visible-hazard label = the student's verdict target), status, path (which prompt made
  the final answer: E3 kept / E6 / E6C / E7 / E7pre / flagged)
  runs: E3 (stage 1, blind), E6 (anchored; its chain answers carry the tag E6C), E7 (hindsight; E7-pre answers carry the tag E7pre),
        FINAL = the answer used in the dataset (a reference to E3 when E3 was kept)
Needs flow_final.jsonl (r1_teacher_flow.py) for the routing; without it every window shows its E3 answer.

    python build_teacher_data.py [--no-large-grids]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "student_training" / "scripts"))
import build_pilot_data as bp  # noqa: E402  (set_runs, run_entry, large_grid, calls_summary)
import r1_teacher_prompt_study_report as study  # noqa: E402

FULL = ROOT / "outputs" / "teacher_dataset_2026-10"
REL = "../outputs/teacher_dataset_2026-10"
RUNS = {"E3": "Stage 1 · E3 · blind, boxed frames · Flex (all windows)",
        "E6": "Stage 2 · E6 · anchored on a valid correct window of the same video (+ timing check); chain answers carry the tag ↻ E6C",
        "E7": "Stage 2 · E7 · hindsight (frames after the window + outcome); E7-pre answers (pre-alert windows) carry the tag ↻ E7pre",
        "FINAL": "FINAL answer of each window: E3 where it matched the visible-hazard label, otherwise the stage-2 answer (label-informed on those windows)"}
PATHS = [["E3", "E3 kept"], ["E6", "E6 anchored"], ["E6C", "E6 chain (E6C)"], ["E7", "E7 (no valid anchor)"], ["E7pre", "E7-pre (pre-alert)"],
         ["flagged", "flagged: hazard not visible / needs review"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-large-grids", action="store_true")
    args = ap.parse_args()
    wins = [json.loads(l) for l in open(FULL / "windows.jsonl", encoding="utf-8")]
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(FULL / "boxes.jsonl", encoding="utf-8")}
    sr = bp.set_runs(FULL)
    flow = sr["flow"]
    if not args.no_large_grids:
        def mk(w):
            lab = f"{w['frames_dir']}  label {w['label']}  {'TTE' if w['label'] else 'mid -'}{w['horizon']}s" + ("  PRE-ALERT" if w.get("pre_alert") else "")
            return bp.large_grid(w["frames_dir"], lab, FULL)
        with ThreadPoolExecutor(8) as ex:
            ok = list(ex.map(mk, wins))
        print(f"[grids] large grids: {sum(ok)} of {len(wins)}")
    rows, paths = [], collections.Counter()
    for w in sorted(wins, key=lambda w: (w["video_id"], -w["horizon"])):
        fd = w["frames_dir"]
        f = flow.get(fd) or {}
        fr = f.get("final_from", "E3")
        path = "flagged" if str(f.get("status", "ok")).startswith("flagged") else fr
        paths[path] += 1
        b = boxes.get(fd) or {}
        row = {"key": fd, "set": "full", "valid": w.get("valid", True), "video_id": w["video_id"], "tte": w["horizon"], "label": w["label"],
               "pre_alert": bool(w.get("pre_alert")), "label_vis": f.get("label_vis", w.get("label_vis", w["label"])),
               "label_vis_source": f.get("label_vis_source"), "status": f.get("status", "ok"), "path": path,
               "grid": f"{REL}/review/{fd}.jpg" if (FULL / "review" / f"{fd}.jpg").exists() else None,
               "grid_large": f"{REL}/grid_large/{fd}.jpg" if (FULL / "grid_large" / f"{fd}.jpg").exists() else None,
               "box_flag": b.get("flag"), "a1_p": b.get("a1_p_collision"), "runs": {}}
        row["runs"]["E3"] = bp.run_entry("E3", sr["E3"].get(fd))
        row["runs"]["E6"] = bp.run_entry("E6", sr["E6"].get(fd)) if fd in sr["E6"] and path in ("E6", "E6C") else None
        row["runs"]["E7"] = bp.run_entry("E7", sr["E7"].get(fd)) if fd in sr["E7"] and path in ("E7", "E7pre") else None
        row["runs"]["FINAL"] = {"ref": "E3"} if fr == "E3" and row["runs"]["E3"] else bp.run_entry("ROUTED", sr["ROUTED"].get(fd))
        rows.append(row)
    # summary vs the visible-hazard label
    summary = {}
    for n in RUNS:
        got = [(r, r["runs"][n] if not (r["runs"][n] or {}).get("ref") else r["runs"]["E3"]) for r in rows if r["runs"][n]]
        got = [(r, x) for r, x in got if x]
        pos = [(r, x) for r, x in got if r["label_vis"] == 1]
        neg = [(r, x) for r, x in got if r["label_vis"] == 0]
        tp = sum(x["verdict"] == "yes" for _, x in pos); tn = sum(x["verdict"] == "no" for _, x in neg)
        auc = study.rank_auc([x["risk"] for _, x in pos if x["risk"] is not None], [x["risk"] for _, x in neg if x["risk"] is not None]) if n == "E3" else None
        s = None
        sp = FULL / "runs" / f"{n}.summary.json"
        if n == "E3" and sp.exists():
            s = json.load(open(sp)); s["elapsed_total_s"] = None            # the summary file only times the last resume
        elif n in ("E6", "E7"):
            src = sr[n]
            s = bp.calls_summary([x for x in src.values() if x.get("parsed") and path_of(flow, x) in (("E6", "E6C") if n == "E6" else ("E7", "E7pre"))])
        summary[n] = {"setting": RUNS[n], "n": len(got), "parsed": len(got), "acc_all": [tp + tn, len(got)], "acc_crash": [tp, len(pos)], "acc_normal": [tn, len(neg)],
                      "tp_ci": study.wilson(tp, len(pos)), "tn_ci": study.wilson(tn, len(neg)), "risk_auc": round(auc, 3) if auc is not None else None, "run": s}
    n_all = len(rows)
    cnt = lambda k: paths.get(k, 0)  # noqa: E731
    pre = sum(r["pre_alert"] for r in rows)
    flagged_soft = sum(bool((r["runs"]["E3"] or {}).get("problems")) for r in rows)
    note = (f"<b>{n_all:,} windows</b> from {len({r['video_id'] for r in rows}):,} Nexar training videos (3 windows each; TTE 1.5 / 1.0 / 0.5 s before the collision, "
            f"normal videos before a fake event at the midpoint). Crash windows kept by the visibility rule: {sum(1 for r in rows if r['label'] == 1 and not r['pre_alert']):,} · "
            f"pre-alert crash windows (label 1 → 0): {pre} · normal windows: {sum(1 for r in rows if r['label'] == 0):,}. "
            f"Final answers by path: E3 kept {cnt('E3'):,} · E6 anchored {cnt('E6'):,} · E6 chain {cnt('E6C')} · E7 {cnt('E7')} · E7-pre {cnt('E7pre')} · flagged {cnt('flagged')}. "
            f"E3 answers with a soft flag (colour / time / length word): {flagged_soft}. The routing is the flowchart of the Experiments page → Teacher prompt study. "
            f"The GT column shows the Nexar label and the visible-hazard label the student is trained on (pre-alert windows read 1 → 0). "
            f"FINAL answers on windows E3 got wrong are label-informed by design: read them for whether the text matches the frames, not as accuracy.")
    data = {"set": "full", "dataset_key": "teacher_dataset_2026-10", "runs": RUNS, "rows": rows, "summary": summary, "default_on": ["E3", "E6", "E7", "FINAL"],
            "hv_key": "teacher_dataset_hitter_visible_2026-10", "hide_cols": ["set"], "fails_default": False, "path_opts": PATHS, "paired_note": "",
            "note_html": note, "routing_fig": "../reports/figures/teacher_prompt_routing_2026-10-10.png"}
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys)) == n_all
    (HERE / "teacher_data.js").write_text("window.TEACHER_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    size = (HERE / "teacher_data.js").stat().st_size / 1e6
    print(f"teacher_data.js: {n_all} rows, {size:.1f} MB, paths {dict(paths)}, grids {sum(bool(r['grid']) for r in rows)}, large {sum(bool(r['grid_large']) for r in rows)}")


def path_of(flow, x):
    f = flow.get(x.get("frames_dir")) or {}
    return f.get("final_from")


if __name__ == "__main__":
    main()

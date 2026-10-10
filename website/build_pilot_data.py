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

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "student_training" / "scripts"))
import r1_teacher_prompt_study_report as study  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = ROOT / "outputs" / "teacher_pilot_2026-10"
REL = "../outputs/teacher_pilot_2026-10"                       # relative to website/experiments.html (served root = Thesis/)
RUNS = {"E1": "Prompt study · E1 · V12 neutral, raw frames · Flex",
        "E2": "Prompt study · E2 · V12 neutral + box · Flex",
        "E3": "Prompt study · E3 · v6_balanced re-balanced + box · Flex (blind, stage 1)",
        "E5": "Anchored · E5 · correct sibling window (frames + E3 reasoning) as reference · Flex · 12 windows",
        "E6": "Anchored · E6 · E5 + timing check and hazard_visible · Flex · 12 windows + the 00997 chain (↻ E6C: TTE 1.0 anchored on E7, TTE 1.5 on the 1.0 answer)",
        "E7": "Hindsight · E7 · frames after the window + outcome in text · Flex · only where the flow uses it: 00997 TTE 0.5 (all 3 windows wrong)",
        "ROUTED": "Final answer by the flowchart: E3 where right; E6 for failures with a correct sibling; E7 then the E6 chain for 00997 (label-informed on 15 windows)",
        "E34": "Prompt study · E3 with the E4 debate answer on E3's mistakes (label-routed) · Flex",
        "E4": "Prompt study · E4 · debate prompts, only on the 15 windows E3 got wrong · Flex",
        "R1": "Pilot · R1 · V12BOX fixed lists · Standard · explanation → verdict",
        "R2": "Pilot · R2 · V12BOX fixed lists · Flex · explanation → verdict",
        "R3": "Pilot · R3 · V12BOX fixed lists · Standard · verdict → explanation"}
DEFAULT_ON = ["E3", "E5", "E6", "E7"]
HV_KEY = "pilot_hitter_visible_2026-10"
SET = "A"


def load_run(name, multi=None):
    """Only answers that are part of the routing flowchart are shown: E6 includes the 00997 chain (tagged ↻ E6C); E7 is kept
    only where the flow uses it (start of the chain); the E7 comparison calls on the other 14 failures are not shown."""
    if name == "E6":
        out = dict(multi["runs"]["E6"])
        for fd, r in multi["runs"]["E6C"].items():
            if r.get("parsed") and not r.get("skipped"):
                out[fd] = dict(r, src_tag="E6C")
        return out
    if name == "E7":
        routed = multi["runs"]["ROUTED"]
        return {fd: r for fd, r in multi["runs"]["E7"].items() if (routed.get(fd) or {}).get("final_from") == "E7"}
    if multi is not None and name in ("E5", "ROUTED"):
        return multi["runs"][name]
    if name == "E34":
        return study.load_all()["E34"]
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


def tags(p, x=None):
    if "hazard_visible" in p or "same_agent" in p:                          # anchored / hindsight answers
        t = []
        if x and x.get("anchor_frames_dir"):
            t.append(f"anchor {x['anchor_frames_dir'].split('_hires_')[0]} TTE {x['anchor_horizon']:.1f} s")
        if "same_agent" in p:
            t.append("same agent: " + ("yes" if p["same_agent"] else "no"))
        if "hazard_visible" in p:
            ff = p.get("first_visible_frame")
            t.append(f"hazard visible: {p['hazard_visible']}" + (f" (from frame {ff})" if ff else ""))
        if p.get("involved_agent"):
            t.append(f"agent: {str(p['involved_agent'])[:40]}")
        return " · ".join(t)
    if "risk_score" in p:                                                     # prompt-study answers have no fixed-list fields
        return " · ".join(f"{k}: {str(p[k])[:60]}" for k in ("primary_agent", "gap_description") if p.get(k))
    return " · ".join(str(p.get(k)) for k in ("agent_class", "position", "object_motion", "gap") if p.get(k))


def calls_summary(recs):
    """Time / cost of exactly the calls shown (E6 + its chain, E7 at the chain start)."""
    if not recs:
        return None
    lat = sorted(x.get("wall_s") or 0 for x in recs)
    cst = [float((x.get("usage") or {}).get("cost") or 0) for x in recs]
    return {"elapsed_total_s": None, "latency_median_s": round(lat[len(lat) // 2], 1), "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 1),
            "cost_per_window_usd": sum(cst) / len(cst), "cost_total_usd": sum(cst), "with_problems": sum(bool(x["problems"]) for x in recs)}


def main():
    wins =[json.loads(l) for l in open(PILOT / "windows.jsonl", encoding="utf-8")]
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(PILOT / "boxes.jsonl", encoding="utf-8")} \
        if (PILOT / "boxes.jsonl").exists() else {}
    multi = study.compute_multi(study.compute())
    runs = {n: load_run(n, multi) for n in RUNS}
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
            src = ("E4" if (n == "E34" and x and x.get("from_e4")) else
                   x.get("final_from") if n == "ROUTED" and x and x.get("final_from") != "E3" else
                   x.get("src_tag") if x else None)
            row["runs"][n] = None if not p else {
                "verdict": p.get("collision"), "explanation": p.get("explanation"), "tags": tags(p, x),
                "box_ok": p.get("box_ok"), "problems": x["problems"], "wall_s": x.get("wall_s"),
                "risk": p.get("risk_score"), "from_e4": src, "prediction": p.get("prediction")}
        rows.append(row)
    summary = {}
    stats = study.compute()["stats"]
    for n, r in runs.items():
        sp = PILOT / "runs" / f"{n}.summary.json"
        s = json.load(open(sp)) if sp.exists() else None
        got = [x for x in r.values() if x["parsed"]]
        acc = lambda sel: (sum((x["parsed"]["collision"] == "yes") == bool(x["label"]) for x in got if sel(x)),  # noqa: E731
                           sum(1 for x in got if sel(x)))
        if n in ("E6", "E7"):                                                 # run files hold more calls than the flow shows
            s = calls_summary(got)
        summary[n] = {"setting": RUNS[n], "n": len(r), "parsed": len(got),
                      "acc_all": acc(lambda x: True), "acc_crash": acc(lambda x: x["label"] == 1), "acc_normal": acc(lambda x: x["label"] == 0),
                      "run": s}
        stats_all = dict(stats, **multi["multi"]["stats"])
        if n in ("E1", "E2", "E3", "E34", "ROUTED"):
            summary[n].update({k: stats_all[n][k] for k in ("tp_ci", "tn_ci", "risk_auc")})
        if n in ("E34", "ROUTED"):
            summary[n]["run"] = None
    data = {"set": SET, "dataset_key": "pilot_boxed_2026-10", "runs": RUNS, "rows": rows, "summary": summary, "default_on": DEFAULT_ON, "mcnemar": multi["mcnemar"], "hv_key": HV_KEY,
            "routing_fig": "../reports/figures/teacher_prompt_routing_2026-10-09.png",
            "note": "Gemini 3.8 Flash via OpenRouter, Flex, native-resolution frames. E3 is blind to the label; E5-E7 and the routed answers are label-informed by design (they are told the outcome through the correct sibling window or in text), so read them for text grounding, not as accuracy. E1 sees raw frames; E2-E6 see the attention-chosen object boxed in red."}
    n_with = {n: sum(1 for r in rows if r["runs"][n]) for n in RUNS}
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys)), "row keys must be unique (the notes store is keyed by them)"
    (HERE / "pilot_data.js").write_text("window.PILOT_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"pilot_data.js: {len(rows)} rows, rows with a result per run: {n_with}, grids: {sum(bool(r['grid']) for r in rows)}")


if __name__ == "__main__":
    main()

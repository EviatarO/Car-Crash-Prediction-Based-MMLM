"""
r1_teacher_flow.py - stage 2 of the teacher flow (plan 2026-10-10_Plan-Teacher-Full-Run.md).

Stage 1 (E3, blind, boxed frames) must already exist for every window (runs/E3.jsonl). This script then routes each window, using the
stage-1 answers of its own video, and writes the final answer per window to flow_final.jsonl.

Labels:  label      = Nexar event label (1 for every window of a crash video);
         pre_alert  = crash window the visibility rule removes (its end is before time_of_alert + 0.27 s; field `valid` = false);
         label_vis  = visible-hazard label used for routing and as the student's verdict target:
                      normal 0 · crash window kept by the rule 1 · pre-alert window 0, unless E7-pre finds the hazard already visible.
Routing (target = default label_vis):
  E3 verdict == target                                   -> keep E3
  pre-alert window, E3 said "yes"                        -> E7-pre (hindsight + "danger is annotated later"); label_vis = 1 if it
                                                            says hazard visible (yes / partly) and collision yes, else 0
  other wrong window, a valid correct sibling exists     -> E6 (anchor = nearest, larger TTE on a tie)
  other wrong window, no valid correct sibling           -> E7 on the video's TTE 0.5 window; if hazard visible and verdict right,
                                                            E6 chain 1.0 -> 1.5 on the remaining wrong valid windows, else flagged
Existing records in runs/E6.jsonl, E7.jsonl, E6C.jsonl are reused when their anchor matches (explanations stay byte-identical).

    python r1_teacher_flow.py --set A --model google/gemini-3.8-flash --tier flex [--dry-run]      # R1_PILOT_OUT picks the folder
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from prompts.PROMPT_TEACHER_STUDY import build_multi_prompt  # noqa: E402
from r1_after_frames import ensure as ensure_after  # noqa: E402
from r1_teacher_pilot import (AFTER, BOXED, OUT, anchor_dict, frames_of, load_run_file, make_client, run_jobs,  # noqa: E402
                              write_summary)

TRAIN = ROOT / "dataset" / "train"


def pre_alert(w):
    return bool(w["label"]) and not w.get("valid", True)


def lvis0(w):
    """Default visible-hazard label: normal 0, crash kept 1, pre-alert crash 0."""
    return 0 if (w["label"] == 0 or pre_alert(w)) else 1


def says_yes(rec):
    return bool(rec and rec.get("parsed")) and rec["parsed"].get("collision") == "yes"


def right(rec, target):
    return says_yes(rec) == bool(target) if rec and rec.get("parsed") else False


def reuse_or_run(jobs, name, args, client, boxes):
    """Run the jobs whose window has no record yet in runs/<name>.jsonl; return {frames_dir: record} for all jobs."""
    path = OUT / "runs" / f"{name}.jsonl"
    have = load_run_file(name)
    for j in jobs:
        r = have.get(j["w"]["frames_dir"])
        if r and r.get("parsed") and "anchor_frames_dir" in j["extra"]:
            assert r.get("anchor_frames_dir") == j["extra"]["anchor_frames_dir"], \
                f"{name} {j['w']['frames_dir']}: existing record has another anchor ({r.get('anchor_frames_dir')})"
    todo = [j for j in jobs if not (have.get(j["w"]["frames_dir"]) or {}).get("parsed")]
    if todo and not args.dry_run:
        run_jobs(todo, args, client, path, name, boxes)
        write_summary(name, list(load_run_file(name).values()), args, 0.0)
    elif todo:
        print(f"  [dry-run] {name}: would run {len(todo)} calls: {[j['w']['frames_dir'] for j in todo]}")
    have = load_run_file(name)
    return {j["w"]["frames_dir"]: have.get(j["w"]["frames_dir"]) for j in jobs}


def e7_job(w, boxes_unused, pre):
    n = json.load(open(AFTER / "index.json"))[w["frames_dir"]]["n_after"]
    clips = [("CLIP B -- target (16 frames, chronological)", frames_of(TRAIN / w["frames_dir"])),
             (f"CLIP C -- what happened next ({n} frames, chronological)", frames_of(AFTER / w["frames_dir"]))]
    return {"w": w, "variant": "e7", "prompt": build_multi_prompt("e7", n_after=n, label=w["label"], pre_alert=pre), "clips": clips,
            "extra": {"n_after": n, "pre_alert": pre}}


def e6_job(w, a, anchor_rec, variant="e6", extra=None):
    delta, earlier = abs(a["horizon"] - w["horizon"]), w["horizon"] > a["horizon"]
    prompt = build_multi_prompt(variant, delta=delta, earlier=earlier, anchor=anchor_dict(anchor_rec))
    clips = [("CLIP A -- reference (16 frames, chronological)", frames_of(BOXED / a["frames_dir"])),
             ("CLIP B -- target (16 frames, chronological)", frames_of(BOXED / w["frames_dir"]))]
    return {"w": w, "variant": variant, "prompt": prompt, "clips": clips,
            "extra": dict({"anchor_frames_dir": a["frames_dir"], "anchor_horizon": a["horizon"], "delta_s": delta}, **(extra or {}))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["A", "R", "ALL"])
    ap.add_argument("--stage1", default="E3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", choices=["standard", "flex"], default="flex")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    wins = [w for w in map(json.loads, open(OUT / "windows.jsonl", encoding="utf-8")) if args.set == "ALL" or w["set"] == args.set]
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(OUT / "boxes.jsonl", encoding="utf-8")}
    e3 = load_run_file(args.stage1)
    miss = [w["frames_dir"] for w in wins if not (e3.get(w["frames_dir"]) or {}).get("parsed")]
    assert not miss, f"stage 1 incomplete: {len(miss)} windows without an E3 answer, e.g. {miss[:3]}"
    client = None if args.dry_run else make_client()
    byv = collections.defaultdict(list)
    for w in wins:
        byv[w["video_id"]].append(w)
    tgt = {w["frames_dir"]: lvis0(w) for w in wins}
    fin = {}                                                              # frames_dir -> dict(final record + routing info)

    # ---- classify every window (all stage-1 answers are known)
    keep, pre_jobs, e6_plan, e7_start = [], [], [], []
    for v, vw in byv.items():
        valid_ok = [s for s in vw if not pre_alert(s) and right(e3[s["frames_dir"]], tgt[s["frames_dir"]])]
        for w in vw:
            fd = w["frames_dir"]
            if right(e3[fd], tgt[fd]):
                keep.append(w)
            elif pre_alert(w):                                            # E3 said "yes" on a pre-alert window
                pre_jobs.append(w)
            elif valid_ok:
                a = min(valid_ok, key=lambda s: (abs(s["horizon"] - w["horizon"]), -s["horizon"]))
                e6_plan.append((w, a))
            else:
                e7_start.append(w)
    chain_videos = sorted({w["video_id"] for w in e7_start})
    print(f"[flow] windows {len(wins)}: keep E3 {len(keep)} | E7-pre {len(pre_jobs)} | E6 anchored {len(e6_plan)} | "
          f"no valid anchor (E7 at TTE 0.5, then chain) {len(e7_start)} windows in videos {chain_videos}", flush=True)

    # ---- after-frames for every E7 / E7-pre window (the chain start is the video's TTE 0.5 window)
    start = {}
    for v in chain_videos:
        start[v] = next(s for s in byv[v] if s["horizon"] == 0.5)
    need = pre_jobs + list(start.values())
    if need:                                                              # local CPU only, no API calls
        ensure_after(need)

    # ---- E7-pre
    pre = reuse_or_run([e7_job(w, None, True) for w in pre_jobs], "E7PRE", args, client, boxes) if pre_jobs else {}
    # ---- E6 anchored (anchor answers are E3's own, correct and valid)
    e6 = reuse_or_run([e6_job(w, a, e3[a["frames_dir"]]) for w, a in e6_plan], "E6", args, client, boxes) if e6_plan else {}
    # ---- E7 start of the chain and the chain
    e7 = reuse_or_run([e7_job(start[v], None, False) for v in chain_videos], "E7", args, client, boxes) if chain_videos else {}
    chain, flagged = {}, {}
    for v in chain_videos:
        s = start[v]
        prev, pw = e7.get(s["frames_dir"]), s
        for w in sorted([x for x in byv[v] if x["frames_dir"] != s["frames_dir"] and x["frames_dir"] in {y["frames_dir"] for y in e7_start} | {z["frames_dir"] for z in []}
                         or (x["frames_dir"] != s["frames_dir"] and not pre_alert(x) and not right(e3[x["frames_dir"]], tgt[x["frames_dir"]]))],
                        key=lambda x: x["horizon"]):
            reason = None
            if not prev or not prev.get("parsed"):
                reason = "no parsed answer at the previous step"
            elif prev["parsed"].get("hazard_visible") == "no":
                reason = "hazard not visible at the previous step: chain stops"
            elif not right(prev, tgt[pw["frames_dir"]]):
                reason = "previous answer does not match the label: not a verified anchor"
            if reason:
                flagged[w["frames_dir"]] = reason
                continue
            res = reuse_or_run([e6_job(w, pw, prev, extra={"anchor_from": "E7" if pw is s else "E6C"})], "E6C", args, client, boxes)
            prev, pw = res[w["frames_dir"]], w
            chain[w["frames_dir"]] = prev
        if not prev or not (e7.get(s["frames_dir"]) or {}).get("parsed"):
            flagged.setdefault(s["frames_dir"], "no parsed E7 answer")
        elif e7[s["frames_dir"]]["parsed"].get("hazard_visible") == "no":
            flagged.setdefault(s["frames_dir"], "E7 says the hazard is not visible at TTE 0.5")

    # ---- final answer per window
    rows = []
    for w in wins:
        fd = w["frames_dir"]
        base = {"frames_dir": fd, "video_id": w["video_id"], "horizon": w["horizon"], "label": w["label"], "pre_alert": pre_alert(w),
                "label_vis": tgt[fd], "label_vis_source": "default", "status": "ok", "anchor": None}
        if right(e3[fd], tgt[fd]):
            src, rec = "E3", e3[fd]
        elif fd in pre:
            src, rec = "E7pre", pre[fd]
            p = (rec or {}).get("parsed") or {}
            if p and p.get("collision") == "yes" and p.get("hazard_visible") in ("yes", "partly"):
                base["label_vis"], base["label_vis_source"] = 1, "E7pre: hazard already visible"
            elif p and p.get("collision") == "yes":
                base["status"] = "check: E7-pre says yes but hazard not visible"
        elif fd in e6:
            src, rec = "E6", e6[fd]
        elif fd in chain:
            src, rec = "E6C", chain[fd]
        elif fd in {s["frames_dir"] for s in start.values()}:
            src, rec = "E7", e7[fd]
        else:
            src, rec = "E3", e3[fd]
            base["status"] = "flagged: " + flagged.get(fd, "not routed")
        if fd in flagged and base["status"] == "ok":
            base["status"] = "flagged: " + flagged[fd]
        if rec and rec.get("parsed") and src in ("E6", "E6C", "E7") and not right(rec, base["label_vis"]) and base["status"] == "ok":
            base["status"] = "check: verdict differs from label_vis"
        base.update({"final_from": src, "anchor": (rec or {}).get("anchor_frames_dir"), "parsed": (rec or {}).get("parsed"),
                     "problems": (rec or {}).get("problems", []), "usage": (rec or {}).get("usage")})
        rows.append(base)
    if not args.dry_run:
        with open(OUT / "flow_final.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter((r["final_from"], r["status"].split(":")[0]) for r in rows)
    print("[flow] final answers by source / status:", dict(c))
    print("[flow] label_vis changes (pre-alert -> 1):", [r["frames_dir"] for r in rows if r["label_vis"] != lvis0(next(w for w in wins if w["frames_dir"] == r["frames_dir"]))])
    print("[flow] flagged:", {r["frames_dir"]: r["status"] for r in rows if r["status"] != "ok"})


if __name__ == "__main__":
    main()

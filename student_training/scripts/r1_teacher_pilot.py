"""
r1_teacher_pilot.py - run the boxed-object teacher prompt (V12BOX) on the pilot windows, one run = one (set, tier, order).
Plan 2026-10-08_Plan-BoxedTeacher-Pilot-rev3. Reuses the image / message / JSON helpers of semsup_caption_promptbakeoff.py.

  python r1_teacher_pilot.py --name R1 --set A --tier standard --order explanation_first
  python r1_teacher_pilot.py --name R2 --set B --tier flex     --order explanation_first
  python r1_teacher_pilot.py --name R3 --set A --tier standard --order verdict_first
  python r1_teacher_pilot.py --flex-test        # ONE call on Flex: is the served tier really flex? (prints, writes nothing)

Per call it logs: wall time, tokens, cost (OpenRouter usage), served tier (response.service_tier if present), retries.
Output: outputs/teacher_pilot_2026-10/runs/<name>.jsonl (resumable on frames_dir) + <name>.summary.json.
Always pass --model explicitly (the repo default has been wrong before).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from prompts.PROMPT_SEMSUP_V12BOX import BANNED, ENUMS, build_prompt  # noqa: E402
from prompts.PROMPT_TEACHER_STUDY import BOXED as STUDY_BOXED, REQUIRED as STUDY_REQUIRED, VARIANTS as STUDY_VARIANTS  # noqa: E402
from prompts.PROMPT_TEACHER_STUDY import build_multi_prompt, build_prompt as build_study_prompt  # noqa: E402
from semsup_caption_promptbakeoff import _build_messages, _encode_image, _extract_json_object  # noqa: E402

OUT = Path(__import__("os").environ.get("R1_PILOT_OUT") or ROOT / "outputs" / "teacher_pilot_2026-10")   # R1_PILOT_OUT: separate test folders
BOXED = OUT / "boxed"
REQUIRED = ("agent_class", "position", "object_motion", "gap", "box_ok", "evidence_frames", "explanation", "collision")


def make_client():
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (.env)")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                  default_headers={"HTTP-Referer": "http://localhost", "X-Title": "MMLM_BoxedTeacherPilot"})


def call(client, model, messages, tier, timeout, retries, temperature):
    """One call with retries. Returns dict(text, usage, served_tier, wall_s, attempts, error)."""
    t0 = time.time()
    err = None
    for attempt in range(1, retries + 1):
        try:
            kw = {"model": model, "messages": messages, "temperature": temperature, "timeout": timeout}
            if tier == "flex":
                kw["extra_body"] = {"service_tier": "flex"}
            r = client.chat.completions.create(**kw)
            usage = r.usage.model_dump() if getattr(r, "usage", None) else {}
            served = getattr(r, "service_tier", None) or (getattr(r, "model_extra", None) or {}).get("service_tier")
            return {"text": (r.choices[0].message.content if r.choices else "") or "", "usage": usage, "served_tier": served,
                    "wall_s": time.time() - t0, "attempts": attempt, "error": None, "model_returned": getattr(r, "model", None)}
        except Exception as e:                                           # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 3 * 2 ** (attempt - 1)))
    return {"text": "", "usage": {}, "served_tier": None, "wall_s": time.time() - t0, "attempts": retries, "error": err}


def validate(p, order):
    """Soft validation: returns list of problems (empty = ok)."""
    if not isinstance(p, dict):
        return ["not a JSON object"]
    pr = [f"missing {k}" for k in REQUIRED if k not in p]
    for k, vals in ENUMS.items():
        if k in p and p[k] not in vals:
            pr.append(f"{k}={p[k]!r} not in list")
    if p.get("collision") not in ("yes", "no"):
        pr.append(f"collision={p.get('collision')!r}")
    ex = str(p.get("explanation", ""))
    n = len(ex.split())
    if not 25 <= n <= 40:
        pr.append(f"explanation words={n}")
    low = " " + re.sub(r"[^a-z/ -]", " ", ex.lower()) + " "
    for kind, words in BANNED.items():
        hit = [w for w in words if re.search(r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])", low)]
        if hit:
            pr.append(f"banned {kind}: {hit[:3]}")
    if "box" in low.split():
        pr.append("mentions the box")
    return pr


STUDY_BANNED = ("outcome", "colour", "time")


def validate_study(p, variant):
    """Soft validation of a prompt-study answer (PROMPT_TEACHER_STUDY): keys, risk 0-100, verdict vs risk, explanation checks."""
    if not isinstance(p, dict):
        return ["not a JSON object"]
    pr = [f"missing {k}" for k in STUDY_REQUIRED[variant] if k not in p]
    risk = p.get("risk_score")
    if not isinstance(risk, (int, float)) or isinstance(risk, bool) or not 0 <= risk <= 100:
        pr.append(f"risk_score={risk!r}")
    if p.get("collision") not in ("yes", "no"):
        pr.append(f"collision={p.get('collision')!r}")
    elif isinstance(risk, (int, float)) and (risk >= 50) != (p["collision"] == "yes"):
        pr.append(f"collision={p['collision']} but risk_score={risk}")
    ex = str(p.get("explanation", ""))
    n = len(ex.split())
    if not 25 <= n <= 45:
        pr.append(f"explanation words={n}")
    low = " " + re.sub(r"[^a-z/ -]", " ", ex.lower()) + " "
    for kind in STUDY_BANNED:
        hit = [w for w in BANNED[kind] if re.search(r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])", low)]
        if hit:
            pr.append(f"banned {kind}: {hit[:3]}")
    if "box" in low.split():
        pr.append("mentions the box")
    return pr


# ------------------------------------------------------------------ multi-clip runs: E5 / E6 (anchored), E7 (hindsight), chain
AFTER = OUT / "after"
CLIP_WORDS = re.compile(r"\bclip [abc]\b|\breference\b|\breviewer\b|\bhindsight\b", re.I)


def load_run_file(name):
    p = OUT / "runs" / f"{name}.jsonl"
    return {r["frames_dir"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if p.exists() else {}


def is_correct(r):
    return bool(r.get("parsed")) and (r["parsed"].get("collision") == "yes") == bool(r["label"])


def validate_multi(p, variant):
    """Soft validation of an E5/E6/E7 answer: keys, risk/verdict, explanation checks, prediction, hazard fields, no clip/reference words."""
    if not isinstance(p, dict):
        return ["not a JSON object"]
    pr = validate_study(p, variant)
    pred = p.get("prediction")
    if not isinstance(pred, str) or not 5 <= len(pred.split()) <= 45:
        pr.append(f"prediction words={len(str(pred).split()) if pred else 0}")
    if variant in ("e6", "e7"):
        if p.get("hazard_visible") not in ("yes", "partly", "no"):
            pr.append(f"hazard_visible={p.get('hazard_visible')!r}")
        f = p.get("first_visible_frame")
        if f is not None and not (isinstance(f, int) and not isinstance(f, bool) and 1 <= f <= 16):
            pr.append(f"first_visible_frame={f!r}")
    hit = sorted({m.group(0).lower() for k, v in p.items() if isinstance(v, str) for m in CLIP_WORDS.finditer(v)})
    if hit:
        pr.append(f"mentions {hit}")
    return pr


def multi_messages(prompt, clips):
    content = [{"type": "text", "text": prompt}]
    for header, paths in clips:
        content.append({"type": "text", "text": header})
        for pth in paths:
            content.append({"type": "image_url", "image_url": {"url": _encode_image(pth, 0), "detail": "high"}})
    return [{"role": "user", "content": content}]


def frames_of(d):
    paths = sorted(Path(d).glob("frame_*.jpg"))
    return paths


def anchor_dict(rec):
    """Reference material of a correct earlier answer (E3 / E6 / E7 record)."""
    p, lab = rec["parsed"], rec["label"]
    why = p.get("early_cues") or p.get("collision_interpretation" if lab else "safe_interpretation") or ""
    return {"label": lab, "temporal": p.get("temporal_analysis", ""), "why": why, "summary": p.get("explanation", ""),
            "risk": p.get("risk_score")}


def make_jobs(args, wins, e3):
    byfd = {w["frames_dir"]: w for w in wins}
    fails = [w for w in wins if w["frames_dir"] in e3 and not is_correct(e3[w["frames_dir"]])]
    jobs, noanchor = [], []
    # an anchor must be a CORRECT stage-1 answer on a VALID window (visibility rule); windows without "valid" (pilot) count as valid
    good = lambda s: is_correct(e3.get(s["frames_dir"], {})) and s.get("valid", True)  # noqa: E731
    for w in fails:
        fd = w["frames_dir"]
        if args.prompt == "e7" and args.flow:
            # flow only: the TTE-0.5 window of a video with no correct valid window (start of the chain)
            if w["horizon"] != 0.5 or any(good(s) for s in wins if s["video_id"] == w["video_id"]):
                continue
        if args.prompt == "e7":
            n = json.load(open(AFTER / "index.json"))[fd]["n_after"]
            clips = [("CLIP B -- target (16 frames, chronological)", frames_of(ROOT / "dataset" / "train" / fd)),
                     (f"CLIP C -- what happened next ({n} frames, chronological)", frames_of(AFTER / fd))]
            jobs.append({"w": w, "variant": "e7", "prompt": build_multi_prompt("e7", n_after=n, label=w["label"]), "clips": clips,
                         "extra": {"n_after": n}})
            continue
        sib = [s for s in wins if s["video_id"] == w["video_id"] and s["frames_dir"] != fd and good(s)]
        if not sib:
            noanchor.append(fd)
            continue
        a = min(sib, key=lambda s: (abs(s["horizon"] - w["horizon"]), -s["horizon"]))
        delta, earlier = abs(a["horizon"] - w["horizon"]), w["horizon"] > a["horizon"]
        prompt = build_multi_prompt(args.prompt, delta=delta, earlier=earlier, anchor=anchor_dict(e3[a["frames_dir"]]))
        clips = [("CLIP A -- reference (16 frames, chronological)", frames_of(BOXED / a["frames_dir"])),
                 ("CLIP B -- target (16 frames, chronological)", frames_of(BOXED / fd))]
        jobs.append({"w": w, "variant": args.prompt, "prompt": prompt, "clips": clips,
                     "extra": {"anchor_frames_dir": a["frames_dir"], "anchor_horizon": a["horizon"], "delta_s": delta}})
    return jobs, noanchor


def run_jobs(jobs, args, client, out_path, name, boxes, parallel=True):
    done = {json.loads(l)["frames_dir"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [j for j in jobs if j["w"]["frames_dir"] not in done]
    print(f"[{name}] {len(done)} already done, {len(todo)} to run", flush=True)

    def work(j):
        for _, paths in j["clips"]:
            assert paths, j["w"]["frames_dir"]
        return j, call(client, args.model, multi_messages(j["prompt"], j["clips"]), args.tier, args.timeout, args.retries, args.temperature)

    results = []
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency if parallel else 1) as pool:
        for k, fut in enumerate(as_completed([pool.submit(work, j) for j in todo]), 1):
            j, res = fut.result()
            w = j["w"]
            parsed = _extract_json_object(res["text"]) if res["text"] else None
            probs = ["call failed: " + str(res["error"])] if res["error"] is not None else validate_multi(parsed, j["variant"])
            rec = {"run": name, "set": w["set"], "frames_dir": w["frames_dir"], "video_id": w["video_id"], "label": w["label"],
                   "horizon": w["horizon"], "tier_requested": args.tier, "served_tier": res["served_tier"], "variant": j["variant"],
                   "wall_s": round(res["wall_s"], 2), "attempts": res["attempts"], "usage": res["usage"], "problems": probs,
                   "box_flag": boxes[w["frames_dir"]]["flag"], "track": boxes[w["frames_dir"]]["track"], "parsed": parsed,
                   "raw": res["text"] if parsed is None else None, **j["extra"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            results.append(rec)
            print(f"  [{k}/{len(todo)}] {w['frames_dir']} ok={parsed is not None} problems={len(probs)} elapsed={time.time() - t0:.0f}s", flush=True)
    cost = sum(float((r['usage'] or {}).get('cost') or 0) for r in results)
    print(f"[{name}] done: {len(results)} calls, cost ${cost:.3f}, elapsed {time.time() - t0:.0f}s", flush=True)
    return results


def write_summary(name, recs, args, t_total):
    lat = [r["wall_s"] for r in recs]
    cst = [float((r["usage"] or {}).get("cost") or 0) for r in recs]
    if not recs:
        return
    s = {"run": name, "tier": args.tier, "model": args.model, "n": len(recs), "failed": sum(r["parsed"] is None for r in recs),
         "elapsed_total_s": round(t_total, 1), "latency_median_s": round(st.median(lat), 1),
         "latency_p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1), "latency_max_s": round(max(lat), 1),
         "cost_total_usd": round(sum(cst), 4), "cost_per_window_usd": round(sum(cst) / len(cst), 5),
         "served_tiers": sorted({str(r["served_tier"]) for r in recs}), "with_problems": sum(bool(r["problems"]) for r in recs)}
    (OUT / "runs" / f"{name}.summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))


def main_multi(args):
    wins = [w for w in map(json.loads, open(OUT / "windows.jsonl", encoding="utf-8")) if w["set"] == (args.set or "A")]
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(OUT / "boxes.jsonl", encoding="utf-8")}
    e3 = load_run_file(args.stage1 or "E3")
    (OUT / "runs").mkdir(exist_ok=True)
    if args.chain:
        return main_chain(args, wins, boxes, e3)
    jobs, noanchor = make_jobs(args, wins, e3)
    name = args.name or args.prompt.upper()
    print(f"[cfg] run={name} prompt={args.prompt} tier={args.tier} model={args.model} windows={len(jobs)} no_anchor={noanchor}", flush=True)
    for j in jobs:
        print("  ", j["w"]["frames_dir"], {k: v for k, v in j["extra"].items()}, "images:", sum(len(p) for _, p in j["clips"]))
    if args.dry_run:
        print(jobs[0]["prompt"] if jobs else "(no jobs)")
        return
    t0 = time.time()
    run_jobs(jobs, args, make_client(), OUT / "runs" / f"{name}.jsonl", name, boxes)
    write_summary(name, list(load_run_file(name).values()), args, time.time() - t0)


def main_chain(args, wins, boxes, e3):
    """00997-style videos (no correct E3 window): E7 at TTE 0.5 -> E6 at 1.0 (anchor = E7) -> E6 at 1.5 (anchor = the 1.0 answer)."""
    e7 = load_run_file(args.chain_from or "E7")
    name = args.name or "E6C"
    out_path = OUT / "runs" / f"{name}.jsonl"
    client = make_client() if not args.dry_run else None
    vids = sorted({w["video_id"] for w in wins})
    t0 = time.time()
    plan = []
    for v in vids:
        vw = {w["horizon"]: w for w in wins if w["video_id"] == v}
        if any(is_correct(e3.get(w["frames_dir"], {})) and w.get("valid", True) for w in vw.values()) or \
                not all(w["frames_dir"] in e3 for w in vw.values()):
            continue                                                       # has a valid anchor in stage 1 -> not a chain video
        plan.append((v, vw))
    print(f"[chain] videos with no correct E3 window: {[v for v, _ in plan]}", flush=True)
    for v, vw in plan:
        prev = e7.get(vw[0.5]["frames_dir"])
        order = [(1.0, vw[1.0]), (1.5, vw[1.5])]
        anchor_w, anchor_rec = vw[0.5], prev
        for h, w in order:
            if is_correct(e3.get(w["frames_dir"], {})):
                print(f"[chain] {v} TTE {h}: E3 already correct -> kept, not re-run", flush=True)
                continue
            reason = None
            if not anchor_rec or not anchor_rec.get("parsed"):
                reason = "no parsed answer at the previous step"
            elif anchor_rec["parsed"].get("hazard_visible") == "no":
                reason = "previous step says the hazard is not visible: chain stops, windows marked"
            elif not is_correct(anchor_rec):
                reason = "previous answer does not match the label: not a verified anchor"
            if reason:
                print(f"[chain] {v} TTE {h}: SKIP ({reason})", flush=True)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"run": name, "frames_dir": w["frames_dir"], "video_id": v, "label": w["label"], "horizon": h,
                                        "skipped": reason, "parsed": None, "problems": [], "variant": "e6"}) + "\n")
                break
            delta, earlier = abs(anchor_w["horizon"] - h), h > anchor_w["horizon"]
            prompt = build_multi_prompt("e6", delta=delta, earlier=earlier, anchor=anchor_dict(anchor_rec))
            clips = [("CLIP A -- reference (16 frames, chronological)", frames_of(BOXED / anchor_w["frames_dir"])),
                     ("CLIP B -- target (16 frames, chronological)", frames_of(BOXED / w["frames_dir"]))]
            job = {"w": w, "variant": "e6", "prompt": prompt, "clips": clips,
                   "extra": {"anchor_frames_dir": anchor_w["frames_dir"], "anchor_horizon": anchor_w["horizon"], "delta_s": delta,
                             "anchor_from": "E7" if anchor_w["horizon"] == 0.5 else name}}
            if args.dry_run:
                print(f"[chain] {v} TTE {h}: anchor {anchor_w['frames_dir']}; prompt chars {len(prompt)}")
                break
            res = run_jobs([job], args, client, out_path, name, boxes, parallel=False)
            anchor_w, anchor_rec = w, (res[0] if res else load_run_file(name).get(w["frames_dir"]))
    if not args.dry_run:
        write_summary(name, [r for r in load_run_file(name).values() if not r.get("skipped")], args, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=list(STUDY_VARIANTS) + ["e5", "e6", "e7"], help="prompt-study variant (PROMPT_TEACHER_STUDY); omit for the V12BOX pilot prompt")
    ap.add_argument("--stage1", help="E5/E6/E7: the stage-1 run whose mistakes are re-run (default E3)")
    ap.add_argument("--chain", action="store_true", help="chain for videos with no correct stage-1 window: E7 -> E6 (1.0) -> E6 (1.5)")
    ap.add_argument("--chain-from", help="run holding the E7 answers (default E7)")
    ap.add_argument("--flow", action="store_true", help="E7: only where the flowchart uses it (TTE 0.5 of a video with no correct valid window)")
    ap.add_argument("--frames", choices=["raw", "boxed"], default="boxed")
    ap.add_argument("--debate-from", help="run name of Exp #3: re-run only its mistakes with e4_tp (crash missed) / e4_tn (normal flagged)")
    ap.add_argument("--name")
    ap.add_argument("--set", choices=["A", "B", "R"])
    ap.add_argument("--tier", choices=["standard", "flex"], default="standard")
    ap.add_argument("--order", choices=["explanation_first", "verdict_first"], default="explanation_first")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flex-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.prompt in ("e5", "e6", "e7") or args.chain:
        return main_multi(args)

    wins = [json.loads(l) for l in open(OUT / "windows.jsonl", encoding="utf-8")]
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(OUT / "boxes.jsonl", encoding="utf-8")}
    if args.flex_test:
        args.set, args.tier, args.name, args.limit = "A", "flex", "flex_test", 1
    rows = [w for w in wins if w["set"] == args.set and w["frames_dir"] in boxes][: args.limit or None]
    study = args.prompt is not None or args.debate_from is not None
    if args.debate_from:                                                   # Exp #4: route Exp #3's mistakes by the label
        prev = {r["frames_dir"]: r for r in map(json.loads, open(OUT / "runs" / f"{args.debate_from}.jsonl", encoding="utf-8"))}
        route = {}
        for w in rows:
            p = (prev.get(w["frames_dir"]) or {}).get("parsed") or {}
            if p.get("collision") in ("yes", "no") and (p["collision"] == "yes") != bool(w["label"]):
                route[w["frames_dir"]] = "e4_tp" if w["label"] else "e4_tn"
        rows = [w for w in rows if w["frames_dir"] in route]
        variant_of = lambda w: route[w["frames_dir"]]                      # noqa: E731
    else:
        variant_of = lambda w: args.prompt                                 # noqa: E731
    prompts = ({v: build_study_prompt(v) for v in {variant_of(w) for w in rows}} if study else {None: build_prompt(args.order)})
    if study and not args.debate_from:
        args.frames = "boxed" if STUDY_BOXED[args.prompt] else "raw"       # the variant fixes which frames it is run on
    elif args.debate_from:
        args.frames = "boxed"
    print(f"[cfg] run={args.name} set={args.set} tier={args.tier} order={args.order} model={args.model} windows={len(rows)} "
          f"concurrency={args.concurrency} prompt={args.prompt or ('debate<-' + args.debate_from if args.debate_from else 'V12BOX')} "
          f"frames={args.frames if study else 'boxed'}", flush=True)
    if args.dry_run:
        for v, pt in prompts.items():
            print(f"--- {v} ({len(pt)} chars)\n{pt[:600]}...")
        return
    client = make_client()
    runs = OUT / "runs"
    runs.mkdir(exist_ok=True)
    out_path = runs / f"{args.name}.jsonl"
    done = {json.loads(l)["frames_dir"] for l in open(out_path, encoding="utf-8")} if out_path.exists() and not args.flex_test else set()
    todo = [w for w in rows if w["frames_dir"] not in done]
    print(f"[cfg] {len(done)} already done, {len(todo)} to run", flush=True)

    def work(w):
        src = ROOT / "dataset" / "train" / w["frames_dir"] if (study and args.frames == "raw") else BOXED / w["frames_dir"]
        paths = sorted(src.glob("frame_*.jpg"))
        assert len(paths) == 16, w["frames_dir"]
        imgs = [_encode_image(p, 0) for p in paths]                       # native resolution (V12 used native)
        res = call(client, args.model, _build_messages(prompts[variant_of(w) if study else None], imgs, "high"), args.tier,
                   args.timeout, args.retries, args.temperature)
        return w, res

    t0 = time.time()
    n_ok = n_fail = 0
    cost = 0.0
    with open(out_path, "a" if not args.flex_test else "w", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as pool:
        futs = [pool.submit(work, w) for w in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            w, res = fut.result()
            parsed = _extract_json_object(res["text"]) if res["text"] else None
            if res["error"] is not None:
                probs = ["call failed: " + str(res["error"])]
            else:
                probs = validate_study(parsed, variant_of(w)) if study else validate(parsed, args.order)
            c = res["usage"].get("cost")
            cost += float(c) if c is not None else 0.0
            rec = {"run": args.name, "set": w["set"], "frames_dir": w["frames_dir"], "video_id": w["video_id"], "label": w["label"],
                   "horizon": w["horizon"], "tier_requested": args.tier, "served_tier": res["served_tier"], "order": args.order,
                   "wall_s": round(res["wall_s"], 2), "attempts": res["attempts"], "usage": res["usage"], "model_returned": res.get("model_returned"),
                   "variant": variant_of(w) if study else "v12box", "frames": args.frames if study else "boxed",
                   "box_flag": boxes[w["frames_dir"]]["flag"], "track": boxes[w["frames_dir"]]["track"], "problems": probs,
                   "parsed": parsed, "raw": res["text"] if parsed is None else None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_ok += res["error"] is None and parsed is not None
            n_fail += not (res["error"] is None and parsed is not None)
            if args.flex_test:
                print("[flex-test] served_tier =", res["served_tier"], "| wall_s =", round(res["wall_s"], 1), "| usage =",
                      res["usage"], "| error =", res["error"], "| model_returned =", res.get("model_returned"))
            elif k % 10 == 0 or k == len(todo):
                print(f"  [{k}/{len(todo)}] ok={n_ok} fail={n_fail} cost=${cost:.3f} elapsed={time.time() - t0:.0f}s", flush=True)
    if args.flex_test:
        return
    recs = [json.loads(l) for l in open(out_path, encoding="utf-8")]
    lat = [r["wall_s"] for r in recs]
    cst = [float(r["usage"].get("cost") or 0) for r in recs]
    summ = {"run": args.name, "tier": args.tier, "order": args.order, "model": args.model, "n": len(recs),
            "failed": sum(r["parsed"] is None for r in recs), "elapsed_total_s": round(time.time() - t0, 1),
            "latency_median_s": round(st.median(lat), 1), "latency_p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
            "latency_max_s": round(max(lat), 1), "cost_total_usd": round(sum(cst), 4), "cost_per_window_usd": round(sum(cst) / len(cst), 5),
            "prompt_tokens_median": st.median([r["usage"].get("prompt_tokens", 0) for r in recs]),
            "completion_tokens_median": st.median([r["usage"].get("completion_tokens", 0) for r in recs]),
            "served_tiers": sorted({str(r["served_tier"]) for r in recs}), "with_problems": sum(bool(r["problems"]) for r in recs)}
    (runs / f"{args.name}.summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

"""
r1_teacher_test.py - teacher reasoning for the Nexar PRIVATE test windows (plan 2026-10-17_Plan-Test-Private-Reasoning, steps 4, 5, 9).
Gemini 3.8 Flash on OpenRouter, Flex tier. Everything lives in outputs/teacher_test_2026-10 (R1_PILOT_OUT is set to it, so r1_add_prediction.py works on it).

  e3     step 4: the BLIND first look on every boxed test window: the same E3 prompt as the training run (PROMPT_TEACHER_STUDY, byte-identical), the 16 boxed
         frames at native resolution, temperature 0.1.  -> runs/E3.jsonl     then:  python r1_add_prediction.py --run E3 ...  (the outcome sentence, text only)
  e8     step 5 = pass 2, on the windows where the pass-1 verdict != the Nexar label (or the pass-1 answer is missing): PROMPT_TEACHER_TEST_P2 (anchored
         self-check, 2026-10-17; replaces the never-run PROMPT_TEACHER_TEST_E8): the same boxed frames + the pass-1 answer as a draft + the true outcome
         in text; fact_check changes a fact only with frame numbers; hazard_visible.  An answer that fails validate_p2 is re-asked (--reask, default 2)
         and the answer with the fewest problems is kept.      -> runs/E8.jsonl
         --pilot N : N mismatching windows, written to runs/E8_pilot.jsonl (the gate before the full pass): the --ids videos first, then random (seed 0)
  final  step 9: dataset_test_final.jsonl, one row per window: blind answer where it agrees with the label, else the second look (source, hazard_visible,
         problems), hand edits from edits_test.jsonl if present.

    python r1_teacher_test.py e3 --model google/gemini-3.8-flash --tier flex [--limit N] [--dry-run]
    python r1_teacher_test.py e8 --model google/gemini-3.8-flash --tier flex [--pilot 20 --ids 00352,00367] [--dry-run]
    python r1_teacher_test.py final
    python r1_teacher_test.py e3 --split public --concurrency 64     # stage 4 (2026-10-22): pass 1 on the PUBLIC windows -> outputs/teacher_test_public_2026-10/runs/
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# plan 2026-10-22 stage 4: --split public = the same pass on the PUBLIC test windows, in their own folder (windows.jsonl = windows_public.jsonl, boxed/ from
# r1_attention_boxes.py --stage boxes with R1_PILOT_OUT set to it); read before argparse because every path below hangs on it
SPLIT = "public" if "--split=public" in sys.argv or any(a == "--split" and sys.argv[i + 1:i + 2] == ["public"] for i, a in enumerate(sys.argv)) else "private"
OUT = ROOT / "outputs" / ("teacher_test_public_2026-10" if SPLIT == "public" else "teacher_test_2026-10")
os.environ.setdefault("R1_PILOT_OUT", str(OUT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from prompts.PROMPT_TEACHER_STUDY import build_prompt as build_study_prompt  # noqa: E402
from prompts.PROMPT_TEACHER_TEST_E8 import REQUIRED_E8  # noqa: E402
from prompts.PROMPT_TEACHER_TEST_P2 import FACTS, REQUIRED_P2, build_p2_prompt  # noqa: E402
from r1_teacher_pilot import STUDY_BANNED, BANNED, call, make_client, validate_study  # noqa: E402
from semsup_caption_promptbakeoff import _build_messages, _encode_image, _extract_json_object  # noqa: E402

BOXED = OUT / "boxed"
RUNS = OUT / "runs"
HINDSIGHT_WORDS = re.compile(r"\bhindsight\b|\breviewer\b|\bverified\b|\bstated outcome\b|\boutcome statement\b", re.I)


def load_windows():
    wins = [json.loads(l) for l in open(OUT / "windows.jsonl", encoding="utf-8")]
    return [w for w in wins if (BOXED / w["frames_dir"]).exists()]


def validate_e8(p):
    """Soft validation of a second-look answer."""
    if not isinstance(p, dict):
        return ["not a JSON object"]
    pr = [f"missing {k}" for k in REQUIRED_E8 if k not in p]
    risk = p.get("risk_score")
    if not isinstance(risk, (int, float)) or isinstance(risk, bool) or not 0 <= risk <= 100:
        pr.append(f"risk_score={risk!r}")
    if p.get("collision") not in ("yes", "no"):
        pr.append(f"collision={p.get('collision')!r}")
    elif isinstance(risk, (int, float)) and (risk >= 50) != (p["collision"] == "yes"):
        pr.append(f"collision={p['collision']} but risk_score={risk}")
    if p.get("hazard_visible") not in ("yes", "partly", "no"):
        pr.append(f"hazard_visible={p.get('hazard_visible')!r}")
    ex = str(p.get("explanation", ""))
    if not 25 <= len(ex.split()) <= 45:
        pr.append(f"explanation words={len(ex.split())}")
    low = " " + re.sub(r"[^a-z/ -]", " ", ex.lower()) + " "
    for kind in STUDY_BANNED:
        hit = [w for w in BANNED[kind] if re.search(r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])", low)]
        if hit:
            pr.append(f"banned {kind}: {hit[:3]}")
    if re.search(r"box(?!\s+(truck|trucks|van|vans))", low):
        pr.append("mentions the box")
    if HINDSIGHT_WORDS.search(ex) or HINDSIGHT_WORDS.search(str(p.get("prediction", ""))):
        pr.append("mentions the outcome statement")
    n = len(str(p.get("prediction", "")).split())
    if not 3 <= n <= 15:
        pr.append(f"prediction words={n}")
    return pr


FRAME_WORDS = re.compile(r"\bframes?\b|\bdraft\b|\bfirst (answer|look)\b", re.I)


def validate_p2(p, w):
    """Pass-2 validation: the E8 checks (format, banned words, box, outcome statement) with the P2 keys, + fact_check, no frame/draft words in the text,
    and a decision that disagrees with the label only when hazard_visible says the frames do not support the outcome."""
    if not isinstance(p, dict):
        return ["not a JSON object"]
    pr = [x for x in validate_e8(p) if not (x.startswith("missing ") and x[8:] not in REQUIRED_P2)]
    pr += [f"missing {k}" for k in REQUIRED_P2 if k not in p and f"missing {k}" not in pr]
    fc = p.get("fact_check")
    if not isinstance(fc, dict):
        pr.append("fact_check not an object")
    else:
        for k in FACTS:
            f = fc.get(k)
            if not isinstance(f, dict) or not isinstance(f.get("changed"), bool):
                pr.append(f"fact_check.{k} malformed")
            elif f["changed"]:
                nums = [int(n) for n in re.findall(r"\d+", str(f.get("frames", "")))]
                if not nums or not all(1 <= n <= 16 for n in nums):
                    pr.append(f"fact_check.{k} changed without valid frames ({f.get('frames')!r})")
    for fld in ("explanation", "prediction"):
        if FRAME_WORDS.search(str(p.get(fld, ""))):
            pr.append(f"{fld} mentions frames / the draft")
    if p.get("collision") in ("yes", "no") and (p["collision"] == "yes") != bool(w["label"]):
        honest = p.get("hazard_visible") == ("no" if w["label"] else "yes")          # crash with nothing visible / normal with a clear hazard
        if not honest:
            pr.append(f"verdict disagrees with the label but hazard_visible={p.get('hazard_visible')!r}")
    return pr


def run(args):
    wins = load_windows()
    RUNS.mkdir(exist_ok=True)
    if args.cmd == "e3":
        name, rows = "E3", wins
        prompt_of = lambda w: build_study_prompt("e3")                        # noqa: E731
        check = lambda p, w: validate_study(p, "e3")                          # noqa: E731
        variant = "e3"
    else:
        e3 = {r["frames_dir"]: r for r in map(json.loads, open(RUNS / "E3.jsonl", encoding="utf-8")) if r.get("parsed")}
        rows = [w for w in wins if (e3.get(w["frames_dir"]) or {}).get("parsed", {}).get("collision") not in ("yes", "no")
                or (e3[w["frames_dir"]]["parsed"]["collision"] == "yes") != bool(w["label"])]
        name = "E8_pilot" if args.pilot else "E8"
        if args.pilot:
            ids = [i.strip() for i in args.ids.split(",") if i.strip()]
            first = [w for w in rows if w["video_id"] in ids]
            print(f"[pilot] {len(first)} of {len(ids)} --ids are pass-1-wrong windows; not: {sorted(set(ids) - {w['video_id'] for w in first})}")
            rest = [w for w in rows if w["video_id"] not in ids]
            rows = first + random.Random(0).sample(rest, max(0, min(args.pilot - len(first), len(rest))))
        prompt_of = lambda w: build_p2_prompt(w["label"], w["horizon"], (e3.get(w["frames_dir"]) or {}).get("parsed"))   # noqa: E731
        check = validate_p2
        variant = "p2_anchored"
    if args.limit:
        rows = rows[:args.limit]
    out_path = RUNS / f"{name}.jsonl"
    done = {r["frames_dir"] for r in map(json.loads, open(out_path, encoding="utf-8")) if r.get("parsed")} if out_path.exists() else set()
    todo = [w for w in rows if w["frames_dir"] not in done]
    print(f"[cfg] {name} model={args.model} tier={args.tier} windows={len(rows)} ({len(done)} done, {len(todo)} to run) concurrency={args.concurrency}", flush=True)
    if args.dry_run:
        for w in todo[:1]:
            p = prompt_of(w)
            print(f"--- {variant} prompt for {w['frames_dir']} (label {w['label']}, TTE {w['horizon']}), {len(p)} chars\n...{p[-2600:]}")
        return
    client = make_client()

    def work(w):
        paths = sorted((BOXED / w["frames_dir"]).glob("frame_*.jpg"))
        assert len(paths) == 16, w["frames_dir"]
        imgs = [_encode_image(p, 0) for p in paths]                          # native resolution, as in training
        msgs = _build_messages(prompt_of(w), imgs, "high")
        best, asks, spent = None, 0, 0.0
        for i in range(1 + max(0, args.reask)):                            # re-ask an answer that fails validation (at temperature >= 0.4, so it can differ)
            res = call(client, args.model, msgs, args.tier, args.timeout, args.retries, args.temperature if i == 0 else max(args.temperature, 0.4))
            asks += 1
            c = res["usage"].get("cost")
            spent += float(c) if c is not None else 0.0
            parsed = _extract_json_object(res["text"]) if res["text"] else None
            probs = ["call failed: " + str(res["error"])] if res["error"] is not None else check(parsed, w)
            if best is None or (parsed is not None and (best[1] is None or len(probs) < len(best[2]))):
                best = (res, parsed, probs)
            if parsed is not None and not probs:
                break
        return w, best, asks, spent

    t0, n_ok, n_fail, cost = time.time(), 0, 0, 0.0
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as pool:
        for k, fut in enumerate(as_completed([pool.submit(work, w) for w in todo]), 1):
            w, (res, parsed, probs), asks, spent = fut.result()
            cost += spent
            rec = {"run": name, "set": w["set"], "frames_dir": w["frames_dir"], "video_id": w["video_id"], "label": w["label"], "horizon": w["horizon"],
                   "tier_requested": args.tier, "served_tier": res["served_tier"], "wall_s": round(res["wall_s"], 2), "attempts": res["attempts"],
                   "asks": asks, "cost_all_asks": round(spent, 5),
                   "usage": res["usage"], "model_returned": res.get("model_returned"), "variant": variant, "problems": probs, "parsed": parsed,
                   "raw": res["text"] if parsed is None else None}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            ok = res["error"] is None and parsed is not None
            n_ok += ok
            n_fail += not ok
            if k % 10 == 0 or k == len(todo):
                print(f"  [{k}/{len(todo)}] ok={n_ok} fail={n_fail} cost=${cost:.3f} elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"[done] {name}: ok {n_ok}, failed {n_fail} (rerun to retry them); cost ${cost:.3f}; {time.time() - t0:.0f}s")


def final(args):
    wins = [json.loads(l) for l in open(OUT / "windows.jsonl", encoding="utf-8")]
    e3 = {r["frames_dir"]: r for r in map(json.loads, open(RUNS / "E3.jsonl", encoding="utf-8")) if r.get("parsed")}
    p2 = {r["frames_dir"]: r for r in map(json.loads, open(RUNS / "E3P2.jsonl", encoding="utf-8")) if r.get("prediction")} if (RUNS / "E3P2.jsonl").exists() else {}
    e8 = {r["frames_dir"]: r for r in map(json.loads, open(RUNS / "E8.jsonl", encoding="utf-8")) if r.get("parsed")} if (RUNS / "E8.jsonl").exists() else {}
    edits = {r["frames_dir"]: r for r in map(json.loads, open(OUT / "edits_test.jsonl", encoding="utf-8"))} if (OUT / "edits_test.jsonl").exists() else {}
    rows, miss = [], []
    for w in wins:
        fd = w["frames_dir"]
        a, b = e3.get(fd), e8.get(fd)
        agree = a is not None and (a["parsed"]["collision"] == "yes") == bool(w["label"])
        if agree:
            p, src = a["parsed"], "blind"
            pred = (p2.get(fd) or {}).get("prediction") or p.get("prediction")
            hv = None
        elif b is not None:
            p, src, pred, hv = b["parsed"], "second_look", b["parsed"].get("prediction"), b["parsed"].get("hazard_visible")
        else:
            miss.append(fd)
            continue
        row = {"frames_dir": fd, "video_id": w["video_id"], "label": w["label"], "tte": w["horizon"], "source": src, "hazard_visible": hv,
               "verdict": p["collision"], "risk": p["risk_score"], "explanation": p["explanation"], "prediction": pred,
               "blind_verdict": (a or {}).get("parsed", {}).get("collision"), "problems": (b or a)["problems"], "edited": False}
        if fd in edits:
            e = edits[fd]
            row.update(explanation=e["explanation"], prediction=e["prediction"], verdict=e.get("verdict", row["verdict"]), source=src + "+edited", edited=True)
        rows.append(row)
    (OUT / "dataset_test_final.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    import collections
    print(f"[final] {len(rows)} of {len(wins)} windows; sources {dict(collections.Counter(r['source'] for r in rows))}; "
          f"hazard_visible {dict(collections.Counter(r['hazard_visible'] for r in rows if r['hazard_visible']))}; missing {len(miss)}")
    if miss:
        print("        missing:", miss[:8])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e3", "e8", "final"])
    ap.add_argument("--split", choices=["private", "public"], default="private", help="test half (read at import time: sets the folder)")
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--tier", choices=["standard", "flex"], default="flex")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--ids", default="", help="pilot: comma-separated video ids to include first (e.g. the user-reviewed windows)")
    ap.add_argument("--reask", type=int, default=2, help="extra asks when an answer fails validation")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    final(a) if a.cmd == "final" else run(a)

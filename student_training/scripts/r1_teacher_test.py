"""
r1_teacher_test.py - teacher reasoning for the Nexar PRIVATE test windows (plan 2026-10-17_Plan-Test-Private-Reasoning, steps 4, 5, 9).
Gemini 3.8 Flash on OpenRouter, Flex tier. Everything lives in outputs/teacher_test_2026-10 (R1_PILOT_OUT is set to it, so r1_add_prediction.py works on it).

  e3     step 4: the BLIND first look on every boxed test window: the same E3 prompt as the training run (PROMPT_TEACHER_STUDY, byte-identical), the 16 boxed
         frames at native resolution, temperature 0.1.  -> runs/E3.jsonl     then:  python r1_add_prediction.py --run E3 ...  (the outcome sentence, text only)
  e8     step 5: the label-informed second look on the windows where the blind verdict != the Nexar label (or the blind answer is missing): PROMPT_TEACHER_TEST_E8,
         the same boxed frames + the true outcome in text; the answer says whether the hazard is visible.      -> runs/E8.jsonl
         --pilot N : only N random mismatching windows (seed 0), written to runs/E8_pilot.jsonl (the 20-window gate before the full pass)
  final  step 9: dataset_test_final.jsonl, one row per window: blind answer where it agrees with the label, else the second look (source, hazard_visible,
         problems), hand edits from edits_test.jsonl if present.

    python r1_teacher_test.py e3 --model google/gemini-3.8-flash --tier flex [--limit N] [--dry-run]
    python r1_teacher_test.py e8 --model google/gemini-3.8-flash --tier flex [--pilot 20] [--dry-run]
    python r1_teacher_test.py final
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
OUT = ROOT / "outputs" / "teacher_test_2026-10"
os.environ.setdefault("R1_PILOT_OUT", str(OUT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from prompts.PROMPT_TEACHER_STUDY import build_prompt as build_study_prompt  # noqa: E402
from prompts.PROMPT_TEACHER_TEST_E8 import REQUIRED_E8, build_e8_prompt  # noqa: E402
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
            rows = random.Random(0).sample(rows, min(args.pilot, len(rows)))
        prompt_of = lambda w: build_e8_prompt(w["label"], w["horizon"])        # noqa: E731
        check = lambda p, w: validate_e8(p)                                    # noqa: E731
        variant = "e8"
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
        res = call(client, args.model, _build_messages(prompt_of(w), imgs, "high"), args.tier, args.timeout, args.retries, args.temperature)
        return w, res

    t0, n_ok, n_fail, cost = time.time(), 0, 0, 0.0
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as pool:
        for k, fut in enumerate(as_completed([pool.submit(work, w) for w in todo]), 1):
            w, res = fut.result()
            parsed = _extract_json_object(res["text"]) if res["text"] else None
            probs = ["call failed: " + str(res["error"])] if res["error"] is not None else check(parsed, w)
            c = res["usage"].get("cost")
            cost += float(c) if c is not None else 0.0
            rec = {"run": name, "set": w["set"], "frames_dir": w["frames_dir"], "video_id": w["video_id"], "label": w["label"], "horizon": w["horizon"],
                   "tier_requested": args.tier, "served_tier": res["served_tier"], "wall_s": round(res["wall_s"], 2), "attempts": res["attempts"],
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
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--tier", choices=["standard", "flex"], default="flex")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    final(a) if a.cmd == "final" else run(a)

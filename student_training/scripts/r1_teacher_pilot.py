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
from semsup_caption_promptbakeoff import _build_messages, _encode_image, _extract_json_object  # noqa: E402

OUT = ROOT / "outputs" / "teacher_pilot_2026-10"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--set", choices=["A", "B"])
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

    wins = [json.loads(l) for l in open(OUT / "windows.jsonl", encoding="utf-8")]
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(OUT / "boxes.jsonl", encoding="utf-8")}
    if args.flex_test:
        args.set, args.tier, args.name, args.limit = "A", "flex", "flex_test", 1
    rows = [w for w in wins if w["set"] == args.set and w["frames_dir"] in boxes][: args.limit or None]
    prompt = build_prompt(args.order)
    print(f"[cfg] run={args.name} set={args.set} tier={args.tier} order={args.order} model={args.model} windows={len(rows)} "
          f"concurrency={args.concurrency}", flush=True)
    if args.dry_run:
        print(prompt)
        return
    client = make_client()
    runs = OUT / "runs"
    runs.mkdir(exist_ok=True)
    out_path = runs / f"{args.name}.jsonl"
    done = {json.loads(l)["frames_dir"] for l in open(out_path, encoding="utf-8")} if out_path.exists() and not args.flex_test else set()
    todo = [w for w in rows if w["frames_dir"] not in done]
    print(f"[cfg] {len(done)} already done, {len(todo)} to run", flush=True)

    def work(w):
        paths = sorted((BOXED / w["frames_dir"]).glob("frame_*.jpg"))
        assert len(paths) == 16, w["frames_dir"]
        imgs = [_encode_image(p, 0) for p in paths]                       # native resolution (V12 used native)
        res = call(client, args.model, _build_messages(prompt, imgs, "high"), args.tier, args.timeout, args.retries, args.temperature)
        return w, res

    t0 = time.time()
    n_ok = n_fail = 0
    cost = 0.0
    with open(out_path, "a" if not args.flex_test else "w", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as pool:
        futs = [pool.submit(work, w) for w in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            w, res = fut.result()
            parsed = _extract_json_object(res["text"]) if res["text"] else None
            probs = validate(parsed, args.order) if res["error"] is None else ["call failed: " + str(res["error"])]
            c = res["usage"].get("cost")
            cost += float(c) if c is not None else 0.0
            rec = {"run": args.name, "set": w["set"], "frames_dir": w["frames_dir"], "video_id": w["video_id"], "label": w["label"],
                   "horizon": w["horizon"], "tier_requested": args.tier, "served_tier": res["served_tier"], "order": args.order,
                   "wall_s": round(res["wall_s"], 2), "attempts": res["attempts"], "usage": res["usage"], "model_returned": res.get("model_returned"),
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

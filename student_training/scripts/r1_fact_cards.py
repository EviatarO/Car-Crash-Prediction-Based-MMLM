"""
r1_fact_cards.py - fact cards: the 4 facts of a reasoning text in CLOSED lists, read by Gemini 3.8 Flash (OpenRouter Flex, text only).
Plan 2026-10-15_Plan-Phase2-Three-Formats, step 1. The same reader fills a card for the teacher's text (-> the Facts line of the facts-first answer)
and for a model's generated text (-> fact grading: card(model) vs card(teacher), replacing the word-list grader of r1_fact_eval.py).

  road_user : car (sedan, hatchback, coupe, wagon, taxi ...) | suv | pickup | van | truck (incl. trailer, bus-size lorries) | bus | motorcycle |
              cyclist (bicycle, scooter rider) | pedestrian | object (sign, cone, barrier, debris, pole, animal) | not_stated
  side      : AT THE END of the clip: ahead (same lane / directly in the ego path) | left | right | behind | not_stated
  motion    : MAIN manoeuvre of that road user DURING the clip, not of the ego vehicle: steady | braking (slows, decelerates) | stopped (stationary,
              parked) | turning | crossing (moves across the ego path) | cutting_in (merges / swerves into the ego lane) | reversing | not_stated
              (a manoeuvre wins over braking, braking over stopped / steady)
  gap       : trend of the distance ego <-> that road user over the LAST part of the clip: closing | widening | stable | not_stated
  (v2, 2026-10-15: slots tied to a fixed moment because the scene is dynamic; v1 cards kept as *_v1.jsonl)

    python r1_fact_cards.py --source teacher                      # 4,169 teacher texts -> outputs/teacher_dataset_2026-10/fact_cards_teacher.jsonl
    python r1_fact_cards.py --source rephrased --n 500            # random rephrased versions -> fact_cards_rephrased.jsonl (consistency check)
    python r1_fact_cards.py --source model --pred <predictions.jsonl> --out <cards.jsonl>   # a run's generated explanations
    python r1_fact_cards.py --source teacher --dry-run            # no API call: prints one request + cost estimate
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = ROOT / "outputs" / "teacher_dataset_2026-10"
MAN = ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl"
if __import__("os").environ.get("R1_REASONING_MANIFEST"):          # plan 2026-10-19: the manifest of the extra windows
    MAN = Path(__import__("os").environ["R1_REASONING_MANIFEST"])
SLOTS = {
    "road_user": ["car", "suv", "pickup", "van", "truck", "bus", "motorcycle", "cyclist", "pedestrian", "object", "not_stated"],
    "side": ["ahead", "left", "right", "behind", "not_stated"],
    "motion": ["steady", "braking", "stopped", "turning", "crossing", "cutting_in", "reversing", "not_stated"],
    "gap": ["closing", "widening", "stable", "not_stated"],
}
IN_PER_M, OUT_PER_M, FLEX = 0.75, 3.75, 0.5

SYSTEM = """You read short descriptions of 2-second dashcam clips and fill a fact card for each. The "ego vehicle" is the car with the camera.

First pick the MAIN road user: the one the description is about (the hazard, or the road user it focuses on). Then fill 4 slots, each with exactly one value from its list. Use only what the text states; never guess. If the text does not state a slot, use not_stated.

road_user: car | suv | pickup | van | truck | bus | motorcycle | cyclist | pedestrian | object | not_stated
  car includes sedan, hatchback, coupe, wagon, taxi, "vehicle" of unstated kind -> car only if the text says car/sedan/etc.; a vehicle of unstated kind -> not_stated.
  suv includes sport utility vehicle, crossover. truck includes lorry, semi, trailer, tractor-trailer, box truck, dump truck. cyclist includes bicycle or scooter rider.
  object = a sign, cone, barrier, pole, debris, animal or other non-vehicle.
The scene moves, so each slot refers to a FIXED moment:
- side and gap describe the END of the clip (the last frames, the moment a warning would be given);
- motion describes the MAIN manoeuvre of the road user DURING the clip (this keeps where it came from).
Example: a car on the right overtakes, moves into the ego lane and speeds away -> side = ahead (in the ego lane at the end), motion = cutting_in, gap = widening.
The same car moving into the ego lane and then braking -> side = ahead, motion = cutting_in, gap = closing.

side (where the main road user is relative to the ego vehicle AT THE END of the clip): ahead | left | right | behind | not_stated
  ahead = in the ego lane / directly in the ego path / directly in front. left / right = to that side (adjacent lane, shoulder, curb, cross street from that side).
motion (MAIN manoeuvre of the main road user during the clip, NOT of the ego vehicle): steady | braking | stopped | turning | crossing | cutting_in | reversing | not_stated
  steady = moves along normally; braking = slows / decelerates; stopped = stationary / parked / waiting; turning = turns; crossing = moves across the ego path;
  cutting_in = merges, swerves or pulls into the ego lane / path; reversing = backs up.
  Priority when several apply: cutting_in / crossing / turning / reversing (a manoeuvre) over braking over stopped / steady.
gap (trend of the distance between the ego vehicle and the main road user over the LAST part of the clip): closing | widening | stable | not_stated
  closing = the distance shrinks / is reduced / they approach; widening = grows / pulls away; stable = stays the same / steady spacing / clear separation maintained.
  If the distance first shrinks and then grows (e.g. it is overtaken and pulls away), use the trend at the end.

Reply with ONLY JSON lines, one per item, same order, no markdown:
{"id": "<id>", "road_user": "...", "side": "...", "motion": "...", "gap": "..."}"""


def user_msg(rows):
    return "Items:\n\n" + "\n\n".join(f'id: {r["key"]}\nTEXT: {r["text"]}' for r in rows)


def call(client, model, rows, tier, timeout, retries):
    kw = {"model": model, "temperature": 0, "timeout": timeout,
          "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(rows)}],
          "extra_body": {"reasoning": {"effort": "low"}, **({"service_tier": "flex"} if tier == "flex" else {})}}
    last = None
    for a in range(1, retries + 1):
        try:
            r = client.chat.completions.create(**kw)
            return r.choices[0].message.content or "", (r.usage.model_dump() if getattr(r, "usage", None) else {}), None
        except Exception as e:                                                       # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 4 * a * a))
    return "", {}, last


def parse(txt, keys):
    out = {}
    for l in txt.splitlines():
        l = l.strip().strip("`")
        if not l.startswith("{"):
            continue
        try:
            x = json.loads(l)
        except json.JSONDecodeError:
            continue
        if x.get("id") in keys and all(x.get(s) in v for s, v in SLOTS.items()):
            out[x["id"]] = {s: x[s] for s in SLOTS}
    return out


def items(a):
    if a.source == "teacher":
        return [{"key": r["id"], "id": r["id"], "text": r["gt"]["explanation"]} for r in map(json.loads, open(MAN, encoding="utf-8"))], \
            Path(a.out or D / "fact_cards_teacher.jsonl")
    if a.source == "rephrased":
        rows = [json.loads(l) for l in open(D / "paraphrases_judged.jsonl", encoding="utf-8")]
        random.Random(0).shuffle(rows)
        return [{"key": f'{r["id"]}#{r["source"]}', "id": r["id"], "src": r["source"], "text": r["explanation"]} for r in rows[:a.n]], \
            Path(a.out or D / "fact_cards_rephrased.jsonl")
    rows = [json.loads(l) for l in open(a.pred, encoding="utf-8")]
    return [{"key": r["id"], "id": r["id"], "text": r.get("gen_explanation") or ""} for r in rows], Path(a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["teacher", "rephrased", "model"], required=True)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--tier", choices=["standard", "flex"], default="flex")
    ap.add_argument("--per-request", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows, out = items(a)
    done = {json.loads(l)["key"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in rows if r["key"] not in done and r["text"].strip()]
    chunks = [todo[i:i + a.per_request] for i in range(0, len(todo), a.per_request)]
    t_in = sum(len(SYSTEM) / 3 + len(user_msg(c)) / 3 for c in chunks)
    t_out = len(todo) * 45
    print(f"[cards] {a.source}: {len(todo)} texts to read ({len(done)} done) in {len(chunks)} requests; est ${(t_in * IN_PER_M + t_out * OUT_PER_M) / 1e6 * (FLEX if a.tier == 'flex' else 1):.2f}")
    if a.dry_run or not chunks:
        if chunks:
            print("\n===== SYSTEM =====\n" + SYSTEM + "\n\n===== USER =====\n" + user_msg(chunks[0][:3]))
        return
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    by_key = {r["key"]: r for r in todo}
    cost, got, miss, t0 = 0.0, 0, 0, time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as pool:
        futs = {pool.submit(call, client, a.model, c, a.tier, a.timeout, a.retries): c for c in chunks}
        for k, fu in enumerate(as_completed(futs), 1):
            c = futs[fu]
            txt, u, err = fu.result()
            cost += float(u.get("cost") or 0)
            res = parse(txt, {r["key"] for r in c})
            for key, card in res.items():
                r = by_key[key]
                f.write(json.dumps({"key": key, "id": r["id"], **({"src": r["src"]} if "src" in r else {}), **card}, ensure_ascii=False) + "\n")
            f.flush()
            got += len(res)
            miss += len(c) - len(res)
            if k % 20 == 0 or k == len(chunks):
                print(f"[cards] {k}/{len(chunks)} requests; read {got}, unreadable {miss}; ${cost:.3f}; {time.time() - t0:.0f}s" + (f"; last error {err}" if err else ""), flush=True)
    print(f"[cards] done: {got} cards, {miss} missing (rerun to retry); ${cost:.3f} -> {out}")


if __name__ == "__main__":
    main()

"""
r1_add_prediction.py - (re)write the one-sentence `prediction` of existing teacher answers WITHOUT regenerating their explanations.

The final targets end with a prediction: the OUTCOME only ("A collision with the truck follows." / "No collision; the road stays clear."),
4-15 words, not repeating the explanation (which is already above it). E3 answers (stage 1, most of the dataset) have no prediction;
E6 / E7 answers have an older, longer one. This sends a TEXT-ONLY call per answer (no images): the existing explanation + verdict
+ risk_score, and asks for the outcome clause only. One retry if the clause is longer than 15 words or overlaps the explanation
(token-F1 > 0.5). The explanation itself is never touched.
Output: <R1_PILOT_OUT or outputs/teacher_pilot_2026-10>/runs/<run>P2.jsonl  ({frames_dir, prediction, usage, problems}); resumable.

    python r1_add_prediction.py --run E3 --model google/gemini-3.8-flash --tier flex      # also: --run E6 / E7 / E6C
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from r1_phase_report import token_f1  # noqa: E402
from r1_teacher_pilot import OUT, call, make_client  # noqa: E402
from semsup_caption_promptbakeoff import _extract_json_object  # noqa: E402

PROMPT = (
    "A driving-safety reviewer watched a 2-second dashcam clip from a car (the \"ego vehicle\") and wrote the description below, "
    "followed by their final judgement.\n\n"
    "DESCRIPTION: \"{explanation}\"\n"
    "JUDGEMENT: collision = \"{collision}\" (risk_score {risk} of 100)\n\n"
    "Write ONLY the outcome, in one short clause of 4-15 words. Do not repeat or summarise the description -- it is already written "
    "above. Name the road user involved if there is one. Examples of the form: \"A collision with the truck follows.\" / "
    "\"No collision; the road stays clear.\" Rules: use only facts that are in the description (do not add road users, colours or "
    "times); the clause must agree with the judgement; outcome words are allowed; do not copy the examples.\n\n"
    "Return ONLY this JSON, no markdown fences: {{\"prediction\": \"outcome clause\"}}"
)
RETRY = ("\n\nYour previous clause was \"{prev}\". It was too long or repeated the description. Answer again with at most 12 words, "
         "naming only the outcome.")


def check(pred, explanation):
    n = len(str(pred).split()) if isinstance(pred, str) else 0
    pr = []
    if not 4 <= n <= 15:
        pr.append(f"prediction words={n}")
    if isinstance(pred, str) and token_f1(pred, explanation) > 0.5:
        pr.append(f"overlap with explanation F1={token_f1(pred, explanation):.2f}")
    return pr


def one(client, args, r):
    ex = r["parsed"]["explanation"]
    base = PROMPT.format(explanation=ex, collision=r["parsed"]["collision"], risk=r["parsed"]["risk_score"])
    usage, pred, probs = [], None, []
    for attempt in range(2):
        text = base if attempt == 0 else base + RETRY.format(prev=pred)
        res = call(client, args.model, [{"role": "user", "content": text}], args.tier, 300.0, 3, 0.1)
        usage.append(res["usage"])
        p = _extract_json_object(res["text"]) if res["text"] else None
        pred = (p or {}).get("prediction")
        probs = check(pred, ex) + (["call failed: " + str(res["error"])] if res["error"] else [])
        if not probs:
            break
    return pred, probs, usage, res["served_tier"], attempt + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="E3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", choices=["standard", "flex"], default="flex")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    src = [r for r in map(json.loads, open(OUT / "runs" / f"{args.run}.jsonl", encoding="utf-8")) if r.get("parsed")]
    out_path = OUT / "runs" / f"{args.run}P2.jsonl"
    done = {json.loads(l)["frames_dir"] for l in open(out_path, encoding="utf-8")} if out_path.exists() else set()
    todo = [r for r in src if r["frames_dir"] not in done]
    print(f"[cfg] {args.run}: {len(src)} answers, {len(done)} done, {len(todo)} to run, tier={args.tier}", flush=True)
    if args.dry_run:
        r = todo[0] if todo else src[0]
        print(PROMPT.format(explanation=r["parsed"]["explanation"], collision=r["parsed"]["collision"], risk=r["parsed"]["risk_score"]))
        return
    client = make_client()
    t0, cost, n_bad = time.time(), 0.0, 0
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as pool:
        futs = {pool.submit(one, client, args, r): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            pred, probs, usage, tier, attempts = fut.result()
            cost += sum(float((u or {}).get("cost") or 0) for u in usage)
            n_bad += bool(probs)
            f.write(json.dumps({"frames_dir": r["frames_dir"], "label": r["label"], "collision": r["parsed"]["collision"],
                                "prediction": pred, "served_tier": tier, "attempts": attempts, "problems": probs,
                                "usage": usage}, ensure_ascii=False) + "\n")
            f.flush()
    print(f"[done] {len(todo)} answers, {n_bad} still with problems, ${cost:.4f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

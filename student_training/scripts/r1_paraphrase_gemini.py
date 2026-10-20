"""
r1_paraphrase_gemini.py - finish the 3-rewordings-per-window text augmentation with Gemini on OpenRouter (Flex tier), TEXT ONLY (no images, no video).

For each window not yet in paraphrase_out/ (or listed in --ids), one request carries a few windows (default 5): the teacher's explanation, its outcome
sentence, the verdict and the facts to keep. Gemini returns 3 rewordings of each (explanation + outcome). Output lines have the same format the
Claude agents wrote ({"id", "versions": [{"explanation", "outcome"} x3]}) -> paraphrase_out/gemini_NNN.jsonl, then r1_paraphrase_check.py check
validates them exactly like the others (length, no outcome word, type/side sets, polarity, near copy, risky new words -> flags for the Opus verifier).

    python r1_paraphrase_gemini.py --dry-run                 # no API call: prints one full request + token / cost estimate
    python r1_paraphrase_gemini.py --limit 10 --concurrency 2 # tiny live test
    python r1_paraphrase_gemini.py --concurrency 24           # the rest
    python r1_paraphrase_gemini.py --rewrite                  # windows with < 3 accepted versions (uses paraphrase_rejects.jsonl as 'avoid these' hints)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
D = Path(__import__("os").environ["R1_PARA_DIR"]) if __import__("os").environ.get("R1_PARA_DIR") else ROOT / "outputs" / "teacher_dataset_2026-10"   # R1_PARA_DIR: a separate folder for a new run (plan 2026-10-19); unset = the original folder
BATCHES, OUTS = D / "paraphrase_batches", D / "paraphrase_out"
IN_PER_M, OUT_PER_M, FLEX = 0.75, 3.75, 0.5            # USD per million tokens (OpenRouter list price, google/gemini-3.8-flash); Flex = half price

SYSTEM = """You rewrite short dashcam-scene descriptions. Each item has an original EXPLANATION (25-45 words) and an OUTCOME sentence (at most 15 words), written by someone who watched a 2-second dashcam clip. The "ego vehicle" is the car carrying the camera.

For every item write 3 NEW versions (explanation + outcome). Write each one from scratch in your own words; do not edit the original sentence by swapping a few words.

KEEP IDENTICAL (these are facts from the video, you cannot see the video):
- the road user(s): the same kind of vehicle or person. A synonym of the same kind is fine ("sedan" <-> "car", "SUV" <-> "sport utility vehicle"); a different kind never ("car" -> "truck" is wrong).
- the side / position: left, right, ahead, same lane, adjacent lane, oncoming, crossing. The same position may be worded differently ("to the left" <-> "in the left lane").
- the motion (braking, turning, crossing, stopped, drifting, overtaking ...) and the trend of the gap (closing, widening, constant).
- colours, places and objects ONLY if the original mentions them. Never add a colour, place, road marking, weather, traffic light, building or object that the original does not mention. Never drop a fact that the original states.
- the outcome sentence keeps its meaning: if the original outcome says a collision happens, yours says a collision happens; if it says no collision, yours says no collision.
- the exact state words: "stopped" / "stationary" stays stopped (never "parked"); "slows" stays slows (never "brakes hard"); "drifts" stays drifts (never "swerves"). Do not make anything faster, harder, closer or more sudden than the original says, and do not soften it either. Do not add a cause or an intention.
- always say where the road user is (ahead, same lane, left, right, adjacent lane, oncoming ...) as the original does.
- do not name the type of collision (no "rear-end", "side-swipe") unless the original does.

VARY: sentence structure (clause order, active / passive, one long sentence vs two or three short ones), vocabulary, and which fact comes first. The 3 versions must differ clearly from each other as well as from the original.

HARD RULES for the EXPLANATION:
- 25 to 45 words.
- It must NOT contain any of the words: collide, collision, crash, contact, impact, hit, strike, struck (the outcome sentence may).
- Plain statement of what is visible; no speculation beyond the original.

The OUTCOME is one sentence of at most 15 words.

Reply with ONLY JSON lines, one per item, no markdown, in the same order:
{"id": "<id>", "versions": [{"explanation": "...", "outcome": "..."}, {"explanation": "...", "outcome": "..."}, {"explanation": "...", "outcome": "..."}]}"""


def item_text(r):
    facts = ", ".join(f"{k}={v}" for k, v in (r.get("keep_facts") or {}).items()) or "(none detected - keep every detail the explanation states)"
    s = f'id: {r["id"]}\nverdict: {r["verdict"]}\nfacts to keep: {facts}\nEXPLANATION: {r["explanation"]}\nOUTCOME: {r["outcome"]}'
    for b in r.get("avoid", [])[:3]:                                                 # rewrite rounds: earlier versions that were judged wrong, and why
        s += f'\nEARLIER WRONG VERSION (do not repeat its mistake: {b["problem"]}): {b["explanation"]}'
    return s


def user_msg(rows):
    return "Items:\n\n" + "\n\n".join(item_text(r) for r in rows)


def done_ids():
    s = set()
    for p in OUTS.glob("*.jsonl"):
        for l in open(p, encoding="utf-8"):
            try:
                s.add(json.loads(l)["id"])
            except Exception:
                pass
    return s


def todo_rows(rewrite):
    rows = [json.loads(l) for p in sorted(BATCHES.glob("batch_*.jsonl")) for l in open(p, encoding="utf-8")]
    if not rewrite:
        d = done_ids()
        return [r for r in rows if r["id"] not in d]
    # rewrite: windows with < 3 versions that passed BOTH the automatic check and the judge (r1_paraphrase_judge.py finalize writes the counts)
    good = json.loads((D / "paraphrase_good_counts.json").read_text(encoding="utf-8"))
    avoid = {}
    for p in ("paraphrase_rejects.jsonl", "paraphrase_judge_rejects.jsonl"):
        if (D / p).exists():
            for l in open(D / p, encoding="utf-8"):
                x = json.loads(l)
                avoid.setdefault(x["id"], []).append({"explanation": x["explanation"], "problem": "; ".join(x.get("problems", [])) or x.get("problem", "")})
    need = [{**r, "avoid": avoid.get(r["id"], [])[-3:]} for r in rows if good.get(r["id"], 0) < 3]
    return sorted(need, key=lambda r: good.get(r["id"], 0))                         # fewest good versions first (a budget cap keeps the neediest)


def est_tokens(s):
    return int(len(s) / 4 + 1)


def call(client, model, rows, tier, timeout, retries):
    kw = {"model": model, "temperature": 0.4, "timeout": timeout,
          "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(rows)}],
          "extra_body": {"reasoning": {"effort": "low"}, **({"service_tier": "flex"} if tier == "flex" else {})}}
    last = None
    for a in range(1, retries + 1):
        try:
            r = client.chat.completions.create(**kw)
            txt = r.choices[0].message.content or ""
            u = r.usage.model_dump() if getattr(r, "usage", None) else {}
            return txt, u, None
        except Exception as e:                                                       # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 4 * a * a))
    return "", {}, last


def parse(txt, want):
    out = {}
    for l in txt.splitlines():
        l = l.strip().strip("`")
        if not l.startswith("{"):
            continue
        try:
            x = json.loads(l)
        except json.JSONDecodeError:
            continue
        if x.get("id") in want and isinstance(x.get("versions"), list) and len(x["versions"]) >= 3 and all(
                isinstance(v, dict) and v.get("explanation") and v.get("outcome") for v in x["versions"][:3]):
            out[x["id"]] = {"id": x["id"], "versions": [{"explanation": v["explanation"].strip(), "outcome": v["outcome"].strip()} for v in x["versions"][:3]]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemini-3.8-flash")
    ap.add_argument("--tier", choices=["standard", "flex"], default="flex")
    ap.add_argument("--per-request", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--rewrite", action="store_true")
    ap.add_argument("--tag", default="gemini")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = todo_rows(a.rewrite)
    if a.limit:
        rows = rows[:a.limit]
    chunks = [rows[i:i + a.per_request] for i in range(0, len(rows), a.per_request)]
    t_in = sum(est_tokens(SYSTEM) + est_tokens(user_msg(c)) for c in chunks)
    t_out = len(rows) * 330                                                          # 3 x (about 55 + 20 tokens) + JSON + a little thinking
    price = (t_in * IN_PER_M + t_out * OUT_PER_M) / 1e6 * (FLEX if a.tier == "flex" else 1)
    print(f"[plan] windows to do: {len(rows)}; requests: {len(chunks)} ({a.per_request} windows each); est tokens in {t_in:,} / out {t_out:,}; "
          f"est cost {a.tier}: ${price:.2f} (standard price ${price / (FLEX if a.tier == 'flex' else 1):.2f})")
    if a.dry_run or not chunks:
        if chunks:
            print("\n===== SYSTEM =====\n" + SYSTEM + "\n\n===== USER (first request) =====\n" + user_msg(chunks[0]))
        return
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    OUTS.mkdir(parents=True, exist_ok=True)
    out_path = OUTS / f"{a.tag}_{'rewrite' if a.rewrite else 'main'}.jsonl"
    usage = {"p": 0, "c": 0, "cost": 0.0}
    failed, got, t0 = [], 0, time.time()
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.concurrency) as pool:
        futs = {pool.submit(call, client, a.model, c, a.tier, a.timeout, a.retries): c for c in chunks}
        for k, fu in enumerate(as_completed(futs), 1):
            c = futs[fu]
            txt, u, err = fu.result()
            usage["p"] += u.get("prompt_tokens", 0)
            usage["c"] += u.get("completion_tokens", 0)
            usage["cost"] += float(u.get("cost") or 0)
            ok = parse(txt, {r["id"] for r in c})
            for x in ok.values():
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.flush()
            got += len(ok)
            failed += [r["id"] for r in c if r["id"] not in ok]
            if k % 10 == 0 or k == len(chunks):
                print(f"[run] {k}/{len(chunks)} requests; windows ok {got}; missing {len(failed)}; tokens in {usage['p']:,} out {usage['c']:,}; "
                      f"cost so far ${usage['cost']:.3f}; {time.time() - t0:.0f}s" + (f"; last error: {err}" if err else ""), flush=True)
    (D / f"gemini_missing_{'rewrite' if a.rewrite else 'main'}.txt").write_text("\n".join(failed), encoding="utf-8")
    print(f"[done] windows ok {got}, missing {len(failed)} (rerun to retry them); cost ${usage['cost']:.3f}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
